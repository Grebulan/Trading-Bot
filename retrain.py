#!/usr/bin/env python3
"""
Auto retrain script for stock and crypto models.
Runs every Sunday night at 2am.
"""

import os
import json
import time
import joblib
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(
    filename='/home/jv/retrain.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)
log = logging.getLogger()

def p(msg):
    print(msg)
    log.info(msg)

STOCK_TICKERS = [
    'AAPL','GOOGL','MSFT','META','AMZN','TSLA','NVDA',
    'JPM','BAC','GS','PFE','MRK','RIVN','NIO',
    'WMT','TGT','AMD','QCOM','MS'
]

CRYPTOS = {
    'BTC/USD':  'BTC-USD',
    'ETH/USD':  'ETH-USD',
    'SOL/USD':  'SOL-USD',
    'LINK/USD': 'LINK-USD',
    'AVAX/USD': 'AVAX-USD',
    'XRP/USD':  'XRP-USD',
    'DOGE/USD': 'DOGE-USD',
    'UNI/USD':  'UNI7083-USD',
    'DOT/USD':  'DOT-USD',
    'ADA/USD':  'ADA-USD',
}

SECTORS = {
    'Tech':    ['AAPL','GOOGL','MSFT','META','AMZN','NVDA','AMD','QCOM'],
    'EV':      ['TSLA','RIVN','NIO'],
    'Finance': ['JPM','BAC','GS','MS'],
    'Pharma':  ['PFE','MRK'],
    'Retail':  ['WMT','TGT'],
}
TICKER_SECTOR = {t: s for s, tickers in SECTORS.items() for t in tickers}

STOCK_FEATURES = [
    'SMA_10','SMA_30','SMA_50','EMA_12','EMA_26','MACD','RSI',
    'Return','Return_5d','Return_10d','Volatility','High_Low',
    'BB_position','BB_width','Vol_Ratio','Vol_Spike','DayOfWeek',
    'Sector_Momentum','Sector_Momentum_5d',
    'VIX','SPY_Return','SPY_Trend','SPY_Return_5d',
    'QQQ_Return','QQQ_Trend','QQQ_Return_5d',
]

# Removed noise/calendar features: High_Low, HourOfDay, DayOfWeek, IsWeekend, Overnight_Gap
# These were dominating the model (High_Low alone was 11.8% importance) without
# adding directional signal, causing a strong bearish bias.
# Target lowered from 1% to 0.5% in 6h — more achievable in sideways/recovering markets.
CRYPTO_FEATURES = [
    'SMA_24','SMA_72','SMA_168',
    'EMA_12','EMA_26','MACD','RSI',
    'Return','Return_24h','Return_72h',
    'Volatility',
    'BB_position','BB_width',
    'Vol_Ratio','Vol_Spike',
    'VIX',
    'SPY_Return','SPY_Trend','SPY_Return_24h',
    'BTC_Return','BTC_Trend','BTC_Return_24h',
    'Crypto_Momentum','Crypto_Momentum_24h',
]

def build_base_features(df):
    df['SMA_10']       = df['Close'].rolling(10).mean()
    df['SMA_30']       = df['Close'].rolling(30).mean()
    df['SMA_50']       = df['Close'].rolling(50).mean()
    df['EMA_12']       = df['Close'].ewm(span=12).mean()
    df['EMA_26']       = df['Close'].ewm(span=26).mean()
    df['MACD']         = df['EMA_12'] - df['EMA_26']
    df['RSI']          = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() /
                                           df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
    df['Return']       = df['Close'].pct_change()
    df['Return_5d']    = df['Close'].pct_change(5)
    df['Return_10d']   = df['Close'].pct_change(10)
    df['Volatility']   = df['Return'].rolling(10).std()
    df['High_Low']     = (df['High'] - df['Low']) / df['Close']
    bb_mid             = df['Close'].rolling(20).mean()
    bb_std             = df['Close'].rolling(20).std()
    df['BB_upper']     = bb_mid + (2 * bb_std)
    df['BB_lower']     = bb_mid - (2 * bb_std)
    df['BB_position']  = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['BB_width']     = (df['BB_upper'] - df['BB_lower']) / bb_mid
    df['Volume_MA']    = df['Volume'].rolling(10).mean()
    df['Vol_Ratio']    = df['Volume'] / df['Volume_MA']
    df['Vol_Spike']    = (df['Vol_Ratio'] > 2.0).astype(int)
    df['DayOfWeek']    = pd.to_datetime(df.index).dayofweek
    return df

def get_market_indicators_stock():
    vix = yf.download('^VIX', period="15y", interval="1d", progress=False)
    vix.columns = [col[0] if isinstance(col, tuple) else col for col in vix.columns]
    vix = vix[['Close']].rename(columns={'Close': 'VIX'})

    spy = yf.download('SPY', period="15y", interval="1d", progress=False)
    spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
    spy = spy[['Close']].rename(columns={'Close': 'SPY'})
    spy['SPY_Return']    = spy['SPY'].pct_change()
    spy['SPY_SMA_10']    = spy['SPY'].rolling(10).mean()
    spy['SPY_SMA_50']    = spy['SPY'].rolling(50).mean()
    spy['SPY_Trend']     = (spy['SPY_SMA_10'] > spy['SPY_SMA_50']).astype(int)
    spy['SPY_Return_5d'] = spy['SPY'].pct_change(5)

    qqq = yf.download('QQQ', period="15y", interval="1d", progress=False)
    qqq.columns = [col[0] if isinstance(col, tuple) else col for col in qqq.columns]
    qqq = qqq[['Close']].rename(columns={'Close': 'QQQ'})
    qqq['QQQ_Return']    = qqq['QQQ'].pct_change()
    qqq['QQQ_Trend']     = (qqq['QQQ'].rolling(10).mean() > qqq['QQQ'].rolling(50).mean()).astype(int)
    qqq['QQQ_Return_5d'] = qqq['QQQ'].pct_change(5)

    market = vix.join(spy[['SPY_Return','SPY_SMA_10','SPY_SMA_50',
                            'SPY_Trend','SPY_Return_5d']], how='inner')
    market = market.join(qqq[['QQQ_Return','QQQ_Trend','QQQ_Return_5d']], how='inner')
    market.dropna(inplace=True)
    return market

def get_market_indicators_crypto():
    vix = yf.download('^VIX', period="4y", interval="1d", progress=False)
    vix.columns = [col[0] if isinstance(col, tuple) else col for col in vix.columns]
    vix = vix[['Close']].rename(columns={'Close': 'VIX'})

    spy = yf.download('SPY', period="4y", interval="1d", progress=False)
    spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
    spy = spy[['Close']].rename(columns={'Close': 'SPY'})
    spy['SPY_Return']    = spy['SPY'].pct_change()
    spy['SPY_SMA_10']    = spy['SPY'].rolling(10).mean()
    spy['SPY_SMA_50']    = spy['SPY'].rolling(50).mean()
    spy['SPY_Trend']     = (spy['SPY_SMA_10'] > spy['SPY_SMA_50']).astype(int)
    spy['SPY_Return_5d'] = spy['SPY'].pct_change(5)

    btc = yf.download('BTC-USD', period="4y", interval="1d", progress=False)
    btc.columns = [col[0] if isinstance(col, tuple) else col for col in btc.columns]
    btc = btc[['Close']].rename(columns={'Close': 'BTC'})
    btc['BTC_Return']    = btc['BTC'].pct_change()
    btc['BTC_SMA_10']    = btc['BTC'].rolling(10).mean()
    btc['BTC_SMA_50']    = btc['BTC'].rolling(50).mean()
    btc['BTC_Trend']     = (btc['BTC_SMA_10'] > btc['BTC_SMA_50']).astype(int)
    btc['BTC_Return_5d'] = btc['BTC'].pct_change(5)

    market = vix.join(spy[['SPY_Return','SPY_Trend','SPY_Return_5d']], how='inner')
    market = market.join(btc[['BTC_Return','BTC_Trend','BTC_Return_5d']], how='inner')
    market.dropna(inplace=True)
    return market

# ============================================================
# STOCK MODEL — unchanged
# ============================================================
def retrain_stocks():
    p("\n" + "="*50)
    p("📈 RETRAINING STOCK MODEL")
    p("="*50)
    start = time.time()

    p("📥 Downloading stock data...")
    all_data = {}
    for ticker in STOCK_TICKERS:
        try:
            df = yf.download(ticker, period="15y", interval="1d", progress=False)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df = build_base_features(df)
            future_max   = df['Close'].rolling(3).max().shift(-3)
            df['Target'] = (future_max > df['Close'] * 1.01).astype(int)
            df.dropna(inplace=True)
            all_data[ticker] = df
            p(f"  {ticker}: {len(df):,} rows")
        except Exception as e:
            p(f"  {ticker}: ERROR — {e}")

    p("📥 Downloading market indicators...")
    market = get_market_indicators_stock()
    p(f"  Market: {len(market):,} rows")

    market_cols = ['VIX','SPY_Return','SPY_Trend','SPY_Return_5d',
                   'QQQ_Return','QQQ_Trend','QQQ_Return_5d']
    for ticker in list(all_data.keys()):
        all_data[ticker] = all_data[ticker].join(market[market_cols], how='inner')
        all_data[ticker].dropna(inplace=True)

    p("📊 Calculating sector momentum...")
    sector_returns = {}
    for sector, tickers in SECTORS.items():
        sector_df = pd.concat([
            all_data[t]['Return'].rename(t)
            for t in tickers if t in all_data
        ], axis=1)
        sector_returns[sector] = sector_df.mean(axis=1)

    for ticker in list(all_data.keys()):
        sector = TICKER_SECTOR[ticker]
        all_data[ticker]['Sector_Momentum']    = sector_returns[sector]
        all_data[ticker]['Sector_Momentum_5d'] = sector_returns[sector].rolling(5).mean()
        all_data[ticker].dropna(inplace=True)

    combined = pd.concat(all_data.values())
    X = combined[STOCK_FEATURES]
    y = combined['Target'].squeeze()
    p(f"✅ Combined: {len(combined):,} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    p("⚡ Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
        eval_metric='logloss', verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)
    acc = accuracy_score(y_test, model.predict(X_test))
    p(f"✅ Stock accuracy: {acc:.1%}")

    joblib.dump(model, '/home/jv/trading_model.joblib')
    with open('/home/jv/features.json', 'w') as f:
        json.dump(STOCK_FEATURES, f)
    with open('/home/jv/model_info.json', 'w') as f:
        json.dump({
            'trained_at':    datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model_type':    'XGBoost',
            'accuracy':      f"{acc:.1%}",
            'features':      len(STOCK_FEATURES),
            'training_rows': len(X_train),
            'tickers':       STOCK_TICKERS,
        }, f)

    elapsed = time.time() - start
    p(f"✅ Stock retrain complete in {elapsed:.0f}s")
    return acc

# ============================================================
# CRYPTO MODEL — rebuilt with fixes
# ============================================================
def retrain_crypto():
    p("\n" + "="*50)
    p("🪙 RETRAINING CRYPTO MODEL (hourly)")
    p("="*50)
    start = time.time()

    p("📥 Downloading hourly crypto data...")
    crypto_data = {}
    for symbol, yf_symbol in CRYPTOS.items():
        try:
            df = yf.download(yf_symbol, period="2y", interval="1h", progress=False)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df['SMA_24']      = df['Close'].rolling(24).mean()
            df['SMA_72']      = df['Close'].rolling(72).mean()
            df['SMA_168']     = df['Close'].rolling(168).mean()
            df['EMA_12']      = df['Close'].ewm(span=12).mean()
            df['EMA_26']      = df['Close'].ewm(span=26).mean()
            df['MACD']        = df['EMA_12'] - df['EMA_26']
            df['RSI']         = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() /
                                                  df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
            df['Return']      = df['Close'].pct_change()
            df['Return_24h']  = df['Close'].pct_change(24)
            df['Return_72h']  = df['Close'].pct_change(72)
            df['Volatility']  = df['Return'].rolling(24).std()
            bb_mid            = df['Close'].rolling(48).mean()
            bb_std            = df['Close'].rolling(48).std()
            df['BB_upper']    = bb_mid + (2 * bb_std)
            df['BB_lower']    = bb_mid - (2 * bb_std)
            df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
            df['BB_width']    = (df['BB_upper'] - df['BB_lower']) / bb_mid
            df['Volume_MA']   = df['Volume'].rolling(24).mean()
            df['Vol_Ratio']   = df['Volume'] / df['Volume_MA']
            df['Vol_Spike']   = (df['Vol_Ratio'] > 2.0).astype(int)
            # Target: +0.5% within 6h (was 1% — more achievable in recovering markets)
            future_max        = df['Close'].rolling(6).max().shift(-6)
            df['Target']      = (future_max > df['Close'] * 1.005).astype(int)
            df.dropna(inplace=True)
            df.index = pd.to_datetime(df.index).tz_convert('UTC')
            buy_rate = df['Target'].mean()
            crypto_data[symbol] = df
            p(f"  {symbol}: {len(df):,} rows  (BUY rate: {buy_rate:.1%})")
        except Exception as e:
            p(f"  {symbol}: ERROR — {e}")

    p("📥 Downloading hourly market indicators...")
    vix_d = yf.download('^VIX', period="2y", interval="1d", progress=False)
    vix_d.columns = [col[0] if isinstance(col, tuple) else col for col in vix_d.columns]
    vix_d = vix_d[['Close']].rename(columns={'Close': 'VIX'})
    vix_d.index = pd.to_datetime(vix_d.index).tz_localize('UTC')

    spy_h = yf.download('SPY', period="2y", interval="1h", progress=False)
    spy_h.columns = [col[0] if isinstance(col, tuple) else col for col in spy_h.columns]
    spy_h = spy_h[['Close']].rename(columns={'Close': 'SPY'})
    spy_h.index = pd.to_datetime(spy_h.index).tz_convert('UTC')
    spy_h = spy_h.resample('1h').last().ffill()
    spy_h['SPY_Return']     = spy_h['SPY'].pct_change()
    spy_h['SPY_SMA_24']     = spy_h['SPY'].rolling(24).mean()
    spy_h['SPY_SMA_168']    = spy_h['SPY'].rolling(168).mean()
    spy_h['SPY_Trend']      = (spy_h['SPY_SMA_24'] > spy_h['SPY_SMA_168']).astype(int)
    spy_h['SPY_Return_24h'] = spy_h['SPY'].pct_change(24)

    btc_h = yf.download('BTC-USD', period="2y", interval="1h", progress=False)
    btc_h.columns = [col[0] if isinstance(col, tuple) else col for col in btc_h.columns]
    btc_h = btc_h[['Close']].rename(columns={'Close': 'BTC'})
    btc_h.index = pd.to_datetime(btc_h.index).tz_convert('UTC')
    btc_h['BTC_Return']     = btc_h['BTC'].pct_change()
    btc_h['BTC_SMA_24']     = btc_h['BTC'].rolling(24).mean()
    btc_h['BTC_SMA_168']    = btc_h['BTC'].rolling(168).mean()
    btc_h['BTC_Trend']      = (btc_h['BTC_SMA_24'] > btc_h['BTC_SMA_168']).astype(int)
    btc_h['BTC_Return_24h'] = btc_h['BTC'].pct_change(24)

    spy_cols = spy_h[['SPY_Return','SPY_Trend','SPY_Return_24h']].copy()
    btc_cols = btc_h[['BTC_Return','BTC_Trend','BTC_Return_24h']].copy()
    market_h = spy_cols.join(btc_cols, how='inner')
    vix_hourly = vix_d[['VIX']].reindex(market_h.index, method='ffill')
    market_h['VIX'] = vix_hourly['VIX']
    market_h = market_h.ffill().dropna()
    p(f"  Market: {len(market_h):,} rows")

    market_cols = ['VIX','SPY_Return','SPY_Trend','SPY_Return_24h',
                   'BTC_Return','BTC_Trend','BTC_Return_24h']

    for symbol in list(crypto_data.keys()):
        crypto_data[symbol] = crypto_data[symbol].join(
            market_h[market_cols], how='inner')
        if symbol == 'BTC/USD':
            crypto_data[symbol]['BTC_Return']     = 0.0
            crypto_data[symbol]['BTC_Trend']      = 0.0
            crypto_data[symbol]['BTC_Return_24h'] = 0.0
        crypto_data[symbol].dropna(inplace=True)

    all_returns    = pd.concat([
        crypto_data[s]['Return'].rename(s) for s in crypto_data], axis=1, sort=False)
    crypto_mom     = all_returns.mean(axis=1)
    crypto_mom_24h = crypto_mom.rolling(24).mean()

    for symbol in list(crypto_data.keys()):
        crypto_data[symbol]['Crypto_Momentum']     = crypto_mom
        crypto_data[symbol]['Crypto_Momentum_24h'] = crypto_mom_24h
        crypto_data[symbol].dropna(inplace=True)

    combined = pd.concat(crypto_data.values())
    X = combined[CRYPTO_FEATURES]
    y = combined['Target'].squeeze()

    buy_rate = y.mean()
    # scale_pos_weight balances the class imbalance explicitly
    # e.g. if 35% BUY and 65% SELL, weight = 65/35 ≈ 1.86
    spw = (1 - buy_rate) / buy_rate
    p(f"✅ Combined: {len(combined):,} rows, {X.shape[1]} features")
    p(f"   BUY rate: {buy_rate:.1%}  →  scale_pos_weight: {spw:.2f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    p("⚡ Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=5,       # reduced from 6 — less overfitting
        learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,                # class balance fix
        random_state=42, n_jobs=-1,
        eval_metric='logloss', verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)], verbose=False)

    acc = accuracy_score(y_test, model.predict(X_test))
    preds_test = model.predict(X_test)
    buy_preds  = preds_test.mean()
    p(f"✅ Crypto accuracy: {acc:.1%}")
    p(f"   BUY predictions on test set: {buy_preds:.1%}  (was ~2% before fix)")

    joblib.dump(model, '/home/jv/crypto_model.joblib')
    with open('/home/jv/crypto_features.json', 'w') as f:
        json.dump(CRYPTO_FEATURES, f)
    with open('/home/jv/crypto_model_info.json', 'w') as f:
        json.dump({
            'trained_at':    datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model_type':    'XGBoost hourly',
            'accuracy':      f"{acc:.1%}",
            'features':      len(CRYPTO_FEATURES),
            'training_rows': len(X_train),
            'coins':         list(CRYPTOS.keys()),
            'interval':      '1h',
            'target':        '+0.5% in 6h',
            'buy_rate':      f"{buy_rate:.1%}",
        }, f)

    elapsed = time.time() - start
    p(f"✅ Crypto retrain complete in {elapsed:.0f}s")
    return acc

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    p(f"\n🤖 Auto Retrain — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    total_start = time.time()

    stock_acc  = retrain_stocks()
    crypto_acc = retrain_crypto()

    # Restart dashboard to load new models
    os.system('sudo systemctl restart dashboard')

    total = time.time() - total_start
    p(f"\n{'='*50}")
    p(f"✅ RETRAIN COMPLETE in {total:.0f}s")
    p(f"   Stock model:  {stock_acc:.1%}")
    p(f"   Crypto model: {crypto_acc:.1%}")
    p(f"{'='*50}\n")

