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
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

logging.basicConfig(
    filename='/home/jv/retrain.log',
    level=logging.INFO,
    format='%(asctime)s %(message)s'
)
log = logging.getLogger()

def p(msg):
    # Under cron, stdout is also redirected into retrain.log — printing there too duplicated every line
    if sys.stdout.isatty():
        print(msg)
    log.info(msg)

def make_exit_label(df, take_profit, stop_loss, max_bars):
    """
    For each bar, simulate holding a long position and look forward up to max_bars.
    Uses High/Low so intraday touches count.
    Returns 1 if take-profit is hit before stop-loss, 0 otherwise.
    Last max_bars rows are NaN (insufficient lookahead) and should be dropped.
    """
    close = df['Close'].values
    high  = df['High'].values
    low   = df['Low'].values
    n     = len(close)

    tp_bar = np.full(n, max_bars + 1, dtype=float)
    sl_bar = np.full(n, max_bars + 1, dtype=float)
    tp_prices = close * (1 + take_profit)
    sl_prices = close * (1 - stop_loss)

    # Iterate from furthest lag to nearest so the smallest (first) hit overwrites
    for lag in range(max_bars, 0, -1):
        idx = np.arange(n - lag)
        tp_bar[idx[high[idx + lag] >= tp_prices[idx]]] = lag
        sl_bar[idx[low[idx + lag]  <= sl_prices[idx]]] = lag

    labels = np.where(tp_bar < sl_bar, 1.0, 0.0)
    labels[n - max_bars:] = np.nan
    return pd.Series(labels, index=df.index)

STOCK_TICKERS = [
    'AAPL','GOOGL','AMZN','PLTR','CRWD',                  # Tech
    'NVDA','AMD','QCOM','ARM','MU',                       # Semiconductors
    'TSLA','SHOP','COIN',                                 # Growth
    'JPM','BAC','GS','MS','STT',                         # Finance
    'OXY',                                                # Energy
    'GOLD',                                               # Commodities
    'UNH','ISRG',                                         # Healthcare
    'CAT','VRT',                                          # Industrials
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
    'Tech':       ['AAPL','GOOGL','AMZN','PLTR','CRWD'],
    'Semi':       ['NVDA','AMD','QCOM','ARM','MU'],
    'Growth':     ['TSLA','SHOP','COIN'],
    'Finance':    ['JPM','BAC','GS','MS','STT'],
    'Energy':     ['OXY'],
    'Commodities':['GOLD'],
    'Healthcare': ['UNH','ISRG'],
    'Industrial': ['CAT','VRT'],
}
TICKER_SECTOR = {t: s for s, tickers in SECTORS.items() for t in tickers}

STOCK_FEATURES = [
    'SMA_10','SMA_30','SMA_50','EMA_12','EMA_26','MACD','RSI',
    'Return','Return_5d','Return_10d','Return_20d','Volatility','High_Low',
    'BB_position','BB_width','Vol_Ratio','Vol_Spike',
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
    'Return','Return_24h','Return_72h','Return_168h',
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
    df['Return_20d']   = df['Close'].pct_change(20)
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
# Live-rule evaluation (2026-11-08). The label-based test above counts every day a stock is flagged
# as a separate pick (~1,000 "picks" are ~130 real trades) and uses a fixed stop from entry, which
# flattered the model: it claimed 55.8% TP vs 35.2% random, while live trades went 1 for 12.
# This replays the test period the way trader.py actually trades and judges the model on money vs SPY,
# against random buys of the same tickers (which also beat SPY, because the ticker list was picked
# with hindsight). Constants must match trader.py.
LIVE_TRAIL_STOP  = 0.12   # trader.py STOP_LOSS (trailing from the highest close)
LIVE_TAKE_PROFIT = 0.20   # trader.py TAKE_PROFIT
LIVE_MAX_HOLD    = 60     # trading days; trader.py has no time exit, this matches the label horizon
LIVE_VIX_MAX     = 28     # trader.py VIX_BUY_THRESHOLD (buys also need SPY above its 200-day MA)

def simulate_live_exit(o, h, l, c, i, entry=None):
    """Buy at close of bar i (or at `entry`, e.g. an intraday price on that day), exit with
    trader.py's rules. Returns (exit_bar, return, hit_tp). Gaps through a stop or target fill at
    the open; checked once a day, so it misses the small extra slippage of hourly checks.
    auto_adjust.py uses this too."""
    entry = peak = c[i] if entry is None else entry
    end   = min(i + LIVE_MAX_HOLD, len(c) - 1)
    tp    = entry * (1 + LIVE_TAKE_PROFIT)
    for j in range(i + 1, end + 1):
        stop = peak * (1 - LIVE_TRAIL_STOP)
        if o[j] <= stop: return j, o[j] / entry - 1, False
        if o[j] >= tp:   return j, o[j] / entry - 1, True
        if l[j] <= stop: return j, stop / entry - 1, False   # assume the stop came first on a wide day
        if h[j] >= tp:   return j, LIVE_TAKE_PROFIT, True
        peak = max(peak, c[j])
    return end, c[end] / entry - 1, False

def live_rule_eval(test_df, proba, prices, buy_confidence):
    """Model trades (one open position per ticker, like trader.py) vs random buys on any eligible
    day, both measured against holding SPY over the same days."""
    spy = yf.download('SPY', period="15y", interval="1d", progress=False)
    spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
    spy_c   = spy['Close']
    regime  = (spy_c > spy_c.rolling(200).mean())
    df = test_df[['Ticker', 'VIX']].copy()
    df['proba'] = proba
    df = df[(df['VIX'] <= LIVE_VIX_MAX) & regime.reindex(df.index).fillna(False).values]

    model_out, random_out = [], []
    for ticker, rows in df.groupby('Ticker'):
        px = prices[ticker]
        o, h, l, c = (px[k].values for k in ('Open', 'High', 'Low', 'Close'))
        busy_until = -1
        for d, prob in zip(rows.index, rows['proba']):
            i = px.index.get_loc(d)
            j, ret, hit = simulate_live_exit(o, h, l, c, i)
            spy_ret = float(spy_c.asof(px.index[j]) / spy_c.asof(d) - 1)
            rec = (ret, hit, ret - spy_ret)
            random_out.append(rec)
            if prob >= buy_confidence and i > busy_until:
                model_out.append(rec)
                busy_until = j
    m, r = np.array(model_out, dtype=float), np.array(random_out, dtype=float)
    if len(m) == 0:
        return {'trades': 0, 'has_edge': False}
    se    = float(m[:, 2].std(ddof=1) / np.sqrt(len(m))) if len(m) > 1 else float('nan')
    extra = float(m[:, 2].mean() - r[:, 2].mean())
    return {
        'trades':              len(m),
        'tp_rate':             float(m[:, 1].mean()),
        'avg_return':          float(m[:, 0].mean()),
        'avg_vs_spy':          float(m[:, 2].mean()),
        'beat_spy':            float((m[:, 2] > 0).mean()),
        'random_tp_rate':      float(r[:, 1].mean()),
        'random_avg_vs_spy':   float(r[:, 2].mean()),
        'extra_vs_random':     extra,
        'standard_error':      se,
        # Needs enough trades, and the model's gain over random buys must be clear of noise
        'has_edge':            bool(len(m) >= 50 and extra >= 2 * se),
    }

def retrain_stocks():
    p("\n" + "="*50)
    p("📈 RETRAINING STOCK MODEL")
    p("="*50)
    start = time.time()

    p("📥 Downloading stock data...")
    all_data = {}
    prices   = {}
    for ticker in STOCK_TICKERS:
        try:
            df = yf.download(ticker, period="15y", interval="1d", progress=False)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            df = build_base_features(df)
            df['Target'] = make_exit_label(df, take_profit=0.20, stop_loss=0.12, max_bars=60)
            prices[ticker] = df[['Open', 'High', 'Low', 'Close']].copy()   # before dropna, for live_rule_eval
            df['Ticker'] = ticker
            df.dropna(inplace=True)
            df['Target'] = df['Target'].astype(int)
            buy_rate = df['Target'].mean()
            all_data[ticker] = df
            p(f"  {ticker}: {len(df):,} rows  (TP-first rate: {buy_rate:.1%})")
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

    combined = pd.concat(all_data.values()).sort_index(kind='stable')
    ts = combined.index

    # Time-based evaluation (fixed 2026-10-04) — same reasoning as the crypto model: the old
    # split tested on the last 5 tickers in the list (GOLD/UNH/ISRG/CAT/VRT), not on the future,
    # and those tickers were never trained on. Gap = label lookahead (60 trading days ≈ 90 days).
    LABEL_HORIZON = pd.Timedelta(days=90)
    BUY_CONFIDENCE = 0.60                       # must match trader.py's BUY threshold
    test_cut = ts[int(len(ts) * 0.8)]
    val_cut  = ts[int(len(ts) * 0.7)]
    fit_df   = combined[ts <  val_cut - LABEL_HORIZON]
    val_df   = combined[(ts >= val_cut) & (ts < test_cut - LABEL_HORIZON)]
    test_df  = combined[ts >= test_cut]

    buy_rate = float(combined['Target'].mean())
    fit_rate = float(fit_df['Target'].mean())
    spw = (1 - fit_rate) / fit_rate
    p(f"✅ Combined: {len(combined):,} rows, {len(STOCK_FEATURES)} features")
    p(f"   TP-first rate: {buy_rate:.1%}  →  scale_pos_weight: {spw:.2f}")
    p(f"   Fit: {len(fit_df):,} rows to {val_cut - LABEL_HORIZON:%Y-%m-%d}  |  "
      f"Val: {len(val_df):,}  |  Test: {len(test_df):,} rows from {test_cut:%Y-%m-%d}")

    def make_model(n_estimators, early_stop):
        return xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=42, n_jobs=-1,
            eval_metric='logloss', verbosity=0,
            early_stopping_rounds=50 if early_stop else None,
        )

    p("⚡ Training XGBoost (evaluation model)...")
    model = make_model(500, early_stop=True)
    model.fit(fit_df[STOCK_FEATURES], fit_df['Target'],
              eval_set=[(val_df[STOCK_FEATURES], val_df['Target'])], verbose=False)

    y_test    = test_df['Target']
    proba     = model.predict_proba(test_df[STOCK_FEATURES])[:, 1]
    acc       = accuracy_score(y_test, proba >= 0.5)
    baseline  = 1 - y_test.mean()
    base_rate = y_test.mean()                   # hit rate of buying on a random day
    picks     = y_test[proba >= BUY_CONFIDENCE] # what trader.py would actually buy
    n_picks   = int(len(picks))
    pick_rate = float(picks.mean()) if n_picks else 0.0
    edge      = pick_rate - base_rate
    p(f"✅ Stock accuracy: {acc:.1%}  (always-no baseline: {baseline:.1%}, best iteration: {model.best_iteration})")
    p(f"   Label test (counts every flagged day, fixed stop): {n_picks:,} picks — hit TP {pick_rate:.1%} "
      f"vs random {base_rate:.1%}  →  {edge:+.1%}")

    lr = live_rule_eval(test_df, proba, prices, BUY_CONFIDENCE)
    has_edge = lr['has_edge']
    if lr['trades']:
        p(f"   Live-rule test: {lr['trades']} trades — hit TP {lr['tp_rate']:.1%} (random {lr['random_tp_rate']:.1%}), "
          f"avg {lr['avg_return']:+.1%}/trade, vs SPY {lr['avg_vs_spy']:+.1%} (random buys {lr['random_avg_vs_spy']:+.1%}), "
          f"beat SPY {lr['beat_spy']:.0%}")
    p(f"   Model's gain over random buys: {lr.get('extra_vs_random', 0):+.1%}/trade "
      f"(±{lr.get('standard_error', 0):.1%})  {'✅ HAS EDGE' if has_edge else '⚠️  NO EDGE'}")

    # Deploy a model refit on all labelled data (all tickers, incl. the recent test period)
    p("⚡ Refitting on all data for deployment...")
    model = make_model(model.best_iteration + 1, early_stop=False)
    model.fit(combined[STOCK_FEATURES], combined['Target'], verbose=False)

    joblib.dump(model, '/home/jv/trading_model.joblib')
    with open('/home/jv/features.json', 'w') as f:
        json.dump(STOCK_FEATURES, f)
    with open('/home/jv/model_info.json', 'w') as f:
        json.dump({
            'trained_at':    datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model_type':    'XGBoost',
            'accuracy':      f"{acc:.1%}",
            'baseline_accuracy': f"{baseline:.1%}",
            'pick_hit_rate': f"{pick_rate:.1%}",
            'random_buy_hit_rate': f"{base_rate:.1%}",
            'edge':          f"{edge:+.1%}",
            'has_edge':      has_edge,   # from live_rule_test since 2026-11-08
            'test_picks':    n_picks,
            'label_test_note': 'pick_hit_rate/random_buy_hit_rate/edge/test_picks count every flagged day '
                               'with a fixed stop; they overstate the edge. Use live_rule_test.',
            'live_rule_test': {
                'trades':            lr['trades'],
                'tp_rate':           f"{lr.get('tp_rate', 0):.1%}",
                'random_tp_rate':    f"{lr.get('random_tp_rate', 0):.1%}",
                'avg_return':        f"{lr.get('avg_return', 0):+.1%}",
                'avg_vs_spy':        f"{lr.get('avg_vs_spy', 0):+.1%}",
                'random_avg_vs_spy': f"{lr.get('random_avg_vs_spy', 0):+.1%}",
                'extra_vs_random':   f"{lr.get('extra_vs_random', 0):+.1%}",
                'standard_error':    f"{lr.get('standard_error', 0):.1%}",
                'beat_spy':          f"{lr.get('beat_spy', 0):.0%}",
                'rule': 'one position per ticker, 12% trailing stop, +20% TP, 60-day max, SPY>200MA, VIX<=28; '
                        'has_edge needs >=50 trades and extra_vs_random >= 2 standard errors',
            },
            'buy_confidence': BUY_CONFIDENCE,
            'evaluation':    f"time split, test from {test_cut:%Y-%m-%d}, 90-day purge gap",
            'features':      len(STOCK_FEATURES),
            'training_rows': len(combined),
            'tickers':       STOCK_TICKERS,
            'target':        'exit-simulation: TP +20% before SL -12% within 60 trading days',
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
            df['Return_168h'] = df['Close'].pct_change(168)
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
            df.index = pd.to_datetime(df.index).tz_convert('UTC')
            # Target: exit-simulation — TP +25% before SL -15% within ~30 days (720h)
            df['Target'] = make_exit_label(df, take_profit=0.25, stop_loss=0.15, max_bars=720)
            df.dropna(inplace=True)
            df['Target'] = df['Target'].astype(int)
            buy_rate = df['Target'].mean()
            crypto_data[symbol] = df
            p(f"  {symbol}: {len(df):,} rows  (TP-first rate: {buy_rate:.1%})")
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

    combined = pd.concat(crypto_data.values()).sort_index(kind='stable')
    ts = combined.index

    # Time-based evaluation (fixed 2026-10-04). The old 80/20 split on the coin-by-coin
    # concat tested on DOT+ADA over the whole 2 years, so "accuracy" never measured
    # predicting the future. Now: test on the most recent 20% of time across all coins,
    # with a gap of LABEL_HORIZON hours so no training label peeks into the test period.
    LABEL_HORIZON = pd.Timedelta(hours=720)
    test_cut = ts[int(len(ts) * 0.8)]
    val_cut  = ts[int(len(ts) * 0.6)]
    fit_df   = combined[ts <  val_cut - LABEL_HORIZON]                         # trees learn here
    val_df   = combined[(ts >= val_cut) & (ts < test_cut - LABEL_HORIZON)]     # early stopping
    test_df  = combined[ts >= test_cut]                                        # never seen in training

    buy_rate = combined['Target'].mean()
    fit_rate = fit_df['Target'].mean()
    spw = (1 - fit_rate) / fit_rate
    p(f"✅ Combined: {len(combined):,} rows, {len(CRYPTO_FEATURES)} features")
    p(f"   BUY rate: {buy_rate:.1%}  →  scale_pos_weight: {spw:.2f}")
    p(f"   Fit: {len(fit_df):,} rows to {val_cut - LABEL_HORIZON:%Y-%m-%d}  |  "
      f"Val: {len(val_df):,}  |  Test: {len(test_df):,} rows from {test_cut:%Y-%m-%d}")

    def make_model(n_estimators, early_stop):
        return xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=5,
            learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,                # class balance fix
            random_state=42, n_jobs=-1,
            eval_metric='logloss', verbosity=0,
            early_stopping_rounds=50 if early_stop else None,
        )

    p("⚡ Training XGBoost (evaluation model)...")
    model = make_model(500, early_stop=True)
    model.fit(fit_df[CRYPTO_FEATURES], fit_df['Target'],
              eval_set=[(val_df[CRYPTO_FEATURES], val_df['Target'])], verbose=False)

    y_test     = test_df['Target']
    preds_test = model.predict(test_df[CRYPTO_FEATURES])
    acc        = accuracy_score(y_test, preds_test)
    baseline   = 1 - y_test.mean()                  # accuracy of always saying "don't buy"
    base_rate  = y_test.mean()                      # hit rate of buying at random
    n_buys     = int(preds_test.sum())
    buy_prec   = precision_score(y_test, preds_test, zero_division=0)
    edge       = buy_prec - base_rate
    # Edge = model's BUY calls hit TP more often than random entries, on enough calls to mean something
    has_edge   = bool(edge >= 0.03 and n_buys >= 200)
    p(f"✅ Crypto accuracy: {acc:.1%}  (always-no baseline: {baseline:.1%}, best iteration: {model.best_iteration})")
    p(f"   BUY precision: {buy_prec:.1%} vs random {base_rate:.1%}  →  edge {edge:+.1%} "
      f"on {n_buys:,} BUY calls  {'✅ HAS EDGE' if has_edge else '⚠️  NO EDGE'}")

    # Deploy a model refit on all labelled data (incl. the recent test period) with the
    # tree count found above, so live predictions reflect the latest regime.
    p("⚡ Refitting on all data for deployment...")
    model = make_model(model.best_iteration + 1, early_stop=False)
    model.fit(combined[CRYPTO_FEATURES], combined['Target'], verbose=False)

    joblib.dump(model, '/home/jv/crypto_model.joblib')
    with open('/home/jv/crypto_features.json', 'w') as f:
        json.dump(CRYPTO_FEATURES, f)
    with open('/home/jv/crypto_model_info.json', 'w') as f:
        json.dump({
            'trained_at':    datetime.now().strftime('%Y-%m-%d %H:%M'),
            'model_type':    'XGBoost hourly',
            'accuracy':      f"{acc:.1%}",
            'baseline_accuracy': f"{baseline:.1%}",
            'buy_precision': f"{buy_prec:.1%}",
            'random_buy_hit_rate': f"{base_rate:.1%}",
            'edge':          f"{edge:+.1%}",
            'has_edge':      has_edge,
            'test_buy_calls': n_buys,
            'evaluation':    f"time split, test from {test_cut:%Y-%m-%d}, 720h purge gap",
            'features':      len(CRYPTO_FEATURES),
            'training_rows': len(combined),
            'coins':         list(CRYPTOS.keys()),
            'interval':      '1h',
            'target':        'exit-simulation: TP +25% before SL -15% within 720h',
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

