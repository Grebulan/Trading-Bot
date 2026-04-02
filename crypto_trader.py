#!/usr/bin/env python3
"""
Crypto trader using hourly candles + CryptoPanic sentiment.
Runs every 2 hours via crontab.
"""

import os
import json
import joblib
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from crypto_sentiment import get_crypto_sentiment

def load_env():
    env = {}
    with open(os.path.expanduser('~/.env_trading')) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                env[k] = v
    return env

ENV      = load_env()
BASE_URL = "https://paper-api.alpaca.markets"
api      = tradeapi.REST(ENV['ALPACA_KEY'], ENV['ALPACA_SECRET'],
                         BASE_URL, api_version='v2')

model = joblib.load('/home/jv/crypto_model.joblib')
with open('/home/jv/crypto_features.json') as f:
    FEATURES = json.load(f)

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

STOP_LOSS    = 0.15
TAKE_PROFIT  = 0.10
CONFIDENCE   = 0.65
MIN_CASH     = 500.0
MAX_POSITION = 0.10

def safe_yf_download(symbol, period, interval, progress=False, timeout=30):
    """yf.download() with a timeout to prevent indefinite hangs."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            yf.download, symbol,
            period=period, interval=interval, progress=progress
        )
        try:
            return future.result(timeout=timeout)
        except (FuturesTimeoutError, Exception) as e:
            print(f"  ⚠️  yf.download timeout/error for {symbol}: {e}")
            return None

def get_market_indicators():
    try:
        vix = safe_yf_download('^VIX', period="10d", interval="1d")
        if vix is None:
            raise Exception("VIX download timed out")
        vix.columns = [col[0] if isinstance(col, tuple) else col for col in vix.columns]
        vix = vix[['Close']].rename(columns={'Close': 'VIX'})

        spy = safe_yf_download('SPY', period="30d", interval="1h")
        if spy is None:
            raise Exception("SPY download timed out")
        spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
        spy = spy[['Close']].rename(columns={'Close': 'SPY'})
        spy.index = pd.to_datetime(spy.index).tz_convert('UTC')
        spy = spy.resample('1h').last().ffill()
        spy['SPY_Return']     = spy['SPY'].pct_change()
        spy['SPY_SMA_24']     = spy['SPY'].rolling(24).mean()
        spy['SPY_SMA_168']    = spy['SPY'].rolling(168).mean()
        spy['SPY_Trend']      = (spy['SPY_SMA_24'] > spy['SPY_SMA_168']).astype(int)
        spy['SPY_Return_24h'] = spy['SPY'].pct_change(24)

        btc = safe_yf_download('BTC-USD', period="30d", interval="1h")
        if btc is None:
            raise Exception("BTC download timed out")
        btc.columns = [col[0] if isinstance(col, tuple) else col for col in btc.columns]
        btc = btc[['Close']].rename(columns={'Close': 'BTC'})
        btc.index = pd.to_datetime(btc.index).tz_convert('UTC')
        btc['BTC_Return']     = btc['BTC'].pct_change()
        btc['BTC_SMA_24']     = btc['BTC'].rolling(24).mean()
        btc['BTC_SMA_168']    = btc['BTC'].rolling(168).mean()
        btc['BTC_Trend']      = (btc['BTC_SMA_24'] > btc['BTC_SMA_168']).astype(int)
        btc['BTC_Return_24h'] = btc['BTC'].pct_change(24)

        return {
            'VIX':            float(vix['VIX'].iloc[-1]),
            'SPY_Return':     float(spy['SPY_Return'].iloc[-1]),
            'SPY_Trend':      float(spy['SPY_Trend'].iloc[-1]),
            'SPY_Return_24h': float(spy['SPY_Return_24h'].iloc[-1]),
            'BTC_Return':     float(btc['BTC_Return'].iloc[-1]),
            'BTC_Trend':      float(btc['BTC_Trend'].iloc[-1]),
            'BTC_Return_24h': float(btc['BTC_Return_24h'].iloc[-1]),
        }
    except Exception as e:
        print(f"  Market indicator error: {e}")
        return None

def get_crypto_features(yf_symbol):
    df = safe_yf_download(yf_symbol, period="30d", interval="1h")
    if df is None:
        raise Exception(f"Download timed out for {yf_symbol}")
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df['SMA_24']       = df['Close'].rolling(24).mean()
    df['SMA_72']       = df['Close'].rolling(72).mean()
    df['SMA_168']      = df['Close'].rolling(168).mean()
    df['EMA_12']       = df['Close'].ewm(span=12).mean()
    df['EMA_26']       = df['Close'].ewm(span=26).mean()
    df['MACD']         = df['EMA_12'] - df['EMA_26']
    df['RSI']          = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() /
                                           df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
    df['Return']       = df['Close'].pct_change()
    df['Return_24h']   = df['Close'].pct_change(24)
    df['Return_72h']   = df['Close'].pct_change(72)
    df['Volatility']   = df['Return'].rolling(24).std()
    df['High_Low']     = (df['High'] - df['Low']) / df['Close']
    bb_mid             = df['Close'].rolling(48).mean()
    bb_std             = df['Close'].rolling(48).std()
    df['BB_upper']     = bb_mid + (2 * bb_std)
    df['BB_lower']     = bb_mid - (2 * bb_std)
    df['BB_position']  = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['BB_width']     = (df['BB_upper'] - df['BB_lower']) / bb_mid
    df['Volume_MA']    = df['Volume'].rolling(24).mean()
    df['Vol_Ratio']    = df['Volume'] / df['Volume_MA']
    df['Vol_Spike']    = (df['Vol_Ratio'] > 2.0).astype(int)
    df['HourOfDay']    = pd.to_datetime(df.index).hour
    df['DayOfWeek']    = pd.to_datetime(df.index).dayofweek
    df['IsWeekend']    = (df['DayOfWeek'] >= 5).astype(int)
    df['Overnight_Gap']= (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df.dropna(inplace=True)
    return df

def get_crypto_momentum(all_returns):
    if not all_returns:
        return 0.0, 0.0
    latest = sum(v['latest'] for v in all_returns.values()) / len(all_returns)
    h24    = sum(v['24h'] for v in all_returns.values()) / len(all_returns)
    return latest, h24

def place_trade(symbol, side, cash, current_price):
    alpaca_symbol = symbol.replace('/', '')
    try:
        if side == 'buy':
            amount = min(cash * MAX_POSITION, cash - MIN_CASH)
            if amount < 10:
                print(f"  ⚠️  Not enough cash to buy {symbol}")
                return
            qty = round(amount / current_price, 6)
            api.submit_order(symbol=alpaca_symbol, qty=qty,
                           side='buy', type='market', time_in_force='gtc')
            print(f"  ✅ BUY {qty} {symbol} @ ${current_price:,.4f}")
        else:
            positions = {p.symbol: p for p in api.list_positions()}
            if alpaca_symbol not in positions:
                return
            qty = positions[alpaca_symbol].qty
            api.submit_order(symbol=alpaca_symbol, qty=qty,
                           side='sell', type='market', time_in_force='gtc')
            print(f"  ✅ SELL {qty} {symbol} @ ${current_price:,.4f}")
    except Exception as e:
        print(f"  ❌ Order error {symbol}: {e}")

def main():
    now = datetime.now()
    print(f"\n{'='*50}")
    print(f"🪙 Crypto Trader — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    account = api.get_account()
    cash    = float(account.cash)
    print(f"💰 Cash: ${cash:,.2f}")

    if cash < MIN_CASH:
        print(f"⚠️  Cash below minimum ${MIN_CASH}, skipping")
        return

    positions = {p.symbol: p for p in api.list_positions()}

    print("\n  🌍 Fetching market indicators...")
    mkt = get_market_indicators()
    if mkt:
        btc_trend = '↑' if mkt['BTC_Trend'] else '↓'
        spy_trend = '↑' if mkt['SPY_Trend'] else '↓'
        print(f"  VIX: {mkt['VIX']:.1f} | SPY: {spy_trend} | BTC: {btc_trend}")

    print("\n  📊 Fetching crypto data...")
    all_dfs     = {}
    all_returns = {}
    for symbol, yf_symbol in CRYPTOS.items():
        try:
            df = get_crypto_features(yf_symbol)
            all_dfs[symbol]     = df
            all_returns[symbol] = {
                'latest': float(df['Return'].iloc[-1]),
                '24h':    float(df['Return_24h'].iloc[-1])
            }
        except Exception as e:
            print(f"  ⚠️  {symbol}: {e}")

    crypto_mom, crypto_mom_24h = get_crypto_momentum(all_returns)

    for symbol, yf_symbol in CRYPTOS.items():
        print(f"\n📊 {symbol}")
        if symbol not in all_dfs:
            print(f"  ⚠️  No data, skipping")
            continue

        try:
            df            = all_dfs[symbol]
            latest        = df.iloc[-1].copy()
            current_price = float(df['Close'].iloc[-1])
            alpaca_symbol = symbol.replace('/', '')

            if mkt:
                for k, v in mkt.items():
                    latest[k] = v
                if symbol == 'BTC/USD':
                    latest['BTC_Return']     = 0.0
                    latest['BTC_Trend']      = 0.0
                    latest['BTC_Return_24h'] = 0.0
            else:
                for col in ['VIX','SPY_Return','SPY_Trend','SPY_Return_24h',
                            'BTC_Return','BTC_Trend','BTC_Return_24h']:
                    latest[col] = 0.0

            latest['Crypto_Momentum']     = crypto_mom
            latest['Crypto_Momentum_24h'] = crypto_mom_24h

            feature_row = pd.DataFrame([latest[FEATURES]])
            pred        = model.predict(feature_row)[0]
            conf        = model.predict_proba(feature_row)[0][pred]
            signal      = 'BUY' if pred == 1 else 'SELL'
            print(f"  ML: {'UP 📈' if pred == 1 else 'DOWN 📉'} ({conf:.1%})")
            print(f"  Price: ${current_price:,.4f}")

            # Sentiment
            sent_score = get_crypto_sentiment(symbol)

            # Stop loss / take profit
            if alpaca_symbol in positions:
                pos    = positions[alpaca_symbol]
                entry  = float(pos.avg_entry_price)
                change = (current_price - entry) / entry
                print(f"  Position: {pos.qty} coins, entry ${entry:,.4f}, P&L {change:+.1%}")

                if change <= -STOP_LOSS:
                    print(f"  🛑 STOP LOSS triggered ({change:+.1%})")
                    place_trade(symbol, 'sell', cash, current_price)
                    continue
                elif change >= TAKE_PROFIT:
                    print(f"  🎯 TAKE PROFIT triggered ({change:+.1%})")
                    place_trade(symbol, 'sell', cash, current_price)
                    continue

            # Confidence check
            if conf < CONFIDENCE:
                print(f"  ⏸️  Skipping — confidence too low ({conf:.1%})")
                continue

            # Sentiment override
            if signal == 'BUY' and sent_score < -0.05:
                print(f"  ⏸️  Skipping BUY — negative news ({sent_score:.2f})")
                continue
            if signal == 'SELL' and sent_score > 0.05:
                print(f"  ⏸️  Skipping SELL — positive news ({sent_score:.2f})")
                continue

            if signal == 'BUY' and alpaca_symbol not in positions:
                place_trade(symbol, 'buy', cash, current_price)
            elif signal == 'SELL' and alpaca_symbol in positions:
                place_trade(symbol, 'sell', cash, current_price)
            else:
                action = 'Nothing to sell' if signal == 'SELL' else 'Already holding'
                print(f"  ⏸️  {action}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n{'='*50}")
    print(f"✅ Done!")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    main()
