#!/usr/bin/env python3
"""
Stock trader with performance logging.
Runs hourly during market hours via crontab.
"""
import os
import json
import joblib
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime

print("=" * 50)
print(f"📈 Stock Trader — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 50)

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
api      = tradeapi.REST(ENV['ALPACA_KEY'], ENV['ALPACA_SECRET'], BASE_URL, api_version='v2')

TICKERS = [
    'AAPL','GOOGL','MSFT','META','AMZN','TSLA','NVDA',
    'JPM','BAC','GS','PFE','MRK','RIVN','NIO',
    'WMT','TGT','AMD','QCOM','MS'
]
SECTORS = {
    'Tech':    ['AAPL','GOOGL','MSFT','META','AMZN','NVDA','AMD','QCOM'],
    'EV':      ['TSLA','RIVN','NIO'],
    'Finance': ['JPM','BAC','GS','MS'],
    'Pharma':  ['PFE','MRK'],
    'Retail':  ['WMT','TGT'],
}
TICKER_SECTOR = {t: s for s, tickers in SECTORS.items() for t in tickers}

FRACTIONABLE_CACHE = {}

def is_fractionable(ticker):
    if ticker not in FRACTIONABLE_CACHE:
        try:
            asset = api.get_asset(ticker)
            FRACTIONABLE_CACHE[ticker] = getattr(asset, 'fractionable', False)
        except:
            FRACTIONABLE_CACHE[ticker] = False
    return FRACTIONABLE_CACHE[ticker]

model = joblib.load('/home/jv/trading_model.joblib')
with open('/home/jv/features.json') as f:
    FEATURES = json.load(f)

def get_market_indicators():
    print("  🌍 Fetching market indicators...")
    vix = yf.download('^VIX', period="60d", interval="1d", progress=False)
    vix.columns = [col[0] if isinstance(col, tuple) else col for col in vix.columns]
    vix = vix[['Close']].rename(columns={'Close': 'VIX'})

    spy = yf.download('SPY', period="60d", interval="1d", progress=False)
    spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
    spy = spy[['Close']].rename(columns={'Close': 'SPY'})
    spy['SPY_Return']    = spy['SPY'].pct_change()
    spy['SPY_SMA_10']    = spy['SPY'].rolling(10).mean()
    spy['SPY_SMA_50']    = spy['SPY'].rolling(50).mean()
    spy['SPY_Trend']     = (spy['SPY_SMA_10'] > spy['SPY_SMA_50']).astype(int)
    spy['SPY_Return_5d'] = spy['SPY'].pct_change(5)

    qqq = yf.download('QQQ', period="60d", interval="1d", progress=False)
    qqq.columns = [col[0] if isinstance(col, tuple) else col for col in qqq.columns]
    qqq = qqq[['Close']].rename(columns={'Close': 'QQQ'})
    qqq['QQQ_Return']    = qqq['QQQ'].pct_change()
    qqq['QQQ_Trend']     = (qqq['QQQ'].rolling(10).mean() > qqq['QQQ'].rolling(50).mean()).astype(int)
    qqq['QQQ_Return_5d'] = qqq['QQQ'].pct_change(5)

    market = vix.join(spy[['SPY_Return','SPY_SMA_10','SPY_SMA_50','SPY_Trend','SPY_Return_5d']], how='inner')
    market = market.join(qqq[['QQQ_Return','QQQ_Trend','QQQ_Return_5d']], how='inner')
    market.dropna(inplace=True)
    row = market.iloc[-1]
    print(f"  VIX: {float(row['VIX']):.1f} | SPY: {'↑' if row['SPY_Trend'] else '↓'} | QQQ: {'↑' if row['QQQ_Trend'] else '↓'}")
    return row

def get_features(ticker):
    df = yf.download(ticker, period="60d", interval="1d", progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df['SMA_10']      = df['Close'].rolling(10).mean()
    df['SMA_30']      = df['Close'].rolling(30).mean()
    df['SMA_50']      = df['Close'].rolling(50).mean()
    df['EMA_12']      = df['Close'].ewm(span=12).mean()
    df['EMA_26']      = df['Close'].ewm(span=26).mean()
    df['MACD']        = df['EMA_12'] - df['EMA_26']
    df['RSI']         = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() /
                                          df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
    df['Return']      = df['Close'].pct_change()
    df['Return_5d']   = df['Close'].pct_change(5)
    df['Return_10d']  = df['Close'].pct_change(10)
    df['Volatility']  = df['Return'].rolling(10).std()
    df['High_Low']    = (df['High'] - df['Low']) / df['Close']
    bb_mid            = df['Close'].rolling(20).mean()
    bb_std            = df['Close'].rolling(20).std()
    df['BB_upper']    = bb_mid + (2 * bb_std)
    df['BB_lower']    = bb_mid - (2 * bb_std)
    df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['BB_width']    = (df['BB_upper'] - df['BB_lower']) / bb_mid
    df['Volume_MA']   = df['Volume'].rolling(10).mean()
    df['Vol_Ratio']   = df['Volume'] / df['Volume_MA']
    df['Vol_Spike']   = (df['Vol_Ratio'] > 2.0).astype(int)
    df['DayOfWeek']   = pd.to_datetime(df.index).dayofweek
    df.dropna(inplace=True)
    return df

def get_sector_momentum(ticker, all_returns):
    sector    = TICKER_SECTOR.get(ticker, 'Tech')
    tickers   = SECTORS[sector]
    available = [t for t in tickers if t in all_returns]
    if not available:
        return 0.0, 0.0
    mom    = sum(all_returns[t]['latest'] for t in available) / len(available)
    mom_5d = sum(all_returns[t]['5d'] for t in available) / len(available)
    return mom, mom_5d

def log_signal(symbol, signal, confidence, sentiment, price, acted, reason):
    log_file = '/home/jv/performance.json'
    entry = {
        'time':          datetime.now().isoformat(),
        'symbol':        symbol,
        'asset_type':    'stock',
        'signal':        signal,
        'confidence':    round(float(confidence), 4),
        'sentiment':     round(float(sentiment), 4),
        'price':         round(float(price), 4),
        'acted':         acted,
        'reason':        reason,
        'outcome':       None,
        'outcome_price': None,
        'outcome_time':  None,
    }
    data = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            try:
                data = json.load(f)
            except:
                data = []
    data.append(entry)
    with open(log_file, 'w') as f:
        json.dump(data, f, indent=2)

def run():
    account = api.get_account()
    cash    = float(account.cash)
    print(f"💰 Cash: ${cash:,.2f}")

    mkt = get_market_indicators()

    print("  📊 Fetching sector data...")
    all_returns = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="10d", interval="1d", progress=False)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            ret = df['Close'].pct_change()
            all_returns[ticker] = {
                'latest': float(ret.iloc[-1]),
                '5d':     float(ret.iloc[-5:].mean())
            }
        except:
            all_returns[ticker] = {'latest': 0.0, '5d': 0.0}

    positions = {p.symbol: p for p in api.list_positions()}
    signals_logged = 0

    from sentiment import get_sentiment

    for ticker in TICKERS:
        print(f"\n📊 {ticker}")
        try:
            df            = get_features(ticker)
            latest        = df.iloc[-1].copy()
            current_price = float(df['Close'].iloc[-1])
            mom, mom_5d   = get_sector_momentum(ticker, all_returns)

            latest['Sector_Momentum']    = mom
            latest['Sector_Momentum_5d'] = mom_5d
            latest['VIX']           = float(mkt['VIX'])
            latest['SPY_Return']    = float(mkt['SPY_Return'])
            latest['SPY_Trend']     = float(mkt['SPY_Trend'])
            latest['SPY_Return_5d'] = float(mkt['SPY_Return_5d'])
            latest['QQQ_Return']    = float(mkt['QQQ_Return'])
            latest['QQQ_Trend']     = float(mkt['QQQ_Trend'])
            latest['QQQ_Return_5d'] = float(mkt['QQQ_Return_5d'])

            feature_row = pd.DataFrame([latest[FEATURES]])
            pred        = model.predict(feature_row)[0]
            conf        = model.predict_proba(feature_row)[0][pred]
            direction   = 'UP' if pred == 1 else 'DOWN'
            emoji       = '📈' if pred == 1 else '📉'
            print(f"  ML: {direction} {emoji} ({conf:.1%})")
            print(f"  Price: ${current_price:.4f}")

            sent_score = get_sentiment(ticker)

            # --- SELL logic ---
            if ticker in positions:
                pos         = positions[ticker]
                entry_price = float(pos.avg_entry_price)
                pnl_pct     = (current_price - entry_price) / entry_price

                if pnl_pct <= -0.05:
                    print(f"  🛑 Stop-loss triggered ({pnl_pct:.1%}) — SELLING")
                    api.submit_order(symbol=ticker, qty=float(pos.qty),
                                     side='sell', type='market', time_in_force='day')
                    log_signal(ticker, 'SELL', conf, sent_score, current_price, True, 'stop_loss')
                    signals_logged += 1
                    continue

                if pnl_pct >= 0.03:
                    print(f"  🎯 Take-profit triggered ({pnl_pct:.1%}) — SELLING")
                    api.submit_order(symbol=ticker, qty=float(pos.qty),
                                     side='sell', type='market', time_in_force='day')
                    log_signal(ticker, 'SELL', conf, sent_score, current_price, True, 'take_profit')
                    signals_logged += 1
                    continue

                if pred == 0 and conf >= 0.60:
                    if sent_score > 0.05:
                        print(f"  ⏸️  Skipping SELL — positive news ({sent_score:.2f})")
                        log_signal(ticker, 'SELL', conf, sent_score, current_price, False, 'sentiment_override')
                    else:
                        print(f"  📉 SELL signal — selling position")
                        api.submit_order(symbol=ticker, qty=float(pos.qty),
                                         side='sell', type='market', time_in_force='day')
                        log_signal(ticker, 'SELL', conf, sent_score, current_price, True, 'ml_signal')
                    signals_logged += 1
                    continue
                else:
                    print(f"  ⏸️  Holding position (P&L: {pnl_pct:.1%})")
                    continue

            # --- BUY logic ---
            if pred == 1 and conf >= 0.60:
                if sent_score < -0.05:
                    print(f"  ⏸️  Skipping BUY — negative news ({sent_score:.2f})")
                    log_signal(ticker, 'BUY', conf, sent_score, current_price, False, 'sentiment_override')
                    signals_logged += 1
                    continue

                if cash < 1000:
                    print(f"  ⏸️  Skipping BUY — insufficient cash (${cash:,.2f})")
                    continue

                position_size = min(cash * 0.10, cash - 100)
                fractionable  = is_fractionable(ticker)

                if fractionable:
                    qty = round(position_size / current_price, 9)
                    if qty < 0.001:
                        print(f"  ⏸️  Skipping BUY — position size too small")
                        continue
                else:
                    qty = int(position_size / current_price)
                    if qty < 1:
                        print(f"  ⏸️  Skipping BUY — price too high for position size (not fractionable)")
                        continue

                cost = qty * current_price
                frac_label = '(fractional)' if fractionable and qty != int(qty) else ''
                print(f"  ✅ BUY {qty} shares {frac_label}@ ${current_price:.2f} = ${cost:,.2f}")
                api.submit_order(symbol=ticker, qty=qty, side='buy',
                                 type='market', time_in_force='day')
                cash -= cost
                log_signal(ticker, 'BUY', conf, sent_score, current_price, True, 'ml_signal')
                signals_logged += 1
            else:
                if conf < 0.60:
                    print(f"  ⏸️  Skipping — confidence too low ({conf:.1%})")
                else:
                    print(f"  ⏸️  No signal")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n  📝 Logged {signals_logged} signals")
    print("=" * 50)
    print("✅ Done!")
    print("=" * 50)

if __name__ == "__main__":
    run()
    