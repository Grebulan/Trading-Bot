#!/usr/bin/env python3
"""
Stock trader with performance logging.
Runs hourly during market hours via crontab.
"""
import os
import math
import time
import json
import joblib
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
    'AAPL','GOOGL','AMZN','PLTR','CRWD',                  # Tech
    'NVDA','AMD','QCOM','ARM','MU',                       # Semiconductors
    'TSLA','SHOP','COIN',                                 # Growth
    'JPM','BAC','GS','MS','STT',                         # Finance
    'OXY',                                                # Energy
    'GOLD',                                               # Commodities
    'UNH','ISRG',                                         # Healthcare
    'CAT','VRT',                                          # Industrials
]
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

STOP_LOSS       = 0.12   # 12% trailing — moves up with price
# Hard stop disabled 2026-10-04: walk-forward sim over 5,142 out-of-sample picks (2019-2026) showed
# the 6% hard stop lowered avg return per trade (+6.0% vs +6.9% without). Set to e.g. 0.06 to re-enable.
HARD_STOP       = None
TAKE_PROFIT     = 0.20   # 20% — let winners run for weeks
# Stop override (2026-10-20): if the model still says BUY (>=60%) when the trailing stop is hit,
# keep holding, unless the position is down STOP_OVERRIDE_BACKSTOP from entry, which always sells.
# Walk-forward sim 2019-2026: +$99k vs SPY against +$88k with the stop always selling (better in
# 94% of resamples), mostly by avoiding stop-out-then-rebuy cycles. Set to None to disable.
# Disabled 2026-11-08 (weekly review): live losses ran to -15..-25% (VRT, ARM, CAT) and CAT was
# stopped out then rebought within the hour. Set back to 0.60 to re-enable.
STOP_OVERRIDE_CONF     = None
STOP_OVERRIDE_BACKSTOP = 0.30
# Stock buys paused 2026-11-08 (weekly review): ML picks were $8,900 behind holding SPY, 1 of 12
# closed trades beat it. Signals are still logged (reason 'buys_paused') so the model can be judged
# without risking money. Existing positions still exit via trailing stop / take-profit.
BUYS_PAUSED     = True
MAX_POSITIONS   = 6      # max simultaneous stock positions (excludes SPY); was 8 until 2026-10-20 (6 x 8% caps stocks at ~48%)
VIX_BUY_THRESHOLD = 28  # skip buys when volatility is elevated

# SPY default position — holds spare cash when no stock signals are firing
SPY_TRAIL_STOP  = 0.05   # 5% trailing stop — conservative given SPY is majority of portfolio
SPY_VIX_CRASH   = 40     # hard sell if VIX spikes above this (sudden crash)
SPY_MIN_RESERVE = 2000   # always keep this much cash free
STOCK_POSITION_PCT = 0.08  # target size of each ML stock buy, as a share of equity

HIGHWATER_FILE  = '/home/jv/stock_highwater.json'

def load_highwater():
    if os.path.exists(HIGHWATER_FILE):
        with open(HIGHWATER_FILE) as f:
            return json.load(f)
    return {}

def save_highwater(hw):
    with open(HIGHWATER_FILE, 'w') as f:
        json.dump(hw, f, indent=2)

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

    spy = yf.download('SPY', period="300d", interval="1d", progress=False)
    spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
    spy = spy[['Close']].rename(columns={'Close': 'SPY'})
    spy['SPY_Return']    = spy['SPY'].pct_change()
    spy['SPY_SMA_10']    = spy['SPY'].rolling(10).mean()
    spy['SPY_SMA_50']    = spy['SPY'].rolling(50).mean()
    spy['SPY_SMA_200']   = spy['SPY'].rolling(200).mean()
    spy['SPY_Trend']     = (spy['SPY_SMA_10'] > spy['SPY_SMA_50']).astype(int)
    spy['SPY_Return_5d'] = spy['SPY'].pct_change(5)
    spy['SPY_Regime']    = (spy['SPY'] > spy['SPY_SMA_200']).astype(int)

    qqq = yf.download('QQQ', period="300d", interval="1d", progress=False)
    qqq.columns = [col[0] if isinstance(col, tuple) else col for col in qqq.columns]
    qqq = qqq[['Close']].rename(columns={'Close': 'QQQ'})
    qqq['QQQ_Return']    = qqq['QQQ'].pct_change()
    qqq['QQQ_SMA_200']   = qqq['QQQ'].rolling(200).mean()
    qqq['QQQ_Trend']     = (qqq['QQQ'].rolling(10).mean() > qqq['QQQ'].rolling(50).mean()).astype(int)
    qqq['QQQ_Return_5d'] = qqq['QQQ'].pct_change(5)
    qqq['QQQ_Regime']    = (qqq['QQQ'] > qqq['QQQ_SMA_200']).astype(int)

    market = vix.join(spy[['SPY_Return','SPY_SMA_10','SPY_SMA_50','SPY_Trend','SPY_Return_5d','SPY_Regime']], how='inner')
    market = market.join(qqq[['QQQ_Return','QQQ_Trend','QQQ_Return_5d','QQQ_Regime']], how='inner')
    market.dropna(inplace=True)
    row = market.iloc[-1]
    regime_str = '✅ Bull' if row['SPY_Regime'] else '🚫 Bear'
    print(f"  VIX: {float(row['VIX']):.1f} | SPY: {'↑' if row['SPY_Trend'] else '↓'} | QQQ: {'↑' if row['QQQ_Trend'] else '↓'} | Regime: {regime_str}")
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
    df['Return_20d']  = df['Close'].pct_change(20)
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

_vader = SentimentIntensityAnalyzer()
_newsapi = None

def get_sentiment(ticker):
    global _newsapi
    if _newsapi is None:
        _newsapi = NewsApiClient(api_key=ENV['NEWS_KEY'])
    try:
        resp = _newsapi.get_everything(q=ticker, language='en', page_size=10, sort_by='publishedAt')
        articles = resp.get('articles', [])
        if not articles:
            return 0.0
        scores = []
        for a in articles:
            text = ' '.join(filter(None, [a.get('title', ''), a.get('description', '')]))
            scores.append(_vader.polarity_scores(text)['compound'])
        return sum(scores) / len(scores)
    except Exception:
        return 0.0

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

def fund_from_spy(needed, spy_state):
    """Sell just enough whole SPY shares to raise `needed` cash for an ML stock buy.
    SPY is the parking spot for spare cash, so ML signals take priority over it. Leaves the
    SPY high-water mark alone, so the remaining shares keep their trailing stop.
    Returns cash raised (0 if nothing was sold)."""
    if spy_state['qty'] < 1 or needed <= 0:
        return 0.0
    shares = min(math.ceil(needed / spy_state['price']), spy_state['qty'])
    print(f"  💱 Selling {shares} SPY (~${shares * spy_state['price']:,.2f}) to fund ML buy")
    order = api.submit_order(symbol='SPY', qty=shares, side='sell', type='market', time_in_force='day')
    for _ in range(15):                       # market orders normally fill within seconds
        time.sleep(2)
        order = api.get_order(order.id)
        if order.status == 'filled':
            raised = float(order.filled_qty) * float(order.filled_avg_price)
            spy_state['qty'] -= int(float(order.filled_qty))
            print(f"  ✅ SPY sale filled: ${raised:,.2f}")
            return raised
    try:
        api.cancel_order(order.id)
    except Exception:
        pass
    print(f"  ⚠️  SPY sale not filled ({order.status}) — cancelled, skipping buy")
    return 0.0

def run():
    account = api.get_account()
    cash    = float(account.cash)
    equity  = float(account.equity)
    print(f"💰 Cash: ${cash:,.2f}  |  Equity: ${equity:,.2f}")

    mkt = get_market_indicators()

    print("  📊 Fetching ticker data...")
    all_dfs     = {}
    all_returns = {}
    for ticker in TICKERS:
        try:
            df = get_features(ticker)
            all_dfs[ticker]     = df
            all_returns[ticker] = {
                'latest': float(df['Return'].iloc[-1]),
                '5d':     float(df['Return'].iloc[-5:].mean()),
            }
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")
            all_returns[ticker] = {'latest': 0.0, '5d': 0.0}

    positions = {p.symbol: p for p in api.list_positions()}
    stock_position_count = len([s for s, p in positions.items()   # ML stocks only, not SPY or crypto
                                if s != 'SPY' and p.asset_class == 'us_equity'])
    spy_pos   = positions.get('SPY')
    spy_state = {'qty': int(float(spy_pos.qty)) if spy_pos else 0,
                 'price': float(spy_pos.current_price) if spy_pos else 0.0}
    signals_logged = 0

    hw = load_highwater()
    # Remove stale entries for positions we no longer hold
    for sym in list(hw.keys()):
        if sym not in positions:
            del hw[sym]
    save_highwater(hw)

    for ticker in TICKERS:
        print(f"\n📊 {ticker}")
        if ticker not in all_dfs:
            print(f"  ⚠️  No data, skipping")
            continue
        try:
            df            = all_dfs[ticker]
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

                # Update high-water mark
                peak = hw.get(ticker, entry_price)
                if current_price > peak:
                    hw[ticker] = current_price
                    peak       = current_price
                    save_highwater(hw)

                trail_pct  = (current_price - peak) / peak
                trail_stop = peak * (1 - STOP_LOSS)
                locked_pct = (peak - entry_price) / entry_price
                print(f"  Position: {float(pos.qty):.4f} shares, entry ${entry_price:,.2f}, P&L {pnl_pct:+.1%} | Peak ${peak:,.2f} (+{locked_pct:.1%}) | Trail stop ${trail_stop:,.2f}")

                if HARD_STOP and pnl_pct <= -HARD_STOP:
                    print(f"  🛑 Hard stop — {pnl_pct:.1%} from entry ${entry_price:,.2f}")
                    api.submit_order(symbol=ticker, qty=float(pos.qty),
                                     side='sell', type='market', time_in_force='day')
                    log_signal(ticker, 'SELL', conf, sent_score, current_price, True, 'hard_stop')
                    signals_logged += 1
                    stock_position_count -= 1
                    if ticker in hw:
                        del hw[ticker]
                        save_highwater(hw)
                    continue

                if trail_pct <= -STOP_LOSS:
                    if (STOP_OVERRIDE_CONF and pred == 1 and conf >= STOP_OVERRIDE_CONF
                            and -STOP_OVERRIDE_BACKSTOP < pnl_pct < TAKE_PROFIT):
                        print(f"  ✋ Trailing stop hit ({trail_pct:.1%} from peak) but model still says BUY "
                              f"({conf:.1%}) — holding (sells regardless at {-STOP_OVERRIDE_BACKSTOP:.0%} from entry)")
                        log_signal(ticker, 'SELL', conf, sent_score, current_price, False, 'stop_override')
                        signals_logged += 1
                        continue
                    print(f"  🛑 Trailing stop — {trail_pct:.1%} from peak ${peak:,.2f} (locked {locked_pct:+.1%})")
                    api.submit_order(symbol=ticker, qty=float(pos.qty),
                                     side='sell', type='market', time_in_force='day')
                    log_signal(ticker, 'SELL', conf, sent_score, current_price, True, 'trailing_stop')
                    signals_logged += 1
                    stock_position_count -= 1
                    if ticker in hw:
                        del hw[ticker]
                        save_highwater(hw)
                    continue

                if pnl_pct >= TAKE_PROFIT:
                    print(f"  🎯 Take-profit triggered ({pnl_pct:.1%}) — SELLING")
                    api.submit_order(symbol=ticker, qty=float(pos.qty),
                                     side='sell', type='market', time_in_force='day')
                    log_signal(ticker, 'SELL', conf, sent_score, current_price, True, 'take_profit')
                    signals_logged += 1
                    stock_position_count -= 1
                    if ticker in hw:
                        del hw[ticker]
                        save_highwater(hw)
                    continue

                # ML SELL signals disabled — model predicts BUY outcomes only;
                # pred==0 means "unlikely to hit +20% TP", not "will fall".
                # Exits are handled exclusively by trailing stop and take-profit above.
                continue

            # --- BUY logic ---
            if pred == 1 and conf >= 0.60:
                if BUYS_PAUSED:
                    print(f"  ⏸️  Skipping BUY — stock buys paused (live picks trailing SPY)")
                    log_signal(ticker, 'BUY', conf, sent_score, current_price, False, 'buys_paused')
                    signals_logged += 1
                    continue

                if not mkt['SPY_Regime']:
                    print(f"  ⏸️  Skipping BUY — SPY below 200-day MA (bear regime)")
                    continue

                if float(mkt['VIX']) > VIX_BUY_THRESHOLD:
                    print(f"  ⏸️  Skipping BUY — high VIX ({float(mkt['VIX']):.1f})")
                    continue

                if stock_position_count >= MAX_POSITIONS:
                    print(f"  ⏸️  Skipping BUY — max positions reached ({MAX_POSITIONS})")
                    continue

                if sent_score < -0.05:
                    print(f"  ⏸️  Skipping BUY — negative news ({sent_score:.2f})")
                    log_signal(ticker, 'BUY', conf, sent_score, current_price, False, 'sentiment_override')
                    signals_logged += 1
                    continue

                target_size = equity * STOCK_POSITION_PCT
                if cash - 100 < target_size:
                    cash += fund_from_spy(target_size - (cash - 100), spy_state)

                if cash < 1000:
                    print(f"  ⏸️  Skipping BUY — insufficient cash (${cash:,.2f})")
                    continue

                position_size = min(target_size, cash - 100)
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
                stock_position_count += 1
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

    # --- Index default position (SPY) ---
    # Hold spare cash when no individual stock signals are firing. Exits on
    # its own trailing stop, VIX crash spike, or bear regime (200-day MA).
    print("\n📊 Index default position (SPY)")
    try:
        spy_price = float(get_features('SPY')['Close'].iloc[-1])

        # Refresh cash/positions in case the stock-buying loop above changed them
        account   = api.get_account()
        cash      = float(account.cash)
        positions = {p.symbol: p for p in api.list_positions()}

        # Cancel stale pending SPY buys; deduct other pending buys from
        # available cash so this default doesn't over-commit funds already spoken for.
        open_orders = api.list_orders(status='open')
        pending_spend = 0.0
        for o in open_orders:
            if o.symbol == 'SPY' and o.side == 'buy':
                api.cancel_order(o.id)
                print(f"  🗑️  Cancelled stale pending {o.symbol} buy order ({o.id[:8]}...)")
            elif o.side == 'buy':
                try:
                    o_price = float(get_features(o.symbol)['Close'].iloc[-1])
                except Exception:
                    o_price = 0.0
                pending_spend += float(o.qty) * o_price
        if pending_spend > 0:
            print(f"  ⚠️  Pending buy orders: ${pending_spend:,.2f} — deducting from available cash")
        cash -= pending_spend

        available  = cash - SPY_MIN_RESERVE
        spy_budget = max(available, 0)
        vix_now    = float(mkt['VIX'])

        def manage_default(symbol, price, budget, trail_stop_pct, vix_crash, bear_regime):
            print(f"\n  {symbol}:")

            def try_buy():
                if budget < 1000:
                    print(f"    ⏸️  Skipping buy — insufficient spare cash (${budget:,.2f})")
                elif bear_regime:
                    print(f"    ⏸️  Skipping buy — bear regime ({symbol} below 200-day MA)")
                elif vix_now > vix_crash:
                    print(f"    ⏸️  Skipping buy — VIX too high ({vix_now:.1f})")
                else:
                    qty = int(budget / price)  # whole shares only — no fractional restriction
                    if qty < 1:
                        print(f"    ⏸️  Skipping buy — insufficient cash for 1 share (${budget:,.2f} < ${price:,.2f})")
                    else:
                        print(f"    ✅ BUY {qty} {symbol} @ ${price:.2f} = ${qty * price:,.2f} (spare cash)")
                        api.submit_order(symbol=symbol, qty=qty, side='buy', type='market', time_in_force='day')

            pos = positions.get(symbol)
            if pos:
                entry   = float(pos.avg_entry_price)
                qty     = float(pos.qty)
                pnl_pct = (price - entry) / entry
                peak    = hw.get(symbol, entry)
                if price > peak:
                    hw[symbol] = price
                    peak       = price
                    save_highwater(hw)
                trail_pct  = (price - peak) / peak
                trail_stop = peak * (1 - trail_stop_pct)
                print(f"    Holding {qty:.4f} shares, entry ${entry:,.2f}, P&L {pnl_pct:+.1%} | Trail stop ${trail_stop:,.2f}")

                if trail_pct <= -trail_stop_pct:
                    print(f"    🛑 Trailing stop — {trail_pct:.1%} from peak ${peak:,.2f}")
                    api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                    hw.pop(symbol, None)
                    save_highwater(hw)
                elif vix_now > vix_crash:
                    print(f"    🚨 Crash exit — VIX={vix_now:.1f} above crash threshold ({vix_crash})")
                    api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                    hw.pop(symbol, None)
                    save_highwater(hw)
                elif bear_regime:
                    print(f"    🚫 Bear regime exit — {symbol} below 200-day MA")
                    api.submit_order(symbol=symbol, qty=qty, side='sell', type='market', time_in_force='day')
                    hw.pop(symbol, None)
                    save_highwater(hw)
                else:
                    print(f"    ✅ Holding — no exit conditions triggered")
                    try_buy()  # top up with any newly freed spare cash
            else:
                try_buy()

        manage_default('SPY', spy_price, spy_budget, SPY_TRAIL_STOP, SPY_VIX_CRASH, not mkt['SPY_Regime'])

    except Exception as e:
        print(f"  ❌ Index default position error: {e}")

    print("=" * 50)
    print("✅ Done!")
    print("=" * 50)

if __name__ == "__main__":
    run()
    