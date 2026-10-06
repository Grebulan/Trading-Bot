import json
import os
import time
import joblib
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime
from flask import Flask, render_template_string, request, Response, session, redirect, url_for
import functools
import secrets


app = Flask(__name__)
def _load_env_value(name):
    try:
        with open(os.path.expanduser('~/.env_trading')) as f:
            for line in f:
                if line.startswith(name + '='):
                    return line.strip().split('=', 1)[1]
    except Exception:
        pass
    return None

app.secret_key = _load_env_value('DASHBOARD_SECRET_KEY') or secrets.token_hex(32)

# No fallback: if it's missing from ~/.env_trading, nobody can log in
DASHBOARD_PASSWORD = _load_env_value('DASHBOARD_PASSWORD')
SESSION_DAYS = 30

LOGIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trading Bot — Login</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0a0c14; color: #e0e6ff;
           font-family: -apple-system, 'Segoe UI', sans-serif;
           display: flex; align-items: center; justify-content: center;
           min-height: 100vh; }
    .card { background: #111420; border: 1px solid #1e2540;
            border-radius: 12px; padding: 36px 32px; width: 320px; }
    h1 { font-size: 1.1em; font-weight: 700; margin-bottom: 24px;
         color: #e0e6ff; text-align: center; }
    h1 span { color: #448aff; }
    input[type=password] {
      width: 100%; padding: 10px 14px; border-radius: 7px;
      background: #0d1022; border: 1px solid #2a2d4a;
      color: #e0e6ff; font-size: 1em; margin-bottom: 16px; outline: none;
    }
    input[type=password]:focus { border-color: #448aff; }
    button {
      width: 100%; padding: 11px; border-radius: 7px;
      background: #448aff; border: none; color: #fff;
      font-size: 1em; font-weight: 700; cursor: pointer;
    }
    button:hover { background: #5a9fff; }
    .err { color: #ff5252; font-size: 0.82em; margin-bottom: 12px; text-align: center; }
  </style>
</head>
<body>
<div class="card">
  <h1>📈 <span>Trading</span> Bot</h1>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  <form method="post" action="/login">
    <input type="password" name="password" placeholder="Password" autofocus>
    <button type="submit">Sign in</button>
  </form>
</div>
</body>
</html>"""


def requires_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('authed'):
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated

STOCK_SENTIMENT_CACHE = '/home/jv/stock_sentiment_cache.json'
STOCK_SENTIMENT_TTL   = 3600  # 1 hour

def fmt_qty(q):
    """Format a share/coin quantity: whole numbers stay bare, fractional
    quantities show up to 2 decimal places with trailing zeros trimmed."""
    r = round(float(q), 2)
    if r == int(r):
        return str(int(r))
    return f"{r:.2f}".rstrip('0').rstrip('.')

def load_stock_sentiment_cache():
    if os.path.exists(STOCK_SENTIMENT_CACHE):
        with open(STOCK_SENTIMENT_CACHE) as f:
            return json.load(f)
    return {}

def _parse_history_time(t):
    """Parse a portfolio history timestamp — handles both ISO and legacy MM/DD HH:MM.

    Python 3.9/3.10 fromisoformat() cannot handle timezone-aware strings
    (e.g. '2025-04-04T14:00:00+00:00').  We strip the trailing tz offset
    before parsing so the comparison arithmetic stays consistent (all naive,
    treated as local/UTC — doesn't matter as long as it's consistent).
    """
    from datetime import datetime
    import re
    if not t:
        return None
    try:
        if 'T' in t or (len(t) > 5 and t[4] == '-'):
            # Strip trailing timezone offset (+HH:MM or Z) for 3.9/3.10 compat
            ts = re.sub(r'([+-]\d{2}:\d{2}|Z)$', '', t)
            return datetime.fromisoformat(ts)
        # Legacy format: MM/DD HH:MM — assume current year
        year = datetime.now().year
        return datetime.strptime(f'{year}/{t}', '%Y/%m/%d %H:%M')
    except Exception:
        return None

def build_chart_views(history_file='/home/jv/portfolio_history.json'):
    """
    Return resampled views of portfolio_history.json suitable for
    chart rendering.  Each view is a dict with parallel lists:
      labels, values, stock_values, crypto_values

    24h  — raw 30-min data for the last 24 real hours
    7d   — 1 point per 30 min for the last 7 days
    30d  — 1 point per 2 hours for the last 30 days
    3m   — 1 point per 6 hours for the last 90 days
    6m   — 1 point per 12 hours for the last 182 days
    all  — 1 point per hour for all available data

    Entries marked 'backfilled' were filled in from Alpaca's account history
    and only have a total value — their stock/crypto/cash are None, which
    Chart.js draws as a gap rather than a misleading zero.

    Resampling to equal density stops old sparse data and new dense data
    from producing wildly different visual weights on a category-scale chart.
    """
    from datetime import datetime, timedelta

    if not os.path.exists(history_file):
        empty = dict(labels=[], values=[], stock_values=[], crypto_values=[], cash_values=[])
        return {'24h': empty, '7d': empty, '30d': empty, '3m': empty, '6m': empty, 'all': empty}

    with open(history_file) as f:
        history = json.load(f)

    if not history:
        empty = dict(labels=[], values=[], stock_values=[], crypto_values=[], cash_values=[])
        return {'24h': empty, '7d': empty, '30d': empty, '3m': empty, '6m': empty, 'all': empty}

    def to_row(h):
        return {
            'time':   h['time'],
            'value':  h['value'],
            'stock':  None if h.get('backfilled') else h.get('stock_value', 0),
            'crypto': None if h.get('backfilled') else h.get('crypto_value', 0),
            'cash':   None if h.get('backfilled') else h.get('cash', 0),
            'dt':     _parse_history_time(h['time']),
        }

    rows = [to_row(h) for h in history]
    rows = [r for r in rows if r['dt'] is not None]

    latest = rows[-1]['dt']

    def resample(data, interval_hours):
        """Keep the last entry that falls within each N-hour bucket."""
        epoch = datetime(2000, 1, 1)
        buckets = {}
        for r in data:
            delta    = r['dt'] - epoch
            bucket_n = int(delta.total_seconds() / (interval_hours * 3600))
            buckets[bucket_n] = r          # last entry per bucket wins
        result = [buckets[k] for k in sorted(buckets)]
        return result

    def fmt_label(r):
        """Human-readable label for chart x-axis."""
        dt = r['dt']
        return dt.strftime('%d/%m %H:%M')

    def make_view(data):
        return {
            'labels':       [fmt_label(r) for r in data],
            'timestamps':   [int(r['dt'].timestamp() * 1000) for r in data],
            'values':       [r['value']  for r in data],
            'stock_values': [r['stock']  for r in data],
            'crypto_values':[r['crypto'] for r in data],
            'cash_values':  [None if r['stock'] is None
                             else max(0, round(r['value'] - r['stock'] - r['crypto'], 2)) for r in data],
        }

    earliest   = rows[0]['dt']
    span_hours = (latest - earliest).total_seconds() / 3600

    # 24H — raw data, last 24 hours (cap at 200 points to avoid chart clutter)
    cutoff_24h = latest - timedelta(hours=24)
    raw_24h    = [r for r in rows if r['dt'] >= cutoff_24h]
    view_24h   = make_view(raw_24h if raw_24h else rows[-200:])

    # 7D: if we have >24h of data, resample to 30-min buckets; else show all raw
    cutoff_7d  = latest - timedelta(hours=168)
    raw_7d     = [r for r in rows if r['dt'] >= cutoff_7d]
    view_7d    = make_view(resample(raw_7d, 0.5) if span_hours > 24 else raw_7d)

    # 30D: resample to 2h buckets to keep the point count manageable (~360)
    cutoff_30d = latest - timedelta(days=30)
    raw_30d    = [r for r in rows if r['dt'] >= cutoff_30d]
    view_30d   = make_view(resample(raw_30d, 2) if span_hours > 48 else raw_30d)

    # 3M / 6M: coarser buckets keep both views at roughly 360 points
    cutoff_3m = latest - timedelta(days=90)
    view_3m   = make_view(resample([r for r in rows if r['dt'] >= cutoff_3m], 6))
    cutoff_6m = latest - timedelta(days=182)
    view_6m   = make_view(resample([r for r in rows if r['dt'] >= cutoff_6m], 12))

    # ALL: resample to 1h buckets if >48h of data, 30-min if >24h, else raw
    if span_hours > 48:
        view_all = make_view(resample(rows, 1))
    elif span_hours > 24:
        view_all = make_view(resample(rows, 0.5))
    else:
        view_all = make_view(rows)

    return {'24h': view_24h, '7d': view_7d, '30d': view_30d, '3m': view_3m, '6m': view_6m, 'all': view_all}

def get_stock_sentiment_cached(ticker):
    """Return cached sentiment score, or 0.0 if cache missing/stale."""
    cache = load_stock_sentiment_cache()
    entry = cache.get(ticker)
    if entry and (time.time() - entry.get('fetched', 0)) < STOCK_SENTIMENT_TTL:
        return entry['score']
    return None   # None = no fresh data (show — rather than 0.0)

def get_crypto_sentiment_cached(symbol):
    """Read the existing crypto sentiment cache directly — no live call."""
    try:
        cache_file = '/home/jv/crypto_sentiment_cache.json'
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cache = json.load(f)
            entry = cache.get(symbol)
            if entry:
                return entry['score']
    except Exception:
        pass
    return None

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
TICKERS  = [
    'AAPL','GOOGL','AMZN','PLTR','CRWD',                  # Tech
    'NVDA','AMD','QCOM','ARM','MU',                       # Semiconductors
    'TSLA','SHOP','COIN',                                 # Growth
    'JPM','BAC','GS','MS','STT',                         # Finance
    'OXY',                                                # Energy
    'GOLD',                                               # Commodities
    'UNH','ISRG',                                         # Healthcare
    'CAT','VRT',                                          # Industrials
]
COMPANY_NAMES = {
    'AAPL':     'Apple Inc.',
    'GOOGL':    'Alphabet (Google)',
    'AMZN':     'Amazon',
    'PLTR':     'Palantir Technologies',
    'CRWD':     'CrowdStrike Holdings',
    'NVDA':     'NVIDIA',
    'AMD':      'Advanced Micro Devices',
    'QCOM':     'Qualcomm',
    'ARM':      'Arm Holdings',
    'MU':       'Micron Technology',
    'TSLA':     'Tesla',
    'SHOP':     'Shopify',
    'COIN':     'Coinbase Global',
    'JPM':      'JPMorgan Chase',
    'BAC':      'Bank of America',
    'GS':       'Goldman Sachs',
    'MS':       'Morgan Stanley',
    'STT':      'State Street Corporation',
    'OXY':      'Occidental Petroleum',
    'GOLD':     'Barrick Gold',
    'UNH':      'UnitedHealth Group',
    'ISRG':     'Intuitive Surgical',
    'CAT':      'Caterpillar Inc.',
    'VRT':      'Vertiv Holdings',
    'BTC/USD':  'Bitcoin',
    'ETH/USD':  'Ethereum',
    'SOL/USD':  'Solana',
    'LINK/USD': 'Chainlink',
    'AVAX/USD': 'Avalanche',
    'XRP/USD':  'XRP (Ripple)',
    'DOGE/USD': 'Dogecoin',
    'UNI/USD':  'Uniswap',
    'DOT/USD':  'Polkadot',
    'ADA/USD':  'Cardano',
}
CRYPTOS  = {
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

api          = tradeapi.REST(ENV['ALPACA_KEY'], ENV['ALPACA_SECRET'], BASE_URL, api_version='v2')
stock_model  = joblib.load('/home/jv/trading_model.joblib')
crypto_model = joblib.load('/home/jv/crypto_model.joblib')

with open('/home/jv/features.json') as f:
    stock_features = json.load(f)
with open('/home/jv/crypto_features.json') as f:
    crypto_features = json.load(f)

def get_market_indicators():
    try:
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

        btc = yf.download('BTC-USD', period="30d", interval="1h", progress=False)
        btc.columns = [col[0] if isinstance(col, tuple) else col for col in btc.columns]
        btc = btc[['Close']].rename(columns={'Close': 'BTC'})
        btc['BTC_SMA_24']  = btc['BTC'].rolling(24).mean()
        btc['BTC_SMA_168'] = btc['BTC'].rolling(168).mean()
        btc['BTC_Trend']   = (btc['BTC_SMA_24'] > btc['BTC_SMA_168']).astype(int)
        btc_trend          = int(btc['BTC_Trend'].iloc[-1])

        market = vix.join(spy[['SPY_Return','SPY_SMA_10','SPY_SMA_50','SPY_Trend','SPY_Return_5d']], how='inner')
        market = market.join(qqq[['QQQ_Return','QQQ_Trend','QQQ_Return_5d']], how='inner')
        market.dropna(inplace=True)
        return market.iloc[-1], btc_trend
    except Exception as e:
        print(f"Market indicator error: {e}")
        return None, 0

def get_crypto_market_indicators():
    try:
        vix = yf.download('^VIX', period="10d", interval="1d", progress=False)
        vix.columns = [col[0] if isinstance(col, tuple) else col for col in vix.columns]
        vix = vix[['Close']].rename(columns={'Close': 'VIX'})

        spy = yf.download('SPY', period="30d", interval="1h", progress=False)
        spy.columns = [col[0] if isinstance(col, tuple) else col for col in spy.columns]
        spy = spy[['Close']].rename(columns={'Close': 'SPY'})
        spy.index = pd.to_datetime(spy.index).tz_convert('UTC')
        spy = spy.resample('1h').last().ffill()
        spy['SPY_Return']     = spy['SPY'].pct_change()
        spy['SPY_SMA_24']     = spy['SPY'].rolling(24).mean()
        spy['SPY_SMA_168']    = spy['SPY'].rolling(168).mean()
        spy['SPY_Trend']      = (spy['SPY_SMA_24'] > spy['SPY_SMA_168']).astype(int)
        spy['SPY_Return_24h'] = spy['SPY'].pct_change(24)

        btc = yf.download('BTC-USD', period="30d", interval="1h", progress=False)
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
        print(f"Crypto market indicator error: {e}")
        return None

def get_stock_features(ticker):
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

def get_crypto_features(yf_symbol):
    df = yf.download(yf_symbol, period="30d", interval="1h", progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df['SMA_24']        = df['Close'].rolling(24).mean()
    df['SMA_72']        = df['Close'].rolling(72).mean()
    df['SMA_168']       = df['Close'].rolling(168).mean()
    df['EMA_12']        = df['Close'].ewm(span=12).mean()
    df['EMA_26']        = df['Close'].ewm(span=26).mean()
    df['MACD']          = df['EMA_12'] - df['EMA_26']
    df['RSI']           = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(14).mean() /
                                            df['Close'].diff().clip(upper=0).abs().rolling(14).mean()))
    df['Return']        = df['Close'].pct_change()
    df['Return_24h']    = df['Close'].pct_change(24)
    df['Return_72h']    = df['Close'].pct_change(72)
    df['Return_168h']   = df['Close'].pct_change(168)
    df['Volatility']    = df['Return'].rolling(24).std()
    df['High_Low']      = (df['High'] - df['Low']) / df['Close']
    bb_mid              = df['Close'].rolling(48).mean()
    bb_std              = df['Close'].rolling(48).std()
    df['BB_upper']      = bb_mid + (2 * bb_std)
    df['BB_lower']      = bb_mid - (2 * bb_std)
    df['BB_position']   = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    df['BB_width']      = (df['BB_upper'] - df['BB_lower']) / bb_mid
    df['Volume_MA']     = df['Volume'].rolling(24).mean()
    df['Vol_Ratio']     = df['Volume'] / df['Volume_MA']
    df['Vol_Spike']     = (df['Vol_Ratio'] > 2.0).astype(int)
    df['HourOfDay']     = pd.to_datetime(df.index).hour
    df['DayOfWeek']     = pd.to_datetime(df.index).dayofweek
    df['IsWeekend']     = (df['DayOfWeek'] >= 5).astype(int)
    df['Overnight_Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
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

def get_crypto_momentum(all_returns):
    if not all_returns:
        return 0.0, 0.0
    latest = sum(v['latest'] for v in all_returns.values()) / len(all_returns)
    h24    = sum(v['24h'] for v in all_returns.values()) / len(all_returns)
    return latest, h24

HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>🤖 Trading Bot Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="300">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, sans-serif; background: #0f1117; color: #fff; }
    .topnav { display: flex; align-items: center; justify-content: space-between;
              background: #1a1d2e; padding: 8px 20px; margin-bottom: 15px;
              border-bottom: 2px solid #252840; position: sticky; top: 0; z-index: 100; }
    .topnav h1 { font-size: 1.1em; display: inline; }
    .topnav .subtitle { color: #9aa3b5; font-size: 0.78em; margin-left: 14px; }
    .nav-buttons { display: flex; gap: 10px; }
    .nav-btn { padding: 8px 20px; border-radius: 8px; cursor: pointer; font-weight: bold;
               font-size: 0.95em; border: 2px solid transparent; transition: all 0.2s;
               background: #252840; color: #9aa3b5; }
    .nav-btn.active { color: #fff; border-color: #448aff; background: #2e3250; }
    .page { display: none; padding: 0 20px 20px 20px; }
    .page.active { display: block; }
    .cards-row { display: flex; gap: 15px; margin-bottom: 25px; align-items: stretch; }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
             gap: 15px; flex: 1; }
    .card { background: #1a1d2e; border-radius: 12px; padding: 18px; }
    .card h2 { font-size: 0.75em; color: #9aa3b5; margin-bottom: 8px; }
    .card .value { font-size: 1.4em; font-weight: bold; }
    .alloc-chart-wrap { background: #1a1d2e; border-radius: 12px; padding: 14px 18px;
                        display: flex; flex-direction: column; justify-content: center;
                        min-width: 220px; max-width: 260px; }
    .alloc-chart-wrap h2 { font-size: 0.75em; color: #9aa3b5; margin-bottom: 8px; }
    .alloc-inner { display: flex; align-items: center; gap: 12px; }
    .alloc-inner canvas { width: 90px !important; height: 90px !important; flex-shrink: 0; }
    .alloc-legend { font-size: 0.75em; color: #aaa; line-height: 2; white-space: nowrap; }
    .card .value.small { font-size: 1em; }
    .green { color: #00c853; } .red { color: #ff4d6a; }
    .yellow { color: #ffd600; } .blue { color: #448aff; }
    .market-bar { background: #1a1d2e; border-radius: 12px; padding: 18px;
                  margin-bottom: 25px; display: flex; gap: 30px; flex-wrap: wrap; }
    .market-item h2 { font-size: 0.75em; color: #9aa3b5; margin-bottom: 4px; }
    .market-item .val { font-size: 1.2em; font-weight: bold; }
    .vix-low { color: #00c853; } .vix-mid { color: #ffd600; } .vix-high { color: #ff4d6a; }
    .ticker-wrap { background: #1a1d2e; border-radius: 10px; padding: 12px;
                   margin-bottom: 25px; overflow: hidden; }
    #ticker { display: flex; gap: 40px; white-space: nowrap;
              animation: scroll 60s linear infinite; }
    #ticker:hover { animation-play-state: paused; }
    @keyframes scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
    .chart-wrap { background: #1a1d2e; border-radius: 12px; padding: 20px; margin-bottom: 25px; }
    .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .chart-header h2 { font-size: 1em; color: #9aa3b5; }
    .chart-toggles { display: flex; gap: 8px; }
    .chart-toggle { padding: 4px 12px; border-radius: 20px; font-size: 0.75em;
                    font-weight: bold; cursor: pointer; border: 2px solid; transition: all 0.2s;
                    background: #0f1117; }
    .chart-toggle.total  { border-color: #00c853; color: #00c853; background: #0f1117; }
    .chart-toggle.stocks { border-color: #448aff; color: #448aff; background: #0f1117; }
    .chart-toggle.crypto { border-color: #ffd600; color: #ffd600; background: #0f1117; }
    .chart-toggle.cash   { border-color: #b0b8d8; color: #b0b8d8; background: #0f1117; }
    .chart-toggle.active.total  { background: #00c853 !important; color: #000; }
    .chart-toggle.active.stocks { background: #448aff !important; color: #000; }
    .chart-toggle.active.crypto { background: #ffd600 !important; color: #000; }
    .chart-toggle.active.cash   { background: #b0b8d8 !important; color: #000; }
    .table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch;
                    margin-bottom: 25px; border-radius: 12px; }
    table { width: 100%; border-collapse: collapse; background: #1a1d2e;
            border-radius: 12px; overflow: hidden; margin-bottom: 0; }
    th { background: #252840; padding: 12px 15px; text-align: left; font-size: 0.8em; color: #9aa3b5; }
    td { padding: 12px 15px; border-top: 1px solid #252840; font-size: 0.9em; }
    .badge { padding: 3px 10px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }
    .badge.buy   { background: #00c85322; color: #00c853; }
    .badge.sell  { background: #ff4d6a22; color: #ff4d6a; }
    .badge.hold  { background: #ffd60022; color: #ffd600; }
    .badge.error { background: #9aa3b522; color: #9aa3b5; }
    .section-title { font-size: 1.1em; margin: 0 0 15px 0; }
    /* Tooltip */
    .tip { position: relative; display: inline-block; cursor: default;
           border-bottom: 1px dotted #448aff; }
    .tip .tiptext { visibility: hidden; opacity: 0; background: #2e3250; color: #fff;
                    font-size: 0.8em; font-weight: normal; border: 1px solid #448aff;
                    border-radius: 6px; padding: 5px 10px; white-space: nowrap;
                    position: absolute; z-index: 200; bottom: 130%; left: 50%;
                    transform: translateX(-50%); transition: opacity 0.15s;
                    pointer-events: none; }
    .tip:hover .tiptext { visibility: visible; opacity: 1; }
    /* ── Mobile: compact header ── */
    @media (max-width: 600px) {
      .topnav { padding: 6px 12px; flex-wrap: nowrap; gap: 6px; }
      .topnav h1 { font-size: 0.95em; }
      .topnav .subtitle { display: none; }
      .nav-header-right { flex-direction: column; gap: 2px; align-items: flex-end; }
      .nav-header-right .nav-countdown { font-size: 0.68em; }
      .nav-header-right .nav-counts { display: none; }
      .nav-buttons { gap: 6px; }
      .nav-btn { padding: 6px 12px; font-size: 0.82em; }
      .ticker-wrap { padding: 7px 10px; margin-bottom: 15px; }
      #ticker { font-size: 0.85em; gap: 24px; }
      .page { padding: 0 10px 16px 10px; }
      .cards-row { flex-direction: column; }
      .alloc-chart-wrap { min-width: unset; max-width: unset; width: 100%; }
      .cards { grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; }
      .card { padding: 12px; }
      .card .value { font-size: 1.2em; }
      .market-bar { padding: 12px; gap: 16px; margin-bottom: 15px; }
      .chart-wrap { padding: 12px; }
      .chart-header { flex-wrap: wrap; gap: 6px; }
      .chart-header h2 { width: 100%; margin-bottom: 0; }
      .chart-toggle { padding: 3px 8px; font-size: 0.7em; }
      th { padding: 8px 10px; }
      td { padding: 8px 10px; font-size: 0.82em; }
      body { overflow-x: hidden; }
    }
  </style>
</head>
<body>

  <!-- STICKY TOP NAV -->
  <div class="topnav">
    <div style="display:flex;align-items:center;gap:16px">
      <span class="topnav h1">🤖 Trading Bot</span>
      <span class="subtitle">Updated: {{ updated }} &bull; auto-refresh 5min</span>
      <a href="/milestones" style="padding:4px 12px;border-radius:6px;background:#1a1d2e;border:1px solid #2a2d4a;color:#9aa3b5;font-size:0.78em;font-weight:bold;text-decoration:none" onmouseover="this.style.color='#fff'" onmouseout="this.style.color='#9aa3b5'">📍 Milestones</a>
      <a href="/review" style="padding:4px 12px;border-radius:6px;background:#1a1d2e;border:1px solid #2a2d4a;color:#9aa3b5;font-size:0.78em;font-weight:bold;text-decoration:none" onmouseover="this.style.color='#fff'" onmouseout="this.style.color='#9aa3b5'">🧐 Weekly Review</a>
    </div>
    <div class="nav-header-right" style="display:flex;align-items:center;gap:20px">
      <span class="nav-countdown" style="font-size:0.78em;color:#9aa3b5">
        📈 <span id="nav-stock-countdown" style="font-weight:bold">--:--</span>
      </span>
      <span class="nav-countdown" style="font-size:0.78em;color:#9aa3b5">
        🪙 <span id="nav-crypto-countdown" style="font-weight:bold">--:--</span>
      </span>
      <span class="nav-counts" style="color:#8c93a6;font-size:0.82em">{{ stocks|length }}s · {{ crypto|length }}c</span>
    </div>
  </div>

  <!-- SHARED: ticker tape always visible -->
  <div style="padding: 0 20px;">
    <div class="ticker-wrap">
      <div id="ticker">
        {% for s in stocks + crypto %}
        <span style="font-size:0.95em">
          <span class="tip"><strong>{{ s.ticker }}</strong><span class="tiptext">{{ s.name }}</span></span>
          <span style="color:#9aa3b5">${{ s.price }}</span>
          {% if s.signal == 'BUY' %}<span style="color:#00c853">▲</span>
          {% elif s.signal == 'SELL' %}<span style="color:#ff4d6a">▼</span>
          {% else %}<span style="color:#ffd600">—</span>{% endif %}
        </span>
        {% endfor %}
        {% for s in stocks + crypto %}
        <span style="font-size:0.95em">
          <span class="tip"><strong>{{ s.ticker }}</strong><span class="tiptext">{{ s.name }}</span></span>
          <span style="color:#9aa3b5">${{ s.price }}</span>
          {% if s.signal == 'BUY' %}<span style="color:#00c853">▲</span>
          {% elif s.signal == 'SELL' %}<span style="color:#ff4d6a">▼</span>
          {% else %}<span style="color:#ffd600">—</span>{% endif %}
        </span>
        {% endfor %}
      </div>
    </div>
  </div>

  <!-- COMBINED PAGE -->
  <div id="page-main" class="page active">

    <div class="cards-row">
      <div class="cards">
        <div class="card"><h2>💰 CASH</h2><div class="value green">${{ cash }}</div></div>
        <div class="card"><h2>📈 PORTFOLIO</h2><div class="value">${{ portfolio }}</div></div>
        <div class="card"><h2>📊 POSITIONS</h2><div class="value yellow">{{ position_count }}</div></div>
        <div class="card"><h2>🎯 TODAY P&L</h2>
          <div class="value {{ 'green' if pnl >= 0 else 'red' }}">${{ pnl }}</div></div>
        <div class="card"><h2>🧠 STOCK MODEL</h2>
          <div class="value small blue">{{ stock_model_info.get('accuracy','—') }}</div></div>
        <div class="card"><h2>🪙 CRYPTO MODEL</h2>
          <div class="value small blue">{{ crypto_model_info.get('accuracy','—') }}</div></div>
        <div class="card"><h2>📅 TRAINED</h2>
          <div class="value small">{{ stock_model_info.get('trained_at','—') }}</div></div>
      </div>
      <div class="alloc-chart-wrap">
        <h2>📊 ALLOCATION</h2>
        <div class="alloc-inner">
          <canvas id="allocChart" width="100" height="100"></canvas>
          <div class="alloc-legend">
            <span style="color:#448aff">■</span> Stocks {{ stock_pct }}%<br>
            <span style="color:#c084fc">■</span> SPY {{ spy_pct }}%<br>
            <span style="color:#ffd600">■</span> Crypto {{ crypto_pct }}%<br>
            <span style="color:#00c853">■</span> Cash {{ cash_pct }}%
          </div>
        </div>
      </div>
    </div>

    {% if market %}
    <div class="market-bar">
      <div class="market-item">
        <h2>😱 VIX</h2>
        <div class="val {% if market.vix < 20 %}vix-low{% elif market.vix < 30 %}vix-mid{% else %}vix-high{% endif %}">
          {{ market.vix }} {% if market.vix < 20 %}😊{% elif market.vix < 30 %}😐{% else %}😱{% endif %}
        </div>
      </div>
      <div class="market-item"><h2>📊 S&P 500</h2>
        <div class="val {{ 'green' if market.spy_trend else 'red' }}">
          {{ '↑ Up' if market.spy_trend else '↓ Down' }}
          <span style="font-size:0.6em">({{ '+' if market.spy_24h >= 0 else '' }}{{ market.spy_24h }}% 24h)</span></div></div>
      <div class="market-item"><h2>💻 NASDAQ</h2>
        <div class="val {{ 'green' if market.qqq_trend else 'red' }}">
          {{ '↑ Up' if market.qqq_trend else '↓ Down' }}
          <span style="font-size:0.6em">({{ '+' if market.qqq_24h >= 0 else '' }}{{ market.qqq_24h }}% 24h)</span></div></div>
      <div class="market-item"><h2>₿ BITCOIN</h2>
        <div class="val {{ 'green' if market.btc_trend else 'red' }}">
          {{ '↑ Up' if market.btc_trend else '↓ Down' }}</div></div>
      <div class="market-item"><h2>📅 SPY 5D</h2>
        <div class="val {{ 'green' if market.spy_5d >= 0 else 'red' }}">
          {{ '+' if market.spy_5d >= 0 else '' }}{{ market.spy_5d }}%</div></div>
      <div class="market-item"><h2>📅 QQQ 5D</h2>
        <div class="val {{ 'green' if market.qqq_5d >= 0 else 'red' }}">
          {{ '+' if market.qqq_5d >= 0 else '' }}{{ market.qqq_5d }}%</div></div>
    </div>
    {% endif %}

    {% if chart_labels %}
    <div class="chart-wrap">
      <div class="chart-header">
        <h2>📈 PORTFOLIO PERFORMANCE</h2>
        <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
          <div style="display:flex;background:#252840;border-radius:20px;padding:2px">
            <div id="pvbtn-24h" style="padding:3px 12px;border-radius:18px;font-size:0.72em;font-weight:bold;cursor:pointer;color:#9aa3b5;transition:all 0.2s" onclick="setPortfolioView('24h')">24H</div>
            <div id="pvbtn-7d"  style="padding:3px 12px;border-radius:18px;font-size:0.72em;font-weight:bold;cursor:pointer;background:#2e3250;color:#fff;transition:all 0.2s" onclick="setPortfolioView('7d')">7D</div>
            <div id="pvbtn-30d" style="padding:3px 12px;border-radius:18px;font-size:0.72em;font-weight:bold;cursor:pointer;color:#9aa3b5;transition:all 0.2s" onclick="setPortfolioView('30d')">30D</div>
            <div id="pvbtn-3m"  style="padding:3px 12px;border-radius:18px;font-size:0.72em;font-weight:bold;cursor:pointer;color:#9aa3b5;transition:all 0.2s" onclick="setPortfolioView('3m')">3M</div>
            <div id="pvbtn-6m"  style="padding:3px 12px;border-radius:18px;font-size:0.72em;font-weight:bold;cursor:pointer;color:#9aa3b5;transition:all 0.2s" onclick="setPortfolioView('6m')">6M</div>
          </div>
          <div class="chart-toggles">
            <div class="chart-toggle total active"  onclick="toggleLine(chart, 0, this)">Total</div>
            <div class="chart-toggle stocks"        onclick="toggleLine(chart, 1, this)">📈 Stocks</div>
            <div class="chart-toggle crypto"        onclick="toggleLine(chart, 2, this)">🪙 Crypto</div>
            <div class="chart-toggle cash"          onclick="toggleLine(chart, 3, this)">💵 Cash</div>
          </div>
        </div>
      </div>
      <canvas id="portfolioChart" height="120"></canvas>
    </div>
    {% endif %}

    <h2 class="section-title">📂 Open Positions</h2>
    <div class="table-scroll"><table>
      <tr><th>ASSET</th><th>TYPE</th><th>QTY</th><th>BOUGHT AT</th><th>CURRENT</th><th>VALUE</th><th>UNREALISED P&L</th><th>TRAIL STOP</th><th>GAP</th></tr>
      {% for p in stock_positions %}
      {% if p.is_spy %}
      <tr style="background:rgba(100,181,246,0.10)">
        <td><span class="tip"><strong>{{ p.symbol }}</strong><span class="tiptext">{{ p.name }}</span></span></td>
        <td style="color:#64b5f6;font-size:0.8em">🏦 SPY Default</td>
        <td>{{ p.qty }} sh</td>
        <td>${{ p.avg_price }}</td>
        <td>${{ p.current }}</td>
        <td>${{ p.value }}</td>
        <td class="{{ 'green' if p.pl >= 0 else 'red' }}">{{ '▲' if p.pl >= 0 else '▼' }} ${{ p.pl_abs }} ({{ p.pl_pct }}%)</td>
        <td style="color:#f0a500">${{ p.trail_stop }} <span style="color:#8c93a6;font-size:0.8em">(5%)</span></td>
        <td style="color:{{ '#ff6b57' if p.gap_pct|float < 5 else '#9aa3b5' }}">{{ p.gap_pct }}%</td>
      </tr>
      {% else %}
      <tr style="background:rgba(68,138,255,0.07)">
        <td><span class="tip"><strong>{{ p.symbol }}</strong><span class="tiptext">{{ p.name }}</span></span></td>
        <td style="color:#448aff;font-size:0.8em">📈 Stock</td>
        <td>{{ p.qty }} sh</td>
        <td>${{ p.avg_price }}</td>
        <td>${{ p.current }}</td>
        <td>${{ p.value }}</td>
        <td class="{{ 'green' if p.pl >= 0 else 'red' }}">{{ '▲' if p.pl >= 0 else '▼' }} ${{ p.pl_abs }} ({{ p.pl_pct }}%)</td>
        <td style="color:#f0a500">${{ p.trail_stop }}</td>
        <td style="color:{{ '#ff6b57' if p.gap_pct|float < 5 else '#9aa3b5' }}">{{ p.gap_pct }}%</td>
      </tr>
      {% endif %}
      {% endfor %}
      {% for p in crypto_positions_list %}
      <tr style="background:rgba(255,214,0,0.06)">
        <td><span class="tip"><strong>{{ p.symbol }}</strong><span class="tiptext">{{ p.name }}</span></span></td>
        <td style="color:#ffd600;font-size:0.8em">🪙 Crypto</td>
        <td>{{ p.qty }} coins</td>
        <td>${{ p.avg_price }}</td>
        <td>${{ p.current }}</td>
        <td>${{ p.value }}</td>
        <td class="{{ 'green' if p.pl >= 0 else 'red' }}">{{ '▲' if p.pl >= 0 else '▼' }} ${{ p.pl_abs }} ({{ p.pl_pct }}%)</td>
        <td style="color:#f0a500">${{ p.trail_stop }}</td>
        <td style="color:{{ '#ff6b57' if p.gap_pct|float < 5 else '#9aa3b5' }}">{{ p.gap_pct }}%</td>
      </tr>
      {% else %}
        {% if not stock_positions %}
      <tr><td colspan="9" style="padding:20px;text-align:center;color:#9aa3b5">No open positions</td></tr>
        {% endif %}
      {% endfor %}
    </table></div>

    <h2 class="section-title">📡 Signals</h2>
    <div class="table-scroll"><table>
      <tr><th>ASSET</th><th>TYPE</th><th>PRICE</th><th>SIGNAL</th><th>CONFIDENCE</th><th>SENTIMENT</th><th>HOLDING</th></tr>
      {% for s in stocks %}
      <tr style="background:rgba(68,138,255,0.07)">
        <td><span class="tip"><strong>{{ s.ticker }}</strong><span class="tiptext">{{ s.name }}</span></span></td>
        <td style="color:#448aff;font-size:0.8em">📈 Stock</td>
        <td>${{ s.price }}</td>
        <td>{% if s.signal == 'BUY' %}<span class="badge buy">▲ BUY</span>
            {% elif s.signal == 'SELL' %}<span class="badge sell">▼ SELL</span>
            {% elif s.signal == 'HOLD' %}<span class="badge hold">— HOLD</span>
            {% else %}<span class="badge error">ERROR</span>{% endif %}</td>
        <td>{{ s.confidence }}</td>
        <td>{{ s.sentiment }}</td>
        <td>{{ s.shares }} sh</td>
      </tr>
      {% endfor %}
      {% for c in crypto %}
      <tr style="background:rgba(255,214,0,0.06)">
        <td><span class="tip"><strong>{{ c.ticker }}</strong><span class="tiptext">{{ c.name }}</span></span></td>
        <td style="color:#ffd600;font-size:0.8em">🪙 Crypto</td>
        <td>${{ c.price }}</td>
        <td>{% if c.signal == 'BUY' %}<span class="badge buy">▲ BUY</span>
            {% elif c.signal == 'SELL' %}<span class="badge sell">▼ SELL</span>
            {% elif c.signal == 'HOLD' %}<span class="badge hold">— HOLD</span>
            {% else %}<span class="badge error">ERROR</span>{% endif %}</td>
        <td>{{ c.confidence }}</td>
        <td>{{ c.sentiment }}</td>
        <td>{{ c.coins }} coins</td>
      </tr>
      {% endfor %}
    </table></div>

    <h2 class="section-title">📋 Recent Orders</h2>
    <div class="table-scroll"><table>
      <tr><th>TIME</th><th>ASSET</th><th>TYPE</th><th>ACTION</th><th>QTY</th><th>PRICE</th></tr>
      {% for o in orders %}
      <tr style="background:{{ 'rgba(255,214,0,0.06)' if o.is_crypto else ('rgba(100,181,246,0.10)' if o.symbol == 'SPY' else ('rgba(240,80,176,0.10)' if o.symbol == 'QQQ' else 'rgba(68,138,255,0.07)')) }}">
        <td>{{ o.time }}</td>
        <td><span class="tip"><strong>{{ o.symbol }}</strong><span class="tiptext">{{ o.name }}</span></span></td>
        <td style="color:{{ '#ffd600' if o.is_crypto else ('#64b5f6' if o.symbol == 'SPY' else ('#f050b0' if o.symbol == 'QQQ' else '#448aff')) }};font-size:0.8em">{{ '🪙 Crypto' if o.is_crypto else ('🏦 SPY Default' if o.symbol == 'SPY' else ('🏦 QQQ Default' if o.symbol == 'QQQ' else '📈 Stock')) }}</td>
        <td>{% if o.side == 'buy' %}<span class="badge buy">BUY</span>
            {% else %}<span class="badge sell">SELL</span>{% endif %}</td>
        <td>{{ o.qty }}</td><td>${{ o.price }}</td>
      </tr>
      {% endfor %}
    </table></div>
  </div>

  <script>
    {% if chart_labels %}
    const timestamps   = {{ chart_timestamps | safe }};
    const valTotal     = {{ chart_values | safe }};
    const valStocks    = {{ chart_stock_values | safe }};
    const valCrypto    = {{ chart_crypto_values | safe }};
    const valCash      = {{ chart_cash_values | safe }};

    // Pair each value series with its real timestamp so gaps in data
    // collection (e.g. downtime) show as proportional gaps on the axis,
    // instead of being silently compressed like a category scale would.
    function toXY(ts, vals) { return vals.map((v, i) => ({ x: ts[i], y: v })); }
    function fmtAxisDate(ms) {
      const d = new Date(ms);
      return d.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit' }) + ' ' +
             d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });
    }

    // Shade non-trading periods (outside Mon-Fri 14:30-21:00 UK time)
    // Trading-hours shading is meaningless at 6h/12h resolution, so 3M/6M switch it off
    let shadeTradingHours = true;
    const tradingHoursPlugin = {
      id: 'tradingHoursShading',
      beforeDraw(chart) {
        if (!shadeTradingHours) return;
        const { ctx, chartArea, scales } = chart;
        const xScale = scales.x;
        const pts = chart.data.datasets[0].data;
        if (!pts || !pts.length) return;
        function isTrading(ms) {
          const date = new Date(ms);
          const dow = date.getDay();
          if (dow === 0 || dow === 6) return false;
          const mins = date.getHours() * 60 + date.getMinutes();
          return mins >= 14 * 60 + 30 && mins < 21 * 60;
        }
        ctx.save();
        let tradingStart = null;
        pts.forEach((pt) => {
          const trading = isTrading(pt.x);
          const x = xScale.getPixelForValue(pt.x);
          if (trading && tradingStart === null) {
            tradingStart = x;
          } else if (!trading && tradingStart !== null) {
            ctx.fillStyle = 'rgba(68, 138, 255, 0.12)';
            ctx.fillRect(tradingStart, chartArea.top, x - tradingStart, chartArea.bottom - chartArea.top);
            tradingStart = null;
          }
        });
        if (tradingStart !== null) {
          ctx.fillStyle = 'rgba(68, 138, 255, 0.12)';
          ctx.fillRect(tradingStart, chartArea.top, chartArea.right - tradingStart, chartArea.bottom - chartArea.top);
        }
        ctx.restore();
      }
    };

    const chartOptions = {
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { title: items => fmtAxisDate(items[0].parsed.x) } }
      },
      scales: {
        x: { type: 'linear', min: timestamps[0], max: timestamps[timestamps.length - 1],
             ticks: { color: '#9aa3b5', maxTicksLimit: 10, callback: fmtAxisDate }, grid: { color: '#252840' } },
        y: { ticks: { color: '#9aa3b5', callback: v => '$' + v.toLocaleString() }, grid: { color: '#252840' } }
      }
    };

    const isMobile = window.innerWidth <= 600;
    const ptRadius = isMobile ? 0 : 2;
    const chart = new Chart(
      document.getElementById('portfolioChart').getContext('2d'), {
      type: 'line',
      plugins: [tradingHoursPlugin],
      data: {
        datasets: [
          { label: 'Total ($)',  data: toXY(timestamps, valTotal),  borderColor: '#00c853',
            backgroundColor: 'rgba(0,200,83,0.08)', borderWidth: 2,
            pointRadius: ptRadius, fill: true, tension: 0.3, hidden: false },
          { label: 'Stocks ($)', data: toXY(timestamps, valStocks), borderColor: '#448aff',
            backgroundColor: 'rgba(68,138,255,0.05)', borderWidth: 2,
            pointRadius: ptRadius, fill: false, tension: 0.3, hidden: true },
          { label: 'Crypto ($)', data: toXY(timestamps, valCrypto), borderColor: '#ffd600',
            backgroundColor: 'rgba(255,214,0,0.05)', borderWidth: 2,
            pointRadius: ptRadius, fill: false, tension: 0.3, hidden: true },
          { label: 'Cash ($)', data: toXY(timestamps, valCash), borderColor: '#b0b8d8',
            backgroundColor: 'rgba(176,184,216,0.05)', borderWidth: 2,
            pointRadius: ptRadius, fill: false, tension: 0.3, hidden: true }
        ]
      },
      options: chartOptions
    });

    function toggleLine(c, index, el) {
      const meta = c.getDatasetMeta(index);
      meta.hidden = !meta.hidden;
      el.classList.toggle('active');
      c.update();
    }

    const portfolioViews = {{ chart_views | safe }};
    function setPortfolioView(view) {
      const vd = portfolioViews[view];
      shadeTradingHours = !['3m', '6m'].includes(view);
      chart.data.datasets[0].data = toXY(vd.timestamps, vd.values);
      chart.data.datasets[1].data = toXY(vd.timestamps, vd.stock_values);
      chart.data.datasets[2].data = toXY(vd.timestamps, vd.crypto_values);
      chart.data.datasets[3].data = toXY(vd.timestamps, vd.cash_values);
      chart.options.scales.x.min = vd.timestamps[0];
      chart.options.scales.x.max = vd.timestamps[vd.timestamps.length - 1];
      chart.update('none');
      ['24h','7d','30d','3m','6m'].forEach(v => {
        const btn = document.getElementById('pvbtn-' + v);
        btn.style.background = v === view ? '#2e3250' : 'transparent';
        btn.style.color      = v === view ? '#fff' : '#9aa3b5';
      });
    }
    {% endif %}

    // Donut allocation charts
    const allocData = {
      labels: ['Stocks', 'SPY Default', 'Crypto', 'Cash'],
      datasets: [{ data: [{{ stock_pct }}, {{ spy_pct }}, {{ crypto_pct }}, {{ cash_pct }}],
        backgroundColor: ['#448aff', '#c084fc', '#ffd600', '#00c853'],
        borderWidth: 0, hoverOffset: 6 }]
    };
    const allocOpts = {
      cutout: '68%',
      plugins: { legend: { display: false }, tooltip: {
        callbacks: { label: ctx => ctx.label + ': ' + ctx.raw + '%' }
      }}
    };
    new Chart(document.getElementById('allocChart').getContext('2d'),
      { type: 'doughnut', data: allocData, options: allocOpts });

    // ── Trade countdowns ──
    const STOCK_HOURS = [14,15,16,17,18,19,20,21];
    function getUKParts(d) {
      const fmt = new Intl.DateTimeFormat('en-GB', {
        timeZone: 'Europe/London', hour: 'numeric', minute: 'numeric',
        second: 'numeric', hour12: false, weekday: 'short'
      });
      const parts = {};
      fmt.formatToParts(d).forEach(p => { parts[p.type] = p.value; });
      const h = parseInt(parts.hour) % 24;
      return { dow: ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'].indexOf(parts.weekday),
               h, m: parseInt(parts.minute), s: parseInt(parts.second) };
    }
    function fmtCountdown(s) {
      if (s <= 0) s = 0;
      const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sc = s%60;
      return h > 0 ? h+':'+String(m).padStart(2,'0')+':'+String(sc).padStart(2,'0')
                   : String(m).padStart(2,'0')+':'+String(sc).padStart(2,'0');
    }
    function secsToNextStock(now) {
      const uk = getUKParts(now);
      for (let d = 0; d < 8; d++) {
        if ((uk.dow + d) % 7 === 0 || (uk.dow + d) % 7 === 6) continue;
        for (const slotH of STOCK_HOURS) {
          const secs = (slotH - uk.h)*3600 - uk.m*60 - uk.s + d*86400;
          if (secs > 0) return secs;
        }
      }
      return -1;
    }
    function secsToNextCrypto(now) {
      const uk = getUKParts(now);
      const nextH = uk.h % 2 === 0 ? uk.h + 2 : uk.h + 1;
      const secs  = (nextH - uk.h)*3600 - uk.m*60 - uk.s;
      return secs > 0 ? secs : secs + 7200;
    }
    function updateNavCountdowns() {
      const now = new Date();
      const sStock  = secsToNextStock(now);
      const sCrypto = secsToNextCrypto(now);
      const sEl = document.getElementById('nav-stock-countdown');
      const cEl = document.getElementById('nav-crypto-countdown');
      if (sEl) {
        sEl.textContent = sStock < 0 ? 'CLOSED' : fmtCountdown(sStock);
        sEl.style.color = sStock < 0 ? '#ff6b57' : sStock < 300 ? '#f0a500' : '#00c853';
      }
      if (cEl) {
        cEl.textContent = fmtCountdown(sCrypto);
        cEl.style.color = sCrypto < 300 ? '#f0a500' : '#448aff';
      }
    }
    updateNavCountdowns();
    setInterval(updateNavCountdowns, 1000);
  </script>
</body>
</html>
"""

@app.route('/login', methods=['GET', 'POST'])
def login():
    from datetime import timedelta
    next_url = request.args.get('next', '/')
    if request.method == 'POST':
        if DASHBOARD_PASSWORD and request.form.get('password') == DASHBOARD_PASSWORD:
            session.permanent = True
            app.permanent_session_lifetime = timedelta(days=SESSION_DAYS)
            session['authed'] = True
            return redirect(next_url)
        return render_template_string(LOGIN_HTML, error='Incorrect password')
    if session.get('authed'):
        return redirect(next_url)
    return render_template_string(LOGIN_HTML, error=None)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@requires_auth
def dashboard():
    try:
        account = api.get_account()
    except Exception as e:
        return f"<h1 style='color:white;background:#0f1117;padding:40px;font-family:sans-serif'>⚠️ Connection error: {e}<br><br><a href='/' style='color:#448aff'>Retry</a></h1>", 503

    cash      = f"{float(account.cash):,.2f}"
    portfolio = f"{float(account.portfolio_value):,.2f}"
    pnl       = round(float(account.equity) - float(account.last_equity), 2)
    positions = api.list_positions()
    pos_dict  = {p.symbol: p for p in positions}

    stock_model_info  = {'accuracy': '—', 'trained_at': '—'}
    crypto_model_info = {'accuracy': '—', 'trained_at': '—', 'interval': '1h'}
    if os.path.exists('/home/jv/model_info.json'):
        with open('/home/jv/model_info.json') as f:
            stock_model_info = json.load(f)
    if os.path.exists('/home/jv/crypto_model_info.json'):
        with open('/home/jv/crypto_model_info.json') as f:
            crypto_model_info = json.load(f)

    mkt, btc_trend = get_market_indicators()
    market_data = None
    if mkt is not None:
        market_data = {
            'vix':       round(float(mkt['VIX']), 1),
            'spy_trend': int(mkt['SPY_Trend']),
            'qqq_trend': int(mkt['QQQ_Trend']),
            'btc_trend': btc_trend,
            'spy_24h':   round(float(mkt['SPY_Return']) * 100, 2),
            'qqq_24h':   round(float(mkt['QQQ_Return']) * 100, 2),
            'spy_5d':    round(float(mkt['SPY_Return_5d']) * 100, 2),
            'qqq_5d':    round(float(mkt['QQQ_Return_5d']) * 100, 2),
        }
        with open('/home/jv/market_cache.json', 'w') as f:
            json.dump(market_data, f)

    all_stock_returns = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period="10d", interval="1d", progress=False)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            ret = df['Close'].pct_change()
            all_stock_returns[ticker] = {
                'latest': float(ret.iloc[-1]),
                '5d':     float(ret.iloc[-5:].mean())
            }
        except:
            all_stock_returns[ticker] = {'latest': 0.0, '5d': 0.0}

    stocks = []
    for ticker in TICKERS:
        try:
            df            = get_stock_features(ticker)
            latest        = df.iloc[-1].copy()
            current_price = round(float(df['Close'].iloc[-1]), 2)
            mom, mom_5d   = get_sector_momentum(ticker, all_stock_returns)
            latest['Sector_Momentum']    = mom
            latest['Sector_Momentum_5d'] = mom_5d
            if mkt is not None:
                latest['VIX']           = float(mkt['VIX'])
                latest['SPY_Return']    = float(mkt['SPY_Return'])
                latest['SPY_Trend']     = float(mkt['SPY_Trend'])
                latest['SPY_Return_5d'] = float(mkt['SPY_Return_5d'])
                latest['QQQ_Return']    = float(mkt['QQQ_Return'])
                latest['QQQ_Trend']     = float(mkt['QQQ_Trend'])
                latest['QQQ_Return_5d'] = float(mkt['QQQ_Return_5d'])
            else:
                for col in ['VIX','SPY_Return','SPY_Trend','SPY_Return_5d',
                            'QQQ_Return','QQQ_Trend','QQQ_Return_5d']:
                    latest[col] = 0.0
            feature_row = pd.DataFrame([latest[stock_features]])
            pred        = stock_model.predict(feature_row)[0]
            conf        = stock_model.predict_proba(feature_row)[0][pred]
            sent_score  = get_stock_sentiment_cached(ticker) or 0.0
            signal      = 'BUY' if pred == 1 else 'SELL'
            if conf < 0.60:
                signal = 'HOLD'
            elif signal == 'BUY' and sent_score < -0.05:
                signal = 'HOLD'
            elif signal == 'SELL' and sent_score > 0.05:
                signal = 'HOLD'
            sent_label = '🟢' if sent_score > 0.05 else '🔴' if sent_score < -0.05 else '⚪'
            p          = pos_dict.get(ticker)
            shares     = fmt_qty(p.qty) if p else 0
            stocks.append({
                'ticker':     ticker,
                'name':       COMPANY_NAMES.get(ticker, ticker),
                'price':      current_price,
                'signal':     signal,
                'confidence': f"{conf:.1%}",
                'sentiment':  f"{sent_label} {sent_score:.2f}",
                'shares':     shares
            })
        except Exception as e:
            stocks.append({
                'ticker': ticker, 'name': COMPANY_NAMES.get(ticker, ticker),
                'price': '—', 'signal': 'ERROR',
                'confidence': '—', 'sentiment': '—', 'shares': '—'
            })

    crypto_mkt = get_crypto_market_indicators()

    all_crypto_returns = {}
    for symbol, yf_symbol in CRYPTOS.items():
        try:
            df = yf.download(yf_symbol, period="5d", interval="1h", progress=False)
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            ret = df['Close'].pct_change()
            all_crypto_returns[symbol] = {
                'latest': float(ret.iloc[-1]),
                '24h':    float(ret.iloc[-24:].mean())
            }
        except:
            all_crypto_returns[symbol] = {'latest': 0.0, '24h': 0.0}

    crypto_mom, crypto_mom_24h = get_crypto_momentum(all_crypto_returns)

    crypto = []
    for symbol, yf_symbol in CRYPTOS.items():
        try:
            df            = get_crypto_features(yf_symbol)
            latest        = df.iloc[-1].copy()
            current_price = round(float(df['Close'].iloc[-1]), 4)
            latest['Crypto_Momentum']     = crypto_mom
            latest['Crypto_Momentum_24h'] = crypto_mom_24h
            if crypto_mkt:
                latest['VIX']             = crypto_mkt['VIX']
                latest['SPY_Return']      = crypto_mkt['SPY_Return']
                latest['SPY_Trend']       = crypto_mkt['SPY_Trend']
                latest['SPY_Return_24h']  = crypto_mkt['SPY_Return_24h']
                if symbol == 'BTC/USD':
                    latest['BTC_Return']     = 0.0
                    latest['BTC_Trend']      = 0.0
                    latest['BTC_Return_24h'] = 0.0
                else:
                    latest['BTC_Return']     = crypto_mkt['BTC_Return']
                    latest['BTC_Trend']      = crypto_mkt['BTC_Trend']
                    latest['BTC_Return_24h'] = crypto_mkt['BTC_Return_24h']
            else:
                for col in ['VIX','SPY_Return','SPY_Trend','SPY_Return_24h',
                            'BTC_Return','BTC_Trend','BTC_Return_24h']:
                    latest[col] = 0.0
            feature_row   = pd.DataFrame([latest[crypto_features]])
            pred          = crypto_model.predict(feature_row)[0]
            conf          = crypto_model.predict_proba(feature_row)[0][pred]
            signal        = 'BUY' if pred == 1 else 'SELL'
            if conf < 0.65:
                signal = 'HOLD'
            sent_score  = get_crypto_sentiment_cached(symbol)
            sent_label  = '🟢' if sent_score > 0.05 else '🔴' if sent_score < -0.05 else '⚪'
            if signal == 'BUY' and sent_score < -0.05:
                signal = 'HOLD'
            elif signal == 'SELL' and sent_score > 0.05:
                signal = 'HOLD'
            alpaca_symbol = symbol.replace('/', '')
            p             = pos_dict.get(alpaca_symbol)
            coins         = fmt_qty(p.qty) if p else 0
            crypto.append({
                'ticker':     symbol,
                'name':       COMPANY_NAMES.get(symbol, symbol),
                'price':      f"{current_price:,}",
                'signal':     signal,
                'confidence': f"{conf:.1%}",
                'sentiment':  f"{sent_label} {sent_score:.2f}",
                'coins':      coins
            })
        except Exception as e:
            crypto.append({
                'ticker': symbol, 'name': COMPANY_NAMES.get(symbol, symbol),
                'price': '—', 'signal': 'ERROR',
                'confidence': '—', 'sentiment': '—', 'coins': '—'
            })

    raw_orders  = api.list_orders(status='all', limit=40)
    crypto_syms = list(CRYPTOS.keys())
    all_orders  = sorted(
        [o for o in raw_orders if o.symbol in TICKERS or o.symbol in crypto_syms or o.symbol in ('SPY', 'QQQ')],
        key=lambda o: o.submitted_at or datetime.min, reverse=True
    )[:20]
    orders = [{
        'time':   o.submitted_at.strftime('%d/%m %H:%M') if o.submitted_at else '—',
        'symbol': o.symbol,
        'name':   COMPANY_NAMES.get(o.symbol, o.symbol),
        'side':   o.side, 'qty': fmt_qty(o.qty),
        'price':  o.filled_avg_price or '—',
        'is_crypto': o.symbol in crypto_syms,
    } for o in all_orders]
    crypto_orders = []

    # Portfolio history for chart
    chart_views = build_chart_views()
    view_7d     = chart_views['7d']
    chart_labels        = view_7d['labels']
    chart_timestamps    = view_7d['timestamps']
    chart_values        = view_7d['values']
    chart_stock_values  = view_7d['stock_values']
    chart_crypto_values = view_7d['crypto_values']
    chart_cash_values   = view_7d['cash_values']

    # Build positions lists
    stock_positions_list  = []
    crypto_positions_list = []
    _stock_hw  = json.load(open('/home/jv/stock_highwater.json'))  if os.path.exists('/home/jv/stock_highwater.json')  else {}
    _crypto_hw = json.load(open('/home/jv/crypto_highwater.json')) if os.path.exists('/home/jv/crypto_highwater.json') else {}
    for p in positions:
        entry    = float(p.avg_entry_price)
        current  = float(p.current_price)
        is_stock = p.symbol in TICKERS or p.symbol == 'SPY'
        hw_map   = _stock_hw if is_stock else _crypto_hw
        stop_pct = 0.05 if p.symbol == 'SPY' else (0.12 if is_stock else 0.15)
        peak     = hw_map.get(p.symbol, entry)
        trail_stop = peak * (1 - stop_pct)
        gap_pct    = (current - trail_stop) / current * 100
        row = {
            'symbol':     p.symbol,
            'name':       'S&P 500 ETF (default)' if p.symbol == 'SPY' else COMPANY_NAMES.get(p.symbol, p.symbol),
            'qty':        fmt_qty(p.qty),
            'avg_price':  f"{entry:,.4f}",
            'current':    f"{current:,.4f}",
            'value':      f"{float(p.market_value):,.2f}",
            'pl':         float(p.unrealized_pl),
            'pl_abs':     f"{abs(float(p.unrealized_pl)):,.2f}",
            'pl_pct':     f"{float(p.unrealized_plpc)*100:+.2f}",
            'trail_stop': f"{trail_stop:,.4f}",
            'gap_pct':    f"{gap_pct:.1f}",
            'is_stock':   is_stock,
            'is_spy':     p.symbol == 'SPY',
        }
        if is_stock:
            stock_positions_list.append(row)
        else:
            crypto_positions_list.append(row)

    crypto_positions_count = len(crypto_positions_list)

    port_val   = float(account.portfolio_value)
    spy_val    = sum(float(p.market_value) for p in positions if p.symbol == 'SPY')
    stock_val  = sum(float(p.market_value) for p in positions if p.symbol in TICKERS)
    crypto_val = sum(float(p.market_value) for p in positions if p.symbol not in TICKERS and p.symbol != 'SPY')
    cash_val   = port_val - spy_val - stock_val - crypto_val
    spy_pct    = round(spy_val    / port_val * 100, 1) if port_val else 0.0
    stock_pct  = round(stock_val  / port_val * 100, 1) if port_val else 0.0
    crypto_pct = round(crypto_val / port_val * 100, 1) if port_val else 0.0
    cash_pct   = round(cash_val   / port_val * 100, 1) if port_val else 0.0

    return render_template_string(HTML,
        cash=cash, portfolio=portfolio, pnl=pnl,
        position_count=len(positions),
        stocks=stocks, crypto=crypto,
        orders=orders, crypto_orders=crypto_orders,
        updated=datetime.now().strftime('%H:%M:%S'),
        chart_labels=json.dumps(chart_labels),
        chart_timestamps=json.dumps(chart_timestamps),
        chart_values=json.dumps(chart_values),
        chart_stock_values=json.dumps(chart_stock_values),
        chart_crypto_values=json.dumps(chart_crypto_values),
        chart_cash_values=json.dumps(chart_cash_values),
        chart_views=json.dumps(chart_views),
        stock_model_info=stock_model_info,
        crypto_model_info=crypto_model_info,
        market=market_data,
        crypto_positions=crypto_positions_count,
        stock_positions=stock_positions_list,
        crypto_positions_list=crypto_positions_list,
        spy_pct=spy_pct,
        stock_pct=stock_pct,
        crypto_pct=crypto_pct,
        cash_pct=cash_pct)


MILESTONES_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trading Bot — Milestones</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0a0c14; color: #e0e6ff;
      font-family: -apple-system, 'Segoe UI', sans-serif;
      padding: 24px 20px; min-height: 100vh;
    }
    a { color: #448aff; text-decoration: none; }
    a:hover { text-decoration: underline; }

    .page-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 28px; flex-wrap: wrap; gap: 12px;
    }
    .page-title { font-size: 1.4em; font-weight: 700; color: #e0e6ff; }
    .page-title span { color: #448aff; }
    .back-btn {
      padding: 7px 16px; border-radius: 7px; background: #1a1d2e;
      border: 1px solid #2a2d4a; color: #aaa; font-size: 0.85em;
      font-weight: 600; cursor: pointer;
    }
    .back-btn:hover { color: #fff; border-color: #448aff; }

    .summary-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px; margin-bottom: 28px;
    }
    .summary-card {
      background: #111420; border: 1px solid #1e2540;
      border-radius: 10px; padding: 16px 18px;
    }
    .summary-card .label { font-size: 0.72em; color: #666; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
    .summary-card .value { font-size: 1.35em; font-weight: 700; }
    .green { color: #00c853; } .red { color: #ff4d6a; }
    .blue  { color: #448aff; } .gold { color: #ffd600; }
    .purple { color: #c084fc; }

    .chart-card {
      background: #111420; border: 1px solid #1e2540;
      border-radius: 10px; padding: 20px; margin-bottom: 28px;
    }
    .chart-card h2 { font-size: 0.85em; color: #666; font-weight: 700;
      text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 16px; }

    .milestones-list { display: flex; flex-direction: column; gap: 16px; }
    .milestone-card {
      background: #111420; border: 1px solid #1e2540;
      border-radius: 10px; padding: 18px 20px;
    }
    .milestone-header {
      display: flex; align-items: baseline; justify-content: space-between;
      flex-wrap: wrap; gap: 8px; margin-bottom: 14px;
      padding-bottom: 12px; border-bottom: 1px solid #1e2540;
    }
    .milestone-date { font-size: 1em; font-weight: 700; color: #448aff; }
    .milestone-val  { font-size: 1.25em; font-weight: 700; }
    .milestone-ret  { font-size: 0.85em; font-weight: 700; margin-left: 10px; }
    .milestone-notes { font-size: 0.78em; color: #8c93a6; margin-top: 4px; }

    .positions-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 8px; margin-bottom: 10px;
    }
    .pos-chip {
      background: #0d1022; border: 1px solid #1e2540; border-radius: 6px;
      padding: 8px 10px;
    }
    .pos-sym  { font-size: 0.8em; font-weight: 700; color: #e0e6ff; }
    .pos-sym.spy { color: #c084fc; }
    .pos-sym.qqq { color: #f050b0; }
    .pos-val  { font-size: 0.78em; color: #666; margin-top: 2px; }
    .pos-pl   { font-size: 0.78em; font-weight: 600; margin-top: 1px; }

    .model-row {
      display: flex; gap: 16px; flex-wrap: wrap;
      font-size: 0.78em; color: #8c93a6; margin-top: 10px;
      padding-top: 10px; border-top: 1px solid #1e2540;
    }
    .model-row span { color: #9aa3b5; }
    .model-row strong { color: #aaa; }

    .empty { text-align: center; color: #8c93a6; padding: 60px 20px; font-size: 0.9em; }
  </style>
</head>
<body>

<div class="page-header">
  <div class="page-title">📍 <span>Milestone</span> History</div>
  <a href="/"><button class="back-btn">← Dashboard</button></a>
</div>

{% if milestones %}

<div class="summary-grid">
  <div class="summary-card">
    <div class="label">Started</div>
    <div class="value">$100,000.00</div>
  </div>
  <div class="summary-card">
    <div class="label">Latest Portfolio</div>
    <div class="value blue">${{ latest.portfolio_value | format_money }}</div>
  </div>
  <div class="summary-card">
    <div class="label">Total Return</div>
    <div class="value {{ 'green' if latest.total_return_pct >= 0 else 'red' }}">
      {{ '+' if latest.total_return_pct >= 0 else '' }}{{ latest.total_return_pct }}%
    </div>
  </div>
  <div class="summary-card">
    <div class="label">Milestones Recorded</div>
    <div class="value">{{ milestones | length }}</div>
  </div>
  <div class="summary-card">
    <div class="label">First Recorded</div>
    <div class="value" style="font-size:1em">{{ milestones[0].date }}</div>
  </div>
  <div class="summary-card">
    <div class="label">Latest Model</div>
    <div class="value blue" style="font-size:1em">
      {{ latest.model.stock_accuracy }} stocks
    </div>
  </div>
  <div class="summary-card">
    <div class="label">Sharpe Ratio</div>
    {% if sharpe is not none %}
    <div class="value {{ 'green' if sharpe >= 1.0 else ('gold' if sharpe >= 0.5 else 'red') }}">
      {{ sharpe }}
      <span style="font-size:0.55em;color:#8c93a6;font-weight:400">
        {{ '★ good' if sharpe >= 1.0 else ('ok' if sharpe >= 0.5 else 'poor') }}
      </span>
    </div>
    {% else %}
    <div class="value" style="color:#8c93a6;font-size:0.9em">need ≥3 weeks</div>
    {% endif %}
  </div>
</div>

<div class="chart-card">
  <h2>Portfolio Value Over Time</h2>
  <canvas id="milestoneChart" height="80"></canvas>
</div>

<div class="milestones-list">
  {% for m in milestones | reverse %}
  <div class="milestone-card">
    <div class="milestone-header">
      <div>
        <span class="milestone-date">{{ m.date }}{% if m.time %} · {{ m.time }}{% endif %}</span>
        {% if m.notes is defined and m.notes %}
        <div class="milestone-notes">{{ m.notes | join(' &bull; ') }}</div>
        {% endif %}
      </div>
      <div>
        <span class="milestone-val">${{ m.portfolio_value | format_money }}</span>
        <span class="milestone-ret {{ 'green' if m.total_return_pct >= 0 else 'red' }}">
          {{ '+' if m.total_return_pct >= 0 else '' }}{{ m.total_return_pct }}%
        </span>
      </div>
    </div>

    <div class="positions-grid">
      {% for p in m.positions %}
      <div class="pos-chip">
        <div class="pos-sym {{ 'spy' if p.symbol == 'SPY' else ('qqq' if p.symbol == 'QQQ' else '') }}">{{ p.symbol }}</div>
        <div class="pos-val">${{ p.value | format_money }}</div>
        <div class="pos-pl {{ 'green' if p.pl_pct >= 0 else 'red' }}">{{ '+' if p.pl_pct >= 0 else '' }}{{ p.pl_pct }}%</div>
      </div>
      {% endfor %}
    </div>

    {% if m.model is defined %}
    <div class="model-row">
      <span>Model accuracy —</span>
      <span><strong>Stocks:</strong> {{ m.model.stock_accuracy }}</span>
      <span><strong>Crypto:</strong> {{ m.model.crypto_accuracy }}</span>
      <span><strong>Trained:</strong> {{ m.model.trained_at }}</span>
      {% if m.sharpe is defined and m.sharpe is not none %}
      <span><strong>Sharpe:</strong>
        <span class="{{ 'green' if m.sharpe >= 1.0 else ('gold' if m.sharpe >= 0.5 else 'red') }}">{{ m.sharpe }}</span>
      </span>
      {% endif %}
    </div>
    {% endif %}
  </div>
  {% endfor %}
</div>

<script>
const labels  = {{ chart_labels | safe }};
const values  = {{ chart_values | safe }};
const ctx = document.getElementById('milestoneChart').getContext('2d');
new Chart(ctx, {
  type: 'line',
  data: {
    labels: labels,
    datasets: [{
      label: 'Portfolio Value',
      data: values,
      borderColor: '#448aff',
      backgroundColor: 'rgba(68,138,255,0.15)',
      borderWidth: 2, pointRadius: 4, pointBackgroundColor: '#448aff',
      fill: true, tension: 0.3
    }, {
      label: 'Starting Value',
      data: labels.map(() => 100000),
      borderColor: '#333a5c', borderWidth: 1,
      borderDash: [6,4], pointRadius: 0, fill: false
    }]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: ctx => '$' + ctx.raw.toLocaleString() } }
    },
    scales: {
      x: { ticks: { color: '#8c93a6', maxRotation: 45 }, grid: { color: '#151825' } },
      y: { ticks: { color: '#8c93a6', callback: v => '$' + v.toLocaleString() }, grid: { color: '#151825' } }
    }
  }
});
</script>

{% else %}
<div class="empty">No milestones recorded yet. The first will appear this Sunday at 4am.</div>
{% endif %}

</body>
</html>
"""


@app.template_filter('format_money')
def format_money(val):
    try:
        return f"{float(val):,.2f}"
    except Exception:
        return str(val)


@app.route('/milestones')
@requires_auth
def milestones_page():
    import json as _json

    raw = []
    if os.path.exists('/home/jv/milestones.json'):
        with open('/home/jv/milestones.json') as f:
            raw = _json.load(f)

    if not raw:
        return render_template_string(MILESTONES_HTML, milestones=[], latest={},
                                      sharpe=None, chart_labels='[]', chart_values='[]')

    import math as _math
    def _calc_sharpe(ms):
        values = [m['portfolio_value'] for m in ms if 'portfolio_value' in m]
        if len(values) < 3:
            return None
        weekly_returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        n = len(weekly_returns)
        mean_r = sum(weekly_returns) / n
        variance = sum((r - mean_r) ** 2 for r in weekly_returns) / (n - 1)
        std_r = _math.sqrt(variance)
        if std_r == 0:
            return None
        return round((mean_r - 0.05 / 52) / std_r * _math.sqrt(52), 2)

    sharpe = _calc_sharpe(raw)
    chart_labels = [m['date'] for m in raw]
    chart_values = [m['portfolio_value'] for m in raw]

    return render_template_string(MILESTONES_HTML,
        milestones=raw,
        latest=raw[-1],
        sharpe=sharpe,
        chart_labels=_json.dumps(chart_labels),
        chart_values=_json.dumps(chart_values),
    )


REVIEW_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trading Bot — Weekly Review</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0a0c14; color: #e0e6ff; font-family: -apple-system, 'Segoe UI', sans-serif;
           padding: 24px 20px; min-height: 100vh; max-width: 960px; margin: 0 auto; line-height: 1.5; }
    a { color: #448aff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .page-header { display: flex; align-items: center; justify-content: space-between;
                   margin-bottom: 24px; flex-wrap: wrap; gap: 12px; }
    .page-title { font-size: 1.4em; font-weight: 700; }
    .page-title span { color: #448aff; }
    .back-btn { padding: 7px 16px; border-radius: 7px; background: #1a1d2e; border: 1px solid #2a2d4a;
                color: #aaa; font-size: 0.85em; font-weight: 600; }
    .back-btn:hover { color: #fff; border-color: #448aff; text-decoration: none; }
    select { background: #1a1d2e; color: #e0e6ff; border: 1px solid #2a2d4a; border-radius: 7px;
             padding: 6px 10px; font-size: 0.85em; }
    .card { background: #111420; border: 1px solid #1e2540; border-radius: 10px; padding: 18px 20px; margin-bottom: 20px; }
    .card h2 { font-size: 0.78em; color: #666; font-weight: 700; text-transform: uppercase;
               letter-spacing: 0.1em; margin-bottom: 12px; }
    .status { display: inline-block; padding: 3px 12px; border-radius: 20px; font-weight: 700;
              font-size: 0.8em; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }
    .s-green { background: #00c85322; color: #00c853; } .s-amber { background: #ffab0022; color: #ffab00; }
    .s-red { background: #ff4d6a22; color: #ff4d6a; }
    .headline { font-size: 1.15em; font-weight: 600; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 20px; }
    .stat { background: #111420; border: 1px solid #1e2540; border-radius: 10px; padding: 14px 16px; }
    .stat .label { font-size: 0.7em; color: #666; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; }
    .stat .value { font-size: 1.25em; font-weight: 700; margin-top: 4px; }
    .stat .sub { font-size: 0.78em; color: #9aa3b5; }
    .green { color: #00c853; } .red { color: #ff4d6a; }
    .finding { border-left: 3px solid #2a2d4a; padding: 8px 0 8px 14px; margin-bottom: 12px; }
    .finding .title { font-weight: 600; }
    .finding .meta { font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.08em; color: #666; }
    .finding .detail { color: #c0c8e0; font-size: 0.92em; margin-top: 2px; }
    .f-critical { border-color: #ff4d6a; } .f-warning { border-color: #ffab00; }
    .f-info { border-color: #448aff; } .f-good { border-color: #00c853; }
    table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
    td, th { text-align: left; padding: 7px 8px; border-bottom: 1px solid #1e2540; vertical-align: top; }
    th { color: #666; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.08em; }
    .pri { white-space: nowrap; font-weight: 700; font-size: 0.8em; }
    .error { background: #ff4d6a18; border: 1px solid #ff4d6a55; color: #ffb3c0; }
    .muted { color: #666; font-size: 0.8em; }
    @media (max-width: 600px) { body { padding: 16px; } }
  </style>
</head>
<body>
  <div class="page-header">
    <div class="page-title">🧐 Weekly <span>Review</span></div>
    <div style="display:flex;gap:10px;align-items:center">
      {% if dates %}
      <select onchange="location='/review?date='+this.value">
        {% for d in dates %}<option value="{{ d }}" {% if entry and d == entry.date %}selected{% endif %}>{{ d }}</option>{% endfor %}
      </select>
      {% endif %}
      <a class="back-btn" href="/">← Dashboard</a>
    </div>
  </div>

  {% if not entry %}
    <div class="card">No reviews yet. The review agent runs every Sunday at 04:30.</div>
  {% else %}
    {% set r = entry.review %}
    {% set perf = entry.facts.performance %}

    {% if entry.error %}
      <div class="card error"><b>Review could not be generated:</b> {{ entry.error }}<br>
      <span class="muted">The facts below were still collected.</span></div>
    {% endif %}

    {% if r %}
    <div class="card">
      <div class="status s-{{ r.status }}">{{ r.status }}</div>
      <div class="headline">{{ r.headline }}</div>
    </div>
    {% endif %}

    {% if perf and not perf.error %}
    <div class="grid">
      {% for label, key in [('Since start', 'since_start'), ('Since SPY strategy', 'since_spy_strategy'), ('Last 4 weeks', 'last_4_weeks'), ('Last week', 'last_week')] %}
        {% set p = perf[key] %}{% if p %}
        <div class="stat">
          <div class="label">{{ label }}</div>
          <div class="value {{ 'green' if p.bot_pct >= p.spy_pct else 'red' }}">{{ '%+.2f' % p.bot_pct }}%</div>
          <div class="sub">SPY {{ '%+.2f' % p.spy_pct }}% · from {{ p['from'] }}</div>
        </div>{% endif %}
      {% endfor %}
      <div class="stat">
        <div class="label">Portfolio</div>
        <div class="value">${{ '{:,.0f}'.format(perf.equity) }}</div>
        <div class="sub">{{ perf.positions|length }} positions</div>
      </div>
    </div>
    {% endif %}

    {% if r %}
    <div class="card">
      <h2>Bot vs just holding SPY</h2>
      <div>{{ r.benchmark_summary }}</div>
    </div>

    <div class="card">
      <h2>Recommended actions</h2>
      {% if r.recommended_actions %}
      <table>
        <tr><th>When</th><th>Action</th><th>Why</th></tr>
        {% for a in r.recommended_actions %}
        <tr><td class="pri">{{ a.priority }}</td><td>{{ a.action }}</td><td style="color:#c0c8e0">{{ a.why }}</td></tr>
        {% endfor %}
      </table>
      <div class="muted" style="margin-top:10px">Advisory only — the agent never changes the bot itself.</div>
      {% else %}<div>Nothing to do this week.</div>{% endif %}
    </div>

    <div class="card">
      <h2>Findings</h2>
      {% for f in r.findings %}
      <div class="finding f-{{ f.severity }}">
        <div class="meta">{{ f.severity }} · {{ f.area }}</div>
        <div class="title">{{ f.title }}</div>
        <div class="detail">{{ f.detail }}</div>
      </div>
      {% endfor %}
    </div>
    {% endif %}

    <div class="muted">
      Facts gathered {{ entry.facts.generated_at }}
      {% if entry.usage %} · reviewed by {{ entry.usage.model }} · ~${{ entry.usage.approx_cost_usd }}{% endif %}
    </div>
  {% endif %}
</body>
</html>
"""


@app.route('/review')
@requires_auth
def review_page():
    import json as _json
    reviews = []
    if os.path.exists('/home/jv/weekly_reviews.json'):
        with open('/home/jv/weekly_reviews.json') as f:
            reviews = _json.load(f)
    dates = [r['date'] for r in reversed(reviews)]
    wanted = request.args.get('date')
    entry = next((r for r in reviews if r['date'] == wanted), reviews[-1] if reviews else None)
    return render_template_string(REVIEW_HTML, entry=entry, dates=dates)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
