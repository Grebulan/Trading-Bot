import json
import os
import time
import joblib
import threading
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime
from flask import Flask, render_template_string
from sentiment import get_sentiment
from crypto_sentiment import get_crypto_sentiment

app = Flask(__name__)

STOCK_SENTIMENT_CACHE = '/home/jv/stock_sentiment_cache.json'
STOCK_SENTIMENT_TTL   = 3600  # 1 hour

def load_stock_sentiment_cache():
    if os.path.exists(STOCK_SENTIMENT_CACHE):
        with open(STOCK_SENTIMENT_CACHE) as f:
            return json.load(f)
    return {}

def _parse_history_time(t):
    """Parse a portfolio history timestamp — handles both ISO and legacy MM/DD HH:MM."""
    from datetime import datetime
    if not t:
        return None
    try:
        if 'T' in t or (len(t) > 5 and t[4] == '-'):
            return datetime.fromisoformat(t)
        # Legacy format: MM/DD HH:MM — assume current year
        year = datetime.now().year
        return datetime.strptime(f'{year}/{t}', '%Y/%m/%d %H:%M')
    except Exception:
        return None

def build_chart_views(history_file='/home/jv/portfolio_history.json'):
    """
    Return three resampled views of portfolio_history.json suitable for
    chart rendering.  Each view is a dict with parallel lists:
      labels, values, stock_values, crypto_values

    24h  — raw 30-min data for the last 24 real hours
    7d   — 1 point per 4 hours for the last 7 days
    all  — 1 point per 4 hours for all available data

    Resampling to equal density stops old sparse data and new dense data
    from producing wildly different visual weights on a category-scale chart.
    """
    from datetime import datetime, timedelta

    if not os.path.exists(history_file):
        empty = dict(labels=[], values=[], stock_values=[], crypto_values=[], cash_values=[])
        return {'24h': empty, '7d': empty, 'all': empty}

    with open(history_file) as f:
        history = json.load(f)

    if not history:
        empty = dict(labels=[], values=[], stock_values=[], crypto_values=[], cash_values=[])
        return {'24h': empty, '7d': empty, 'all': empty}

    def to_row(h):
        return {
            'time':   h['time'],
            'value':  h['value'],
            'stock':  h.get('stock_value', 0),
            'crypto': h.get('crypto_value', 0),
            'cash':   h.get('cash', 0),
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
            'values':       [r['value']  for r in data],
            'stock_values': [r['stock']  for r in data],
            'crypto_values':[r['crypto'] for r in data],
            'cash_values':  [max(0, round(r['value'] - r['stock'] - r['crypto'], 2)) for r in data],
        }

    # 24H — raw data, last 24 hours
    cutoff_24h = latest - timedelta(hours=24)
    raw_24h    = [r for r in rows if r['dt'] >= cutoff_24h]

    # 7D — resampled to 4h buckets, last 7 days
    cutoff_7d  = latest - timedelta(hours=168)
    raw_7d     = [r for r in rows if r['dt'] >= cutoff_7d]

    # ALL — resampled to 4h buckets, everything
    return {
        '24h': make_view(raw_24h if raw_24h else rows[-48:]),
        '7d':  make_view(resample(raw_7d, 4)),
        'all': make_view(resample(rows, 4)),
    }

def get_stock_sentiment_cached(ticker):
    """Return cached sentiment score, or 0.0 if cache missing/stale."""
    cache = load_stock_sentiment_cache()
    entry = cache.get(ticker)
    if entry and (time.time() - entry.get('fetched', 0)) < STOCK_SENTIMENT_TTL:
        return entry['score']
    return None   # None = no fresh data (show — rather than 0.0)

def refresh_stock_sentiment_cache():
    """Fetch fresh sentiment for all stock tickers and write cache.
    Called in a background thread so it never blocks a request."""
    cache = {}
    for ticker in TICKERS:
        try:
            score = get_sentiment(ticker)
            cache[ticker] = {'score': score, 'fetched': time.time()}
        except Exception:
            cache[ticker] = {'score': 0.0, 'fetched': time.time()}
    with open(STOCK_SENTIMENT_CACHE, 'w') as f:
        json.dump(cache, f)

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
    'AAPL','GOOGL','MSFT','META','AMZN','TSLA','NVDA',
    'JPM','BAC','GS','PFE','MRK','RIVN','NIO',
    'WMT','TGT','AMD','QCOM','MS'
]
COMPANY_NAMES = {
    'AAPL':     'Apple Inc.',
    'GOOGL':    'Alphabet (Google)',
    'MSFT':     'Microsoft',
    'META':     'Meta Platforms (Facebook)',
    'AMZN':     'Amazon',
    'TSLA':     'Tesla',
    'NVDA':     'NVIDIA',
    'JPM':      'JPMorgan Chase',
    'BAC':      'Bank of America',
    'GS':       'Goldman Sachs',
    'PFE':      'Pfizer',
    'MRK':      'Merck & Co.',
    'RIVN':     'Rivian Automotive',
    'NIO':      'NIO Inc.',
    'WMT':      'Walmart',
    'TGT':      'Target',
    'AMD':      'Advanced Micro Devices',
    'QCOM':     'Qualcomm',
    'MS':       'Morgan Stanley',
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
    'Tech':    ['AAPL','GOOGL','MSFT','META','AMZN','NVDA','AMD','QCOM'],
    'EV':      ['TSLA','RIVN','NIO'],
    'Finance': ['JPM','BAC','GS','MS'],
    'Pharma':  ['PFE','MRK'],
    'Retail':  ['WMT','TGT'],
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
              background: #1a1d2e; padding: 15px 20px; margin-bottom: 25px;
              border-bottom: 2px solid #252840; position: sticky; top: 0; z-index: 100; }
    .topnav h1 { font-size: 1.2em; }
    .topnav .subtitle { color: #888; font-size: 0.8em; }
    .nav-buttons { display: flex; gap: 10px; }
    .nav-btn { padding: 8px 20px; border-radius: 8px; cursor: pointer; font-weight: bold;
               font-size: 0.95em; border: 2px solid transparent; transition: all 0.2s;
               background: #252840; color: #888; }
    .nav-btn.active { color: #fff; border-color: #448aff; background: #2e3250; }
    .page { display: none; padding: 0 20px 20px 20px; }
    .page.active { display: block; }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
             gap: 15px; margin-bottom: 25px; }
    .card { background: #1a1d2e; border-radius: 12px; padding: 18px; }
    .card h2 { font-size: 0.75em; color: #888; margin-bottom: 8px; }
    .card .value { font-size: 1.4em; font-weight: bold; }
    .card .value.small { font-size: 1em; }
    .green { color: #00c853; } .red { color: #ff1744; }
    .yellow { color: #ffd600; } .blue { color: #448aff; }
    .market-bar { background: #1a1d2e; border-radius: 12px; padding: 18px;
                  margin-bottom: 25px; display: flex; gap: 30px; flex-wrap: wrap; }
    .market-item h2 { font-size: 0.75em; color: #888; margin-bottom: 4px; }
    .market-item .val { font-size: 1.2em; font-weight: bold; }
    .vix-low { color: #00c853; } .vix-mid { color: #ffd600; } .vix-high { color: #ff1744; }
    .ticker-wrap { background: #1a1d2e; border-radius: 10px; padding: 12px;
                   margin-bottom: 25px; overflow: hidden; }
    #ticker { display: flex; gap: 40px; white-space: nowrap;
              animation: scroll 60s linear infinite; }
    #ticker:hover { animation-play-state: paused; }
    @keyframes scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }
    .chart-wrap { background: #1a1d2e; border-radius: 12px; padding: 20px; margin-bottom: 25px; }
    .chart-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
    .chart-header h2 { font-size: 1em; color: #888; }
    .chart-toggles { display: flex; gap: 8px; }
    .chart-toggle { padding: 4px 12px; border-radius: 20px; font-size: 0.75em;
                    font-weight: bold; cursor: pointer; border: 2px solid; transition: all 0.2s; }
    .chart-toggle.total  { border-color: #00c853; color: #00c853; }
    .chart-toggle.stocks { border-color: #448aff; color: #448aff; }
    .chart-toggle.crypto { border-color: #ffd600; color: #ffd600; }
    .chart-toggle.active.total  { background: #00c853; color: #000; }
    .chart-toggle.active.stocks { background: #448aff; color: #000; }
    .chart-toggle.active.crypto { background: #ffd600; color: #000; }
    table { width: 100%; border-collapse: collapse; background: #1a1d2e;
            border-radius: 12px; overflow: hidden; margin-bottom: 25px; }
    th { background: #252840; padding: 12px 15px; text-align: left; font-size: 0.8em; color: #888; }
    td { padding: 12px 15px; border-top: 1px solid #252840; font-size: 0.9em; }
    .badge { padding: 3px 10px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }
    .badge.buy   { background: #00c85322; color: #00c853; }
    .badge.sell  { background: #ff174422; color: #ff1744; }
    .badge.hold  { background: #ffd60022; color: #ffd600; }
    .badge.error { background: #88888822; color: #888; }
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
  </style>
</head>
<body>

  <!-- STICKY TOP NAV -->
  <div class="topnav">
    <div>
      <div class="topnav h1">🤖 Trading Bot</div>
      <div class="subtitle">Updated: {{ updated }} &bull; auto-refresh 5min</div>
    </div>
    <div class="nav-buttons">
      <div class="nav-btn active" onclick="switchPage('stocks', this)">📈 Stocks ({{ stocks|length }})</div>
      <div class="nav-btn"       onclick="switchPage('crypto', this)">🪙 Crypto ({{ crypto|length }})</div>
    </div>
  </div>

  <!-- SHARED: ticker tape always visible -->
  <div style="padding: 0 20px;">
    <div class="ticker-wrap">
      <div id="ticker">
        {% for s in stocks + crypto %}
        <span style="font-size:0.95em">
          <span class="tip"><strong>{{ s.ticker }}</strong><span class="tiptext">{{ s.name }}</span></span>
          <span style="color:#888">${{ s.price }}</span>
          {% if s.signal == 'BUY' %}<span style="color:#00c853">▲</span>
          {% elif s.signal == 'SELL' %}<span style="color:#ff1744">▼</span>
          {% else %}<span style="color:#ffd600">—</span>{% endif %}
        </span>
        {% endfor %}
        {% for s in stocks + crypto %}
        <span style="font-size:0.95em">
          <span class="tip"><strong>{{ s.ticker }}</strong><span class="tiptext">{{ s.name }}</span></span>
          <span style="color:#888">${{ s.price }}</span>
          {% if s.signal == 'BUY' %}<span style="color:#00c853">▲</span>
          {% elif s.signal == 'SELL' %}<span style="color:#ff1744">▼</span>
          {% else %}<span style="color:#ffd600">—</span>{% endif %}
        </span>
        {% endfor %}
      </div>
    </div>
  </div>

  <!-- STOCKS PAGE -->
  <div id="page-stocks" class="page active">

    <div class="cards">
      <div class="card"><h2>💰 CASH</h2><div class="value green">${{ cash }}</div></div>
      <div class="card"><h2>📈 PORTFOLIO</h2><div class="value">${{ portfolio }}</div></div>
      <div class="card"><h2>📊 POSITIONS</h2><div class="value yellow">{{ position_count }}</div></div>
      <div class="card"><h2>🎯 TODAY P&L</h2>
        <div class="value {{ 'green' if pnl >= 0 else 'red' }}">${{ pnl }}</div></div>
      <div class="card"><h2>🧠 STOCK MODEL</h2>
        <div class="value small blue">{{ stock_model_info.get('accuracy','—') }}</div></div>
      <div class="card"><h2>📅 TRAINED</h2>
        <div class="value small">{{ stock_model_info.get('trained_at','—') }}</div></div>
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
          {{ '↑ Up' if market.spy_trend else '↓ Down' }}</div></div>
      <div class="market-item"><h2>💻 NASDAQ</h2>
        <div class="val {{ 'green' if market.qqq_trend else 'red' }}">
          {{ '↑ Up' if market.qqq_trend else '↓ Down' }}</div></div>
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
        <div class="chart-toggles">
          <div class="chart-toggle total active"  onclick="toggleLine(0, this)">Total</div>
          <div class="chart-toggle stocks active" onclick="toggleLine(1, this)">📈 Stocks</div>
          <div class="chart-toggle crypto active" onclick="toggleLine(2, this)">🪙 Crypto</div>
          <div class="chart-toggle cash active"   onclick="toggleLine(3, this)">💵 Cash</div>
        </div>
      </div>
      <canvas id="portfolioChart" height="40"></canvas>
    </div>
    {% endif %}

    <h2 class="section-title">📊 Stock Signals</h2>
    <table>
      <tr><th>STOCK</th><th>PRICE</th><th>SIGNAL</th><th>CONFIDENCE</th><th>SENTIMENT</th><th>HOLDING</th></tr>
      {% for s in stocks %}
      <tr>
        <td><span class="tip"><strong>{{ s.ticker }}</strong><span class="tiptext">{{ s.name }}</span></span></td>
        <td>${{ s.price }}</td>
        <td>{% if s.signal == 'BUY' %}<span class="badge buy">▲ BUY</span>
            {% elif s.signal == 'SELL' %}<span class="badge sell">▼ SELL</span>
            {% elif s.signal == 'HOLD' %}<span class="badge hold">— HOLD</span>
            {% else %}<span class="badge error">ERROR</span>{% endif %}</td>
        <td>{{ s.confidence }}</td>
        <td>{{ s.sentiment }}</td>
        <td>{{ s.shares }} shares</td>
      </tr>
      {% endfor %}
    </table>

    <h2 class="section-title">📊 Open Stock Positions</h2>
    <table>
      <tr><th>STOCK</th><th>QTY</th><th>BOUGHT AT</th><th>CURRENT</th><th>VALUE</th><th>UNREALISED P&L</th></tr>
      {% for p in stock_positions %}
      <tr>
        <td><span class="tip"><strong>{{ p.symbol }}</strong><span class="tiptext">{{ p.name }}</span></span></td>
        <td>{{ p.qty }} shares</td>
        <td>${{ p.avg_price }}</td>
        <td>${{ p.current }}</td>
        <td>${{ p.value }}</td>
        <td class="{{ 'green' if p.pl >= 0 else 'red' }}">
          {{ '▲' if p.pl >= 0 else '▼' }} ${{ p.pl_abs }} ({{ p.pl_pct }}%)</td>
      </tr>
      {% else %}
      <tr><td colspan="6" style="padding:20px;text-align:center;color:#888">No open stock positions</td></tr>
      {% endfor %}
    </table>

    <h2 class="section-title">📋 Recent Stock Orders</h2>
    <table>
      <tr><th>TIME</th><th>STOCK</th><th>ACTION</th><th>QTY</th><th>PRICE</th></tr>
      {% for o in orders %}
      <tr>
        <td>{{ o.time }}</td>
        <td><span class="tip"><strong>{{ o.symbol }}</strong><span class="tiptext">{{ o.name }}</span></span></td>
        <td>{% if o.side == 'buy' %}<span class="badge buy">BUY</span>
            {% else %}<span class="badge sell">SELL</span>{% endif %}</td>
        <td>{{ o.qty }}</td><td>${{ o.price }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>

  <!-- CRYPTO PAGE -->
  <div id="page-crypto" class="page">

    <div class="cards">
      <div class="card"><h2>💰 CASH</h2><div class="value green">${{ cash }}</div></div>
      <div class="card"><h2>📈 PORTFOLIO</h2><div class="value">${{ portfolio }}</div></div>
      <div class="card"><h2>🪙 OPEN</h2><div class="value yellow">{{ crypto_positions }}</div></div>
      <div class="card"><h2>🎯 TODAY P&L</h2>
        <div class="value {{ 'green' if pnl >= 0 else 'red' }}">${{ pnl }}</div></div>
      <div class="card"><h2>🧠 CRYPTO MODEL</h2>
        <div class="value small blue">{{ crypto_model_info.get('accuracy','—') }}</div></div>
      <div class="card"><h2>📅 TRAINED</h2>
        <div class="value small">{{ crypto_model_info.get('trained_at','—') }}</div></div>
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
          {{ '↑ Up' if market.spy_trend else '↓ Down' }}</div></div>
      <div class="market-item"><h2>💻 NASDAQ</h2>
        <div class="val {{ 'green' if market.qqq_trend else 'red' }}">
          {{ '↑ Up' if market.qqq_trend else '↓ Down' }}</div></div>
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
      </div>
      <canvas id="portfolioChart2" height="40"></canvas>
    </div>
    {% endif %}

    <h2 class="section-title">🪙 Crypto Signals (Hourly Model)</h2>
    <table>
      <tr><th>COIN</th><th>PRICE</th><th>SIGNAL</th><th>CONFIDENCE</th><th>SENTIMENT</th><th>HOLDING</th></tr>
      {% for c in crypto %}
      <tr>
        <td><span class="tip"><strong>{{ c.ticker }}</strong><span class="tiptext">{{ c.name }}</span></span></td>
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
    </table>

    <h2 class="section-title">🪙 Open Crypto Positions</h2>
    <table>
      <tr><th>COIN</th><th>QTY</th><th>BOUGHT AT</th><th>CURRENT</th><th>VALUE</th><th>UNREALISED P&L</th></tr>
      {% for p in crypto_positions_list %}
      <tr>
        <td><span class="tip"><strong>{{ p.symbol }}</strong><span class="tiptext">{{ p.name }}</span></span></td>
        <td>{{ p.qty }} coins</td>
        <td>${{ p.avg_price }}</td>
        <td>${{ p.current }}</td>
        <td>${{ p.value }}</td>
        <td class="{{ 'green' if p.pl >= 0 else 'red' }}">
          {{ '▲' if p.pl >= 0 else '▼' }} ${{ p.pl_abs }} ({{ p.pl_pct }}%)</td>
      </tr>
      {% else %}
      <tr><td colspan="6" style="padding:20px;text-align:center;color:#888">No open crypto positions</td></tr>
      {% endfor %}
    </table>

    <h2 class="section-title">📋 Recent Crypto Orders</h2>
    <table>
      <tr><th>TIME</th><th>COIN</th><th>ACTION</th><th>QTY</th><th>PRICE</th></tr>
      {% for o in crypto_orders %}
      <tr>
        <td>{{ o.time }}</td>
        <td><span class="tip"><strong>{{ o.symbol }}</strong><span class="tiptext">{{ o.name }}</span></span></td>
        <td>{% if o.side == 'buy' %}<span class="badge buy">BUY</span>
            {% else %}<span class="badge sell">SELL</span>{% endif %}</td>
        <td>{{ o.qty }}</td><td>${{ o.price }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>

  <script>
    {% if chart_labels %}
    const labels       = {{ chart_labels | safe }};
    const valTotal     = {{ chart_values | safe }};
    const valStocks    = {{ chart_stock_values | safe }};
    const valCrypto    = {{ chart_crypto_values | safe }};
    const valCash      = {{ chart_cash_values | safe }};

    const chart = new Chart(
      document.getElementById('portfolioChart').getContext('2d'), {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          { label: 'Total ($)',  data: valTotal,  borderColor: '#00c853',
            backgroundColor: 'rgba(0,200,83,0.08)', borderWidth: 2,
            pointRadius: 2, fill: true, tension: 0.3 },
          { label: 'Stocks ($)', data: valStocks, borderColor: '#448aff',
            backgroundColor: 'rgba(68,138,255,0.05)', borderWidth: 2,
            pointRadius: 2, fill: false, tension: 0.3 },
          { label: 'Crypto ($)', data: valCrypto, borderColor: '#ffd600',
            backgroundColor: 'rgba(255,214,0,0.05)', borderWidth: 2,
            pointRadius: 2, fill: false, tension: 0.3 },
          { label: 'Cash ($)', data: valCash, borderColor: '#b0b8d8',
            backgroundColor: 'rgba(176,184,216,0.05)', borderWidth: 2,
            pointRadius: 2, fill: false, tension: 0.3 }
        ]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#888', maxTicksLimit: 10 }, grid: { color: '#252840' } },
          y: { ticks: { color: '#888', callback: v => '$' + v.toLocaleString() }, grid: { color: '#252840' } }
        }
      }
    });

    const chart2 = new Chart(
      document.getElementById('portfolioChart2').getContext('2d'), {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          { label: 'Total ($)',  data: valTotal,  borderColor: '#00c853',
            backgroundColor: 'rgba(0,200,83,0.08)', borderWidth: 2,
            pointRadius: 2, fill: true, tension: 0.3 },
          { label: 'Crypto ($)', data: valCrypto, borderColor: '#ffd600',
            backgroundColor: 'rgba(255,214,0,0.05)', borderWidth: 2,
            pointRadius: 2, fill: false, tension: 0.3 }
        ]
      },
      options: {
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#888', maxTicksLimit: 10 }, grid: { color: '#252840' } },
          y: { ticks: { color: '#888', callback: v => '$' + v.toLocaleString() }, grid: { color: '#252840' } }
        }
      }
    });

    function toggleLine(index, el) {
      const meta = chart.getDatasetMeta(index);
      meta.hidden = !meta.hidden;
      el.classList.toggle('active');
      chart.update();
    }
    {% endif %}

    function switchPage(name, el) {
      document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('page-' + name).classList.add('active');
      el.classList.add('active');
    }
  </script>
</body>
</html>
"""

@app.route('/')
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
            sent_score  = get_sentiment(ticker)
            signal      = 'BUY' if pred == 1 else 'SELL'
            if conf < 0.60:
                signal = 'HOLD'
            elif signal == 'BUY' and sent_score < -0.05:
                signal = 'HOLD'
            elif signal == 'SELL' and sent_score > 0.05:
                signal = 'HOLD'
            sent_label = '🟢' if sent_score > 0.05 else '🔴' if sent_score < -0.05 else '⚪'
            p          = pos_dict.get(ticker)
            shares     = int(float(p.qty)) if p else 0
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
            sent_score  = get_crypto_sentiment(symbol)
            sent_label  = '🟢' if sent_score > 0.05 else '🔴' if sent_score < -0.05 else '⚪'
            if signal == 'BUY' and sent_score < -0.05:
                signal = 'HOLD'
            elif signal == 'SELL' and sent_score > 0.05:
                signal = 'HOLD'
            alpaca_symbol = symbol.replace('/', '')
            p             = pos_dict.get(alpaca_symbol)
            coins         = round(float(p.qty), 6) if p else 0
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

    raw_orders    = api.list_orders(status='all', limit=20)
    crypto_syms   = list(CRYPTOS.keys())
    orders        = [{
        'time':   o.submitted_at.strftime('%m/%d %H:%M') if o.submitted_at else '—',
        'symbol': o.symbol,
        'name':   COMPANY_NAMES.get(o.symbol, o.symbol),
        'side':   o.side, 'qty': o.qty,
        'price':  o.filled_avg_price or '—'
    } for o in raw_orders if o.symbol in TICKERS]
    crypto_orders = [{
        'time':   o.submitted_at.strftime('%m/%d %H:%M') if o.submitted_at else '—',
        'symbol': o.symbol,
        'name':   COMPANY_NAMES.get(o.symbol, o.symbol),
        'side':   o.side, 'qty': o.qty,
        'price':  o.filled_avg_price or '—'
    } for o in raw_orders if o.symbol in crypto_syms]

    # Portfolio history for chart
    chart_labels        = []
    chart_values        = []
    chart_stock_values  = []
    chart_crypto_values = []
    chart_cash_values   = []
    if os.path.exists('/home/jv/portfolio_history.json'):
        with open('/home/jv/portfolio_history.json') as f:
            history = json.load(f)
        chart_labels        = [h['time'] for h in history]
        chart_values        = [h['value'] for h in history]
        chart_stock_values  = [h.get('stock_value', 0) for h in history]
        chart_crypto_values = [h.get('crypto_value', 0) for h in history]
        chart_cash_values   = [h.get('cash', 0) for h in history]

    # Build positions lists
    stock_positions_list  = []
    crypto_positions_list = []
    for p in positions:
        row = {
            'symbol':    p.symbol,
            'name':      COMPANY_NAMES.get(p.symbol, p.symbol),
            'qty':       int(float(p.qty)) if p.symbol in TICKERS else round(float(p.qty), 6),
            'avg_price': f"{float(p.avg_entry_price):,.4f}",
            'current':   f"{float(p.current_price):,.4f}",
            'value':     f"{float(p.market_value):,.2f}",
            'pl':        float(p.unrealized_pl),
            'pl_abs':    f"{abs(float(p.unrealized_pl)):,.2f}",
            'pl_pct':    f"{float(p.unrealized_plpc)*100:+.2f}",
        }
        if p.symbol in TICKERS:
            stock_positions_list.append(row)
        else:
            crypto_positions_list.append(row)

    crypto_positions_count = len(crypto_positions_list)

    return render_template_string(HTML,
        cash=cash, portfolio=portfolio, pnl=pnl,
        position_count=len(positions),
        stocks=stocks, crypto=crypto,
        orders=orders, crypto_orders=crypto_orders,
        updated=datetime.now().strftime('%H:%M:%S'),
        chart_labels=json.dumps(chart_labels),
        chart_values=json.dumps(chart_values),
        chart_stock_values=json.dumps(chart_stock_values),
        chart_crypto_values=json.dumps(chart_crypto_values),
        chart_cash_values=json.dumps(chart_cash_values),
        stock_model_info=stock_model_info,
        crypto_model_info=crypto_model_info,
        market=market_data,
        crypto_positions=crypto_positions_count,
        stock_positions=stock_positions_list,
        crypto_positions_list=crypto_positions_list)


# ─────────────────────────────────────────────────────────────
#  MONITOR PAGE  –  fullscreen display for the Pi's monitor
#  Visit http://localhost:5000/monitor
#  Designed to fill a 1080p/1440p landscape screen, no scrolling.
#  Auto-refreshes every 60 seconds.
# ─────────────────────────────────────────────────────────────

MONITOR_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Trading Bot — Monitor</title>
  <meta name="viewport" content="width=1920">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700;900&display=swap" rel="stylesheet">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
  <style>
    :root {
      --bg:        #04060e;
      --panel:     #0b0e1c;
      --border:    #1e2540;
      --border2:   #2d3660;
      --text:      #e8eeff;
      --muted:     #8b96c0;
      --dim:       #5a6490;
      --green:     #00f07a;
      --red:       #ff4d6a;
      --gold:      #ffd740;
      --blue:      #5dccff;
      --purple:    #c9b5ff;
      --font-mono: 'Share Tech Mono', monospace;
      --font-ui:   'Barlow Condensed', sans-serif;
      font-size:   17px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    html, body {
      width: 100%; height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: var(--font-ui);
      overflow: hidden;
    }

    /* ── Layout ── */
    .grid {
      display: grid;
      grid-template-columns: 280px 1fr 300px;
      grid-template-rows: 64px 1fr 1fr 120px;
      grid-template-areas:
        "header  header  header"
        "sidebar chart   metrics"
        "sidebar chart   metrics"
        "footer  footer  footer";
      gap: 8px;
      padding: 8px;
      width: 100vw;
      height: 100vh;
    }

    /* ── Header ── */
    .header {
      grid-area: header;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--panel);
      border: 1px solid var(--border2);
      border-radius: 6px;
      padding: 0 20px;
    }
    .header-left {
      display: flex; align-items: center; gap: 18px;
    }
    .header-logo {
      font-family: var(--font-mono);
      font-size: 1.05em;
      color: var(--blue);
      letter-spacing: 0.04em;
    }
    .header-logo span { color: var(--dim); }
    .header-clock {
      font-family: var(--font-mono);
      font-size: 1.55em;
      color: var(--text);
      letter-spacing: 0.06em;
    }
    .header-right {
      display: flex; align-items: center; gap: 26px;
    }
    .hstat { text-align: right; }
    .hstat-label { font-size: 0.68em; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; }
    .hstat-val   { font-size: 1.25em; font-weight: 700; letter-spacing: 0.02em; }
    .hstat-val--hero { font-size: 1.75em; font-weight: 900; letter-spacing: 0.01em; }
    .divider { width: 1px; height: 32px; background: var(--border2); }
    .status-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 10px var(--green);
      animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse {
      0%,100% { opacity: 1; } 50% { opacity: 0.35; }
    }

    /* ── Sidebar ── */
    .sidebar {
      grid-area: sidebar;
      display: flex; flex-direction: column; gap: 8px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 12px 14px;
    }
    .panel-title {
      font-size: 0.7em; font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
      padding-bottom: 7px;
      border-bottom: 1px solid var(--border2);
    }

    /* Market indicators */
    .mkt-row {
      display: flex; justify-content: space-between; align-items: baseline;
      padding: 7px 0;
      border-bottom: 1px solid var(--border);
      font-size: 0.92em;
    }
    .mkt-row:last-child { border-bottom: none; }
    .mkt-label { color: var(--muted); font-size: 0.82em; letter-spacing: 0.07em; font-weight: 600; }
    .mkt-val { font-family: var(--font-mono); font-weight: 700; font-size: 1.05em; }

    /* Ticker tape */
    .ticker-panel {
      overflow: hidden;
      flex: 1;
      display: flex; flex-direction: column;
    }
    .ticker-scroll-wrap { overflow: hidden; flex: 1; position: relative; }
    .ticker-scroll {
      display: flex; flex-direction: column; gap: 2px;
      animation: tickerScroll 60s linear infinite;
    }
    .ticker-scroll:hover { animation-play-state: paused; }
    @keyframes tickerScroll {
      0%   { transform: translateY(0); }
      100% { transform: translateY(-50%); }
    }
    .ticker-item {
      display: flex; justify-content: space-between; align-items: center;
      padding: 6px 8px;
      border-radius: 5px;
      font-size: 0.95em;
      border: 1px solid transparent;
      gap: 4px;
    }
    .ticker-item.held { border-color: var(--border2); background: rgba(255,255,255,0.04); }
    .ticker-sym   { font-family: var(--font-mono); font-weight: 700; color: var(--text); min-width: 50px; }
    .ticker-price { font-family: var(--font-mono); color: var(--muted); font-size: 0.88em; flex: 1; }
    .ticker-right { display: flex; align-items: center; gap: 6px; }
    .ticker-sig   { font-size: 0.85em; font-weight: 900; padding: 3px 9px; border-radius: 4px; letter-spacing: 0.07em; }
    .sig-open    { color: #040e1a; background: var(--blue); }
    .sig-buymore { color: #0a1a10; background: var(--green); }
    .sig-hold    { color: #1a1400; background: var(--gold); }
    .sig-sell    { color: #1a040a; background: var(--red); }
    .dim { color: var(--dim); }

    /* ── Chart area ── */
    .chart-area {
      grid-area: chart;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px 16px;
      display: flex; flex-direction: column;
    }
    .chart-header {
      display: flex; justify-content: space-between; align-items: center;
      margin-bottom: 10px;
    }
    .chart-title {
      font-size: 0.72em; font-weight: 700; letter-spacing: 0.14em;
      text-transform: uppercase; color: var(--muted);
    }
    .chart-controls {
      display: flex; align-items: center; gap: 16px;
    }
    .chart-view-btns {
      display: flex; gap: 5px;
    }
    .view-btn {
      font-family: var(--font-mono); font-size: 0.65em; font-weight: 700;
      letter-spacing: 0.1em; padding: 3px 10px; border-radius: 4px; cursor: pointer;
      border: 1px solid var(--border2); background: transparent; color: var(--muted);
      transition: all 0.15s;
    }
    .view-btn.active { background: var(--border2); color: var(--text); }
    .view-btn:hover  { color: var(--text); }
    .chart-legend {
      display: flex; gap: 16px;
    }
    .legend-item {
      display: flex; align-items: center; gap: 6px;
      font-size: 0.82em; color: var(--muted); font-weight: 600;
    }
    .legend-dot { width: 10px; height: 10px; border-radius: 50%; }
    .chart-canvas-wrap { flex: 1; position: relative; min-height: 0; }

    /* ── Metrics — right column, 4 cards stacked ── */
    .metrics {
      grid-area: metrics;
      display: flex; flex-direction: column; gap: 8px;
    }
    .metric-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 11px;
      display: flex; flex-direction: column;
      flex: 1;
    }
    .metric-label {
      font-size: 0.7em; font-weight: 700; letter-spacing: 0.13em;
      text-transform: uppercase; color: var(--muted);
      margin-bottom: 2px;
    }
    .metric-val {
      font-family: var(--font-mono);
      font-size: 1.25em;
      font-weight: 600;
      line-height: 1;
    }
    .metric-sub {
      font-size: 0.72em; color: var(--muted); margin-top: 1px;
    }
    /* Dense stat rows inside metric cards */
    .metric-stats {
      display: flex; flex-direction: column; gap: 3px;
      margin-top: 5px; padding-top: 5px;
      border-top: 1px solid var(--border);
      flex: 1;
    }
    .mstat {
      display: flex; justify-content: space-between; align-items: baseline;
      font-size: 0.75em;
    }
    .mstat-label { color: var(--muted); font-weight: 600; }
    .mstat-val   { font-family: var(--font-mono); color: var(--text); font-weight: 700; }

    /* ── Recent Trades card ── */
    .trades-card  { flex: 3; min-height: 0; display: flex; flex-direction: column; }
    .trades-wrap  {
      overflow: hidden; flex: 1; margin-top: 6px; padding-top: 6px;
      border-top: 1px solid var(--border); min-height: 0;
      display: flex; flex-direction: column; gap: 4px;
    }
    .trade-row {
      display: grid; grid-template-columns: 5.5em 3.2em 3.2em 1fr auto;
      gap: 5px; align-items: baseline;
      font-size: 0.76em; font-family: var(--font-mono);
    }
    .trade-time  { color: var(--muted); }
    .trade-sym   { font-weight: 700; color: var(--text); }
    .trade-side.buy  { color: var(--green); font-weight: 700; }
    .trade-side.sell { color: var(--red);   font-weight: 700; }
    .trade-qty   { color: var(--dim); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .trade-price { color: var(--text); text-align: right; }
    .trade-empty { color: var(--muted); font-size: 0.8em; font-style: italic; margin-top: 6px; }


    /* ── Footer — 3-row scrolling position ticker ── */
    .footer {
      grid-area: footer;
      background: var(--panel);
      border: 1px solid var(--border2);
      border-radius: 6px;
      display: flex; align-items: stretch;
      overflow: hidden;
    }
    .footer-label {
      flex-shrink: 0; padding: 0 14px;
      font-size: 0.65em; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
      color: var(--muted); border-right: 1px solid var(--border2);
      white-space: nowrap; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 6px;
    }
    .footer-label-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--green); box-shadow: 0 0 8px var(--green);
      animation: pulse 2s ease-in-out infinite;
    }
    /* Three stacked rows, each scrolling independently */
    .footer-rows {
      flex: 1; overflow: hidden;
      display: flex; flex-direction: column;
    }
    .footer-row {
      flex: 1; overflow: hidden;
      display: flex; align-items: center;
      border-bottom: 1px solid var(--border);
    }
    .footer-row:last-child { border-bottom: none; }
    .footer-row-label {
      flex-shrink: 0; width: 64px; padding: 0 10px;
      font-size: 0.6em; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
      color: var(--dim); border-right: 1px solid var(--border);
      height: 100%; display: flex; align-items: center; justify-content: flex-end;
    }
    .ft-track { overflow: hidden; flex: 1; height: 100%; display: flex; align-items: center; }
    .ft-belt {
      display: flex; align-items: center; white-space: nowrap;
      animation: footerScroll 60s linear infinite;
    }
    .ft-belt.paused { animation-play-state: paused; }
    @keyframes footerScroll { 0% { transform: translateX(0); } 100% { transform: translateX(-50%); } }

    /* Each position chip — three rows show different data for the same position */
    .ft-chip {
      display: inline-flex; align-items: center;
      padding: 0 20px; border-right: 1px solid var(--border);
      min-width: 120px;
    }
    /* Row 1: symbol (large) */
    .ft-chip-sym {
      font-family: var(--font-mono); font-weight: 800;
      font-size: 1.05em; color: var(--text); letter-spacing: 0.05em;
    }
    /* Row 2: current price + P&L */
    .ft-chip-price { font-family: var(--font-mono); font-size: 0.88em; color: var(--muted); font-weight: 600; margin-right: 10px; }
    .ft-chip-pl    { font-family: var(--font-mono); font-size: 0.88em; font-weight: 800; }
    .ft-chip-pct   { font-family: var(--font-mono); font-size: 0.78em; font-weight: 600; opacity: 0.8; margin-left: 4px; }
    /* Row 3: qty, avg, value */
    .ft-chip-meta  { font-family: var(--font-mono); font-size: 0.8em; color: var(--muted); font-weight: 600; }
    .ft-chip-val   { font-family: var(--font-mono); font-size: 0.8em; color: var(--text); font-weight: 700; margin-left: 8px; }

    .footer-meta {
      flex-shrink: 0; padding: 0 14px;
      font-size: 0.6em; font-weight: 600; letter-spacing: 0.1em;
      color: var(--dim); border-left: 1px solid var(--border2);
      white-space: nowrap; display: flex; flex-direction: column;
      align-items: center; justify-content: center; gap: 4px;
    }

    /* Colour utils */
    .green  { color: var(--green); }
    .red    { color: var(--red); }
    .gold   { color: var(--gold); }
    .blue   { color: var(--blue); }
    .purple { color: var(--purple); }
    .vix-ok   { color: var(--green); }
    .vix-warn { color: var(--gold); }
    .vix-bad  { color: var(--red); }
  </style>
</head>
<body>
<div class="grid">

  <!-- ══ HEADER ══ -->
  <header class="header">
    <div class="header-left">
      <div class="status-dot"></div>
      <div class="header-logo">ALGO<span>/</span>TRADER <span style="font-size:0.7em;color:var(--muted)">PAPER</span></div>
      <div class="header-clock" id="clock">--:--:--</div>
    </div>
    <div class="header-right">
      <div class="hstat">
        <div class="hstat-label">Portfolio</div>
        <div class="hstat-val hstat-val--hero blue" id="h-portfolio">${{ portfolio }}</div>
      </div>
      <div class="divider"></div>
      <div class="hstat">
        <div class="hstat-label">Today P&amp;L</div>
        <div class="hstat-val hstat-val--hero {{ 'green' if pnl >= 0 else 'red' }}" id="h-pnl">
          {{ '+' if pnl >= 0 else '' }}${{ '{:,.2f}'.format(pnl) }}
          <span id="h-pnl-pct" style="font-size:0.6em;opacity:0.8;letter-spacing:0.02em">({{ '+' if pnl_pct >= 0 else '' }}{{ '{:.2f}'.format(pnl_pct) }}%)</span>
        </div>
      </div>
      <div class="divider"></div>
      <div class="hstat">
        <div class="hstat-label">Total Return</div>
        <div class="hstat-val hstat-val--hero {{ 'green' if total_return >= 0 else 'red' }}" id="h-return">
          {{ '+' if total_return >= 0 else '' }}{{ '{:.2f}'.format(total_return) }}%
        </div>
      </div>
      <div class="divider"></div>
      <div class="hstat">
        <div class="hstat-label">Cash</div>
        <div class="hstat-val" id="h-cash">${{ cash }}</div>
      </div>
      <div class="divider"></div>
      <div class="hstat">
        <div class="hstat-label">Positions</div>
        <div class="hstat-val gold" id="h-positions">{{ position_count }}</div>
      </div>
      <div class="divider"></div>
      <div class="hstat">
        <div class="hstat-label">Next Stock Trade</div>
        <div class="hstat-val" id="h-stock-countdown" style="font-size:1em;letter-spacing:0.04em">--:--:--</div>
      </div>
      <div class="divider"></div>
      <div class="hstat">
        <div class="hstat-label">Next Crypto Trade</div>
        <div class="hstat-val blue" id="h-crypto-countdown" style="font-size:1em;letter-spacing:0.04em">--:--:--</div>
      </div>
    </div>
  </header>

  <!-- ══ SIDEBAR ══ -->
  <aside class="sidebar">

    <!-- Market indicators -->
    <div class="panel">
      <div class="panel-title">Market Overview</div>
      {% if market %}
      <div class="mkt-row">
        <span class="mkt-label">VIX</span>
        <span class="mkt-val {% if market.vix < 20 %}vix-ok{% elif market.vix < 30 %}vix-warn{% else %}vix-bad{% endif %}">
          {{ market.vix }}
        </span>
      </div>
      <div class="mkt-row">
        <span class="mkt-label">S&amp;P 500</span>
        <span class="mkt-val {{ 'green' if market.spy_trend else 'red' }}">
          {{ '↑ BULL' if market.spy_trend else '↓ BEAR' }}
          <span style="font-size:0.8em;color:var(--muted)">({{ '+' if market.spy_5d >= 0 else '' }}{{ market.spy_5d }}%)</span>
        </span>
      </div>
      <div class="mkt-row">
        <span class="mkt-label">NASDAQ</span>
        <span class="mkt-val {{ 'green' if market.qqq_trend else 'red' }}">
          {{ '↑ BULL' if market.qqq_trend else '↓ BEAR' }}
          <span style="font-size:0.8em;color:var(--muted)">({{ '+' if market.qqq_5d >= 0 else '' }}{{ market.qqq_5d }}%)</span>
        </span>
      </div>
      <div class="mkt-row">
        <span class="mkt-label">BTC Trend</span>
        <span class="mkt-val {{ 'green' if market.btc_trend else 'red' }}">
          {{ '↑ UP' if market.btc_trend else '↓ DOWN' }}
        </span>
      </div>
      {% else %}
      <div style="color:var(--muted);font-size:0.8em;padding:8px 0">Market data unavailable</div>
      {% endif %}
    </div>

    <!-- Scrolling ticker of actionable signals -->
    <div class="panel ticker-panel">
      <div class="panel-title">Next Trade Action</div>
      <div class="ticker-scroll-wrap">
        <div class="ticker-scroll" id="tickerScroll">
          {% for item in all_signals %}
          <div class="ticker-item {{ 'held' if item.held }}">
            <span class="ticker-sym">{{ item.ticker }}</span>
            <span class="ticker-price">${{ item.price }}</span>
            <div class="ticker-right">
              <span class="ticker-sig sig-{{ item.action_cls }}">{{ item.action }}</span>
            </div>
          </div>
          {% endfor %}
          {# Duplicate for seamless loop #}
          {% for item in all_signals %}
          <div class="ticker-item {{ 'held' if item.held }}">
            <span class="ticker-sym">{{ item.ticker }}</span>
            <span class="ticker-price">${{ item.price }}</span>
            <div class="ticker-right">
              <span class="ticker-sig sig-{{ item.action_cls }}">{{ item.action }}</span>
            </div>
          </div>
          {% endfor %}
        </div>
      </div>
    </div>

  </aside>

  <!-- ══ CHART ══ -->
  <section class="chart-area">
    <div class="chart-header">
      <span class="chart-title">Portfolio Performance</span>
      <div class="chart-controls">
        <div class="chart-view-btns">
          <button class="view-btn active" id="vbtn-24h" onclick="setChartView('24h')">24H</button>
          <button class="view-btn"        id="vbtn-7d"  onclick="setChartView('7d')">7D</button>
          <button class="view-btn"        id="vbtn-all" onclick="setChartView('all')">ALL</button>
        </div>
        <div class="chart-legend">
          <div class="legend-item"><div class="legend-dot" style="background:var(--blue)"></div>Stocks</div>
          <div class="legend-item"><div class="legend-dot" style="background:var(--gold)"></div>Crypto</div>
          <div class="legend-item"><div class="legend-dot" style="background:#b0b8d8"></div>Cash</div>
        </div>
      </div>
    </div>
    <div class="chart-canvas-wrap">
      <canvas id="monChart"></canvas>
    </div>
  </section>

  <!-- ══ METRICS ══ -->
  <div class="metrics">

    <!-- Stock Value -->
    <div class="metric-card">
      <div class="metric-label">Stock Value</div>
      <div class="metric-val blue" id="m-stock-val">${{ '{:,.0f}'.format(stock_value) }}</div>
      <div class="metric-sub" id="m-stock-count">{{ stock_position_count }} position{{ 's' if stock_position_count != 1 }}</div>
      <div class="metric-stats">
        <div class="mstat">
          <span class="mstat-label">Today P&amp;L</span>
          <span class="mstat-val {{ 'green' if stock_pnl >= 0 else 'red' }}" id="m-stock-pnl">{{ '+' if stock_pnl >= 0 else '' }}${{ '{:,.2f}'.format(stock_pnl) }}</span>
        </div>
        <div class="mstat">
          <span class="mstat-label">% of Portfolio</span>
          <span class="mstat-val" id="m-stock-pct">{{ stock_pct }}%</span>
        </div>
        <div class="mstat">
          <span class="mstat-label">Unrealised P&amp;L</span>
          <span class="mstat-val {{ 'green' if stock_upl >= 0 else 'red' }}" id="m-stock-upl">{{ '+' if stock_upl >= 0 else '' }}${{ '{:,.2f}'.format(stock_upl) }} <span id="m-stock-upl-pct" style="opacity:0.8">({{ '+' if stock_upl_pct >= 0 else '' }}{{ '{:.2f}'.format(stock_upl_pct) }}%)</span></span>
        </div>
      </div>
    </div>

    <!-- Crypto Value -->
    <div class="metric-card">
      <div class="metric-label">Crypto Value</div>
      <div class="metric-val gold" id="m-crypto-val">${{ '{:,.0f}'.format(crypto_value) }}</div>
      <div class="metric-sub" id="m-crypto-count">{{ crypto_position_count }} position{{ 's' if crypto_position_count != 1 }}</div>
      <div class="metric-stats">
        <div class="mstat">
          <span class="mstat-label">Today P&amp;L</span>
          <span class="mstat-val {{ 'green' if crypto_pnl >= 0 else 'red' }}" id="m-crypto-pnl">{{ '+' if crypto_pnl >= 0 else '' }}${{ '{:,.2f}'.format(crypto_pnl) }}</span>
        </div>
        <div class="mstat">
          <span class="mstat-label">% of Portfolio</span>
          <span class="mstat-val" id="m-crypto-pct">{{ crypto_pct }}%</span>
        </div>
        <div class="mstat">
          <span class="mstat-label">Unrealised P&amp;L</span>
          <span class="mstat-val {{ 'green' if crypto_upl >= 0 else 'red' }}" id="m-crypto-upl">{{ '+' if crypto_upl >= 0 else '' }}${{ '{:,.2f}'.format(crypto_upl) }} <span id="m-crypto-upl-pct" style="opacity:0.8">({{ '+' if crypto_upl_pct >= 0 else '' }}{{ '{:.2f}'.format(crypto_upl_pct) }}%)</span></span>
        </div>
      </div>
    </div>

    <!-- Recent Trades -->
    <div class="metric-card trades-card">
      <div class="metric-label">Recent Trades</div>
      <div class="trades-wrap" id="m-trades-scroll">
        {% for t in recent_trades %}
        <div class="trade-row">
          <span class="trade-time">{{ t.time }}</span>
          <span class="trade-sym">{{ t.symbol }}</span>
          <span class="trade-side {{ 'buy' if t.side == 'BUY' else 'sell' }}">{{ t.side }}</span>
          <span class="trade-qty">{{ t.qty }}</span>
          <span class="trade-price">${{ t.price }}</span>
        </div>
        {% else %}
        <div class="trade-empty">No recent trades</div>
        {% endfor %}
      </div>
    </div>

  </div>


  <!-- ══ FOOTER — 3-row position ticker ══ -->
  <footer class="footer">
    <div class="footer-label">
      <div class="footer-label-dot"></div>
      POSITIONS
    </div>

    <div class="footer-rows">
      {% set all_pos = stock_positions + crypto_positions_list %}

      <!-- Row 1: Symbol -->
      <div class="footer-row">
        <div class="footer-row-label">SYM</div>
        <div class="ft-track">
          <div class="ft-belt" id="belt-sym">
            {% for p in all_pos + all_pos %}
            <div class="ft-chip">
              <span class="ft-chip-sym">{{ p.symbol | replace('/USD','') }}</span>
            </div>
            {% endfor %}
          </div>
        </div>
      </div>

      <!-- Row 2: Price + P&L -->
      <div class="footer-row">
        <div class="footer-row-label">P&amp;L</div>
        <div class="ft-track">
          <div class="ft-belt" id="belt-pl">
            {% for p in all_pos + all_pos %}
            <div class="ft-chip">
              <span class="ft-chip-price">${{ p.current }}</span>
              <span class="ft-chip-pl {{ 'green' if p.pl >= 0 else 'red' }}">{{ '+' if p.pl >= 0 else '-' }}${{ p.pl_abs }}</span>
              <span class="ft-chip-pct {{ 'green' if p.pl >= 0 else 'red' }}">({{ p.pl_pct }}%)</span>
            </div>
            {% endfor %}
          </div>
        </div>
      </div>

      <!-- Row 3: Qty / Avg / Value -->
      <div class="footer-row">
        <div class="footer-row-label">POS</div>
        <div class="ft-track">
          <div class="ft-belt" id="belt-pos">
            {% for p in all_pos + all_pos %}
            <div class="ft-chip">
              <span class="ft-chip-meta">{{ p.qty }}{{ 'sh' if p.symbol in tickers else 'c' }} · avg ${{ p.avg_price }}</span>
              <span class="ft-chip-val">${{ p.value }}</span>
            </div>
            {% endfor %}
          </div>
        </div>
      </div>

    </div>

    <div class="footer-meta">
      <span>UPDATED</span>
      <span id="updated-time">{{ updated }}</span>
      <span style="margin-top:4px;font-size:0.85em">AJAX 60s</span>
    </div>
  </footer>

</div>

<script>
  // ── Live clock ──
  function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent = now.toTimeString().slice(0, 8);
  }
  updateClock();
  setInterval(updateClock, 1000);

  // ── Trade countdowns ──
  // Stock:  :00 of 14,15,16,17,18,19,20,21 UK time, Mon-Fri
  // Crypto: :00 of every even hour, every day

  const STOCK_HOURS = [14,15,16,17,18,19,20,21];

  // Reliably extract UK time parts from any Date using Intl
  function getUKParts(d) {
    const fmt = new Intl.DateTimeFormat('en-GB', {
      timeZone: 'Europe/London',
      hour: 'numeric', minute: 'numeric', second: 'numeric',
      hour12: false,
      weekday: 'short', year: 'numeric', month: 'numeric', day: 'numeric'
    });
    const parts = {};
    fmt.formatToParts(d).forEach(p => { parts[p.type] = p.value; });
    // hour12:false gives '24' for midnight on some engines — normalise
    const h = parseInt(parts.hour) % 24;
    return {
      dow:  ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'].indexOf(parts.weekday),
      h, m: parseInt(parts.minute), s: parseInt(parts.second)
    };
  }

  function fmtCountdown(totalSecs) {
    if (totalSecs <= 0) totalSecs = 0;
    const h = Math.floor(totalSecs / 3600);
    const m = Math.floor((totalSecs % 3600) / 60);
    const s = totalSecs % 60;
    if (h > 0)
      return h + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
    return String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
  }

  // Seconds from now until next stock trade, or -1 if market stays closed 7 days
  function secsToNextStock(now) {
    const uk = getUKParts(now);
    // Walk forward minute by minute is slow; instead compute directly:
    // For each candidate slot (day offset + hour), find first future one
    for (let dayOff = 0; dayOff < 8; dayOff++) {
      // Compute dow for this candidate day
      const candDow = (uk.dow + dayOff) % 7;
      if (candDow === 0 || candDow === 6) continue; // skip weekend

      for (let i = 0; i < STOCK_HOURS.length; i++) {
        const slotH = STOCK_HOURS[i];
        // Seconds from current UK clock to this slot's :00
        const secsToSlot = (slotH - uk.h) * 3600 - uk.m * 60 - uk.s
                         + dayOff * 86400;
        if (secsToSlot > 0) return secsToSlot;
      }
    }
    return -1;
  }

  // Seconds from now until next even-hour :00
  function secsToNextCrypto(now) {
    const uk = getUKParts(now);
    // Next even hour after current time
    const nextEvenH = uk.h % 2 === 0
      ? (uk.m === 0 && uk.s === 0 ? uk.h + 2 : uk.h + 2)   // already on even hour — next one
      : uk.h + 1;                                             // on odd hour — round up
    // Actually: next even hour strictly in future
    const nextH = uk.m === 0 && uk.s === 0
      ? (uk.h % 2 === 0 ? uk.h + 2 : uk.h + 1)
      : (uk.h % 2 === 0 ? uk.h + 2 : uk.h + 1);
    const secsToSlot = (nextH - uk.h) * 3600 - uk.m * 60 - uk.s;
    return secsToSlot > 0 ? secsToSlot : secsToSlot + 7200;
  }

  function setCountdown(elId, secs, activeLabel, closedLabel) {
    const el = document.getElementById(elId);
    if (!el) return;
    const clearCls = s => s.replace(/\b(green|red|gold|blue|purple)\b/g, '').trim();
    if (secs < 0) {
      el.textContent = closedLabel;
      el.className = clearCls(el.className) + ' red';
    } else if (secs <= 60) {
      el.textContent = activeLabel;
      el.className = clearCls(el.className) + ' green';
    } else if (secs < 300) {
      el.textContent = fmtCountdown(secs);
      el.className = clearCls(el.className) + ' gold';
    } else {
      el.textContent = fmtCountdown(secs);
      el.className = clearCls(el.className) + ' green';
    }
  }

  function updateCountdowns() {
    const now  = new Date();
    const sStock  = secsToNextStock(now);
    const sCrypto = secsToNextCrypto(now);
    setCountdown('h-stock-countdown',  sStock,  'TRADING', 'CLOSED');
    // Crypto is always running — use blue as default, gold when close
    const cryptoEl = document.getElementById('h-crypto-countdown');
    if (cryptoEl) {
      const clearCls = s => s.replace(/\b(green|red|gold|blue|purple)\b/g,'').trim();
      cryptoEl.textContent = fmtCountdown(sCrypto);
      cryptoEl.className = clearCls(cryptoEl.className) + (sCrypto < 300 ? ' gold' : ' blue');
    }
  }

  updateCountdowns();
  setInterval(updateCountdowns, 1000);

  // ── Footer belt hover — pause/resume all three rows together ──
  (function() {
    const footer = document.querySelector('.footer');
    const belts  = () => document.querySelectorAll('.ft-belt');
    footer.addEventListener('mouseenter', () => belts().forEach(b => b.classList.add('paused')));
    footer.addEventListener('mouseleave', () => belts().forEach(b => b.classList.remove('paused')));
  })();

  // ── Sync belt chip widths so columns align across the 3 rows ──
  function syncBelts() {
    const belts = document.querySelectorAll('.ft-belt');
    if (belts.length < 3) return;
    const n = belts[0].children.length;
    for (let i = 0; i < n; i++) {
      let maxW = 0;
      belts.forEach(b => { if (b.children[i]) maxW = Math.max(maxW, b.children[i].offsetWidth); });
      belts.forEach(b => { if (b.children[i]) b.children[i].style.minWidth = maxW + 'px'; });
    }
  }
  syncBelts();

  // ── Portfolio chart ──
  {% if chart_views %}
  // Pre-resampled views from server — no date adapter needed, plain category scale
  const CHART_VIEWS = {{ chart_views | safe }};
  let currentView = '24h';

  const ctx = document.getElementById('monChart').getContext('2d');
  const v0  = CHART_VIEWS['24h'];

  const monChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: v0.labels,
      datasets: [
        {
          label: 'Stocks',
          data: v0.stock_values,
          borderColor: 'rgba(93,204,255,1)',
          backgroundColor: 'rgba(93,204,255,0.3)',
          borderWidth: 2, pointRadius: 0, fill: 'origin', tension: 0.3
        },
        {
          label: 'Crypto',
          data: v0.crypto_values,
          borderColor: 'rgba(255,215,64,1)',
          backgroundColor: 'rgba(255,215,64,0.25)',
          borderWidth: 2, pointRadius: 0, fill: '-1', tension: 0.3
        },
        {
          label: 'Cash',
          data: v0.cash_values,
          borderColor: 'rgba(176,184,216,0.85)',
          backgroundColor: 'rgba(176,184,216,0.15)',
          borderWidth: 1.5, pointRadius: 0, fill: '-1', tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { display: false },
        tooltip: { mode: 'index', intersect: false }
      },
      scales: {
        x: {
          ticks: {
            color: '#4a5270',
            font: { family: "'Share Tech Mono'", size: 10 },
            maxTicksLimit: 12,
            maxRotation: 0,
            autoSkip: true,
          },
          grid: { color: 'rgba(26,32,53,0.5)' }
        },
        y: {
          stacked: true,
          ticks: {
            color: '#8b96c0',
            font: { family: "'Share Tech Mono'", size: 10 },
            callback: v => '$' + v.toLocaleString()
          },
          grid: { color: 'rgba(30,37,64,0.8)' }
        }
      }
    }
  });

  function setChartView(view) {
    currentView = view;
    ['24h', '7d', 'all'].forEach(v => {
      document.getElementById('vbtn-' + v).classList.toggle('active', v === view);
    });
    const vd = CHART_VIEWS[view];
    monChart.data.labels           = vd.labels;
    monChart.data.datasets[0].data = vd.stock_values;
    monChart.data.datasets[1].data = vd.crypto_values;
    monChart.data.datasets[2].data = vd.cash_values;
    monChart.update('none');
  }
  {% endif %}

  // ── AJAX refresh — update DOM in place, no page jump ──
  function colourClass(val) { return val >= 0 ? 'green' : 'red'; }
  function setEl(id, text, cls) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    if (cls !== undefined) { el.className = el.className.replace(/\b(green|red|blue|gold|purple)\b/g, '').trim() + ' ' + cls; }
  }
  function setElHtml(id, html, cls) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = html;
    if (cls !== undefined) { el.className = el.className.replace(/\b(green|red|blue|gold|purple)\b/g, '').trim() + ' ' + cls; }
  }

  function rebuildBelt(id, chips) {
    const belt = document.getElementById(id);
    if (!belt) return;
    // Preserve animation offset by reading current transform before swap
    const style = window.getComputedStyle(belt);
    belt.innerHTML = chips + chips; // duplicate for seamless loop
    // Reset animation so it restarts cleanly from current visual position
    belt.style.animation = 'none';
    requestAnimationFrame(() => {
      belt.style.animation = '';
      syncBelts();
    });
  }

  async function ajaxRefresh() {
    try {
      const r = await fetch('/monitor/data');
      if (!r.ok) return;
      const d = await r.json();

      // Header
      setEl('h-portfolio', '$' + d.portfolio);
      setEl('h-pnl',        (d.pnl >= 0 ? '+' : '') + '$' + d.pnl_fmt, colourClass(d.pnl));
      setEl('h-pnl-pct',    '(' + (d.pnl_pct >= 0 ? '+' : '') + d.pnl_pct + '%)');
      setEl('h-cash',      '$' + d.cash);
      setEl('h-return',    (d.total_return >= 0 ? '+' : '') + d.total_return + '%', colourClass(d.total_return));
      setEl('h-positions', d.position_count);

      // Metric cards
      setEl('m-stock-val',   '$' + d.stock_value);
      setEl('m-stock-count', d.stock_position_count + ' position' + (d.stock_position_count !== 1 ? 's' : ''));
      setEl('m-stock-pnl',   (d.stock_pnl >= 0 ? '+' : '') + '$' + d.stock_pnl_fmt, colourClass(d.stock_pnl));
      setEl('m-stock-pct',   d.stock_pct + '%');
      setEl('m-stock-upl',     (d.stock_upl >= 0 ? '+' : '') + '$' + d.stock_upl_fmt, colourClass(d.stock_upl));
      setEl('m-stock-upl-pct', '(' + (d.stock_upl_pct >= 0 ? '+' : '') + d.stock_upl_pct + '%)');

      setEl('m-crypto-val',    '$' + d.crypto_value);
      setEl('m-crypto-count',  d.crypto_position_count + ' position' + (d.crypto_position_count !== 1 ? 's' : ''));
      setEl('m-crypto-pnl',    (d.crypto_pnl >= 0 ? '+' : '') + '$' + d.crypto_pnl_fmt, colourClass(d.crypto_pnl));
      setEl('m-crypto-pct',    d.crypto_pct + '%');
      setEl('m-crypto-upl',    (d.crypto_upl >= 0 ? '+' : '') + '$' + d.crypto_upl_fmt, colourClass(d.crypto_upl));
      setEl('m-crypto-upl-pct','(' + (d.crypto_upl_pct >= 0 ? '+' : '') + d.crypto_upl_pct + '%)');

      // Recent trades
      if (d.recent_trades) {
        const scroll = document.getElementById('m-trades-scroll');
        if (scroll) {
          if (d.recent_trades.length === 0) {
            scroll.innerHTML = '<div class="trade-empty">No recent trades</div>';
          } else {
            scroll.innerHTML = d.recent_trades.map(t =>
              `<div class="trade-row">
                <span class="trade-time">${t.time}</span>
                <span class="trade-sym">${t.symbol}</span>
                <span class="trade-side ${t.side === 'BUY' ? 'buy' : 'sell'}">${t.side}</span>
                <span class="trade-qty">${t.qty}</span>
                <span class="trade-price">$${t.price}</span>
              </div>`
            ).join('');
          }
        }
      }

      // Sidebar ticker
      const tickerEl = document.getElementById('tickerScroll');
      if (tickerEl) {
        const rows = d.signals.map(s =>
          `<div class="ticker-item${s.held ? ' held' : ''}">
            <span class="ticker-sym">${s.ticker}</span>
            <span class="ticker-price">$${s.price}</span>
            <div class="ticker-right">
              <span class="ticker-sig sig-${s.action_cls}">${s.action}</span>
            </div>
          </div>`
        ).join('');
        tickerEl.innerHTML = rows + rows;
      }

      // Footer belts — rebuild HTML then re-sync widths
      const symChips = d.positions.map(p =>
        `<div class="ft-chip"><span class="ft-chip-sym">${p.sym}</span></div>`
      ).join('');
      const plChips = d.positions.map(p =>
        `<div class="ft-chip">
          <span class="ft-chip-price">$${p.current}</span>
          <span class="ft-chip-pl ${colourClass(p.pl)}">${p.pl >= 0 ? '+' : '-'}$${p.pl_abs}</span>
          <span class="ft-chip-pct ${colourClass(p.pl)}"> (${p.pl_pct}%)</span>
        </div>`
      ).join('');
      const posChips = d.positions.map(p =>
        `<div class="ft-chip">
          <span class="ft-chip-meta">${p.qty}${p.is_stock ? 'sh' : 'c'} · avg $${p.avg_price}</span>
          <span class="ft-chip-val">$${p.value}</span>
        </div>`
      ).join('');

      rebuildBelt('belt-sym', symChips);
      rebuildBelt('belt-pl',  plChips);
      rebuildBelt('belt-pos', posChips);

      // Chart — replace view data if a new snapshot arrived
      if (typeof monChart !== 'undefined' && d.chart_views) {
        const newLast = (d.chart_views['24h'].labels || []).slice(-1)[0];
        const curLast = (CHART_VIEWS['24h'].labels || []).slice(-1)[0];
        if (newLast !== curLast) {
          Object.assign(CHART_VIEWS, d.chart_views);
          setChartView(currentView);
        }
      }
      setEl('updated-time', d.updated);
    } catch(e) {
      console.warn('AJAX refresh failed:', e);
    }
  }

  // Refresh every 60 seconds without reloading the page
  setInterval(ajaxRefresh, 60000);
</script>
</body>
</html>
"""


@app.route('/monitor')
def monitor():
    try:
        account = api.get_account()
    except Exception as e:
        return f"<body style='background:#060810;color:#ff3d57;font-family:monospace;padding:40px'>Connection error: {e}<br><a href='/monitor' style='color:#4fc3f7'>Retry</a></body>", 503

    port_val   = float(account.portfolio_value)
    cash_val   = float(account.cash)
    last_eq    = float(account.last_equity)
    pnl        = round(float(account.equity) - last_eq, 2)
    pnl_pct    = round(pnl / last_eq * 100, 2) if last_eq else 0.0
    start_val  = 100000.0
    total_ret  = round(((port_val - start_val) / start_val) * 100, 2)

    positions  = api.list_positions()
    pos_dict   = {p.symbol: p for p in positions}

    # ── Model info ──
    stock_model_info  = {'accuracy': '—', 'trained_at': '—'}
    crypto_model_info = {'accuracy': '—', 'trained_at': '—', 'target': '+0.5% / 6h'}
    if os.path.exists('/home/jv/model_info.json'):
        with open('/home/jv/model_info.json') as f:
            stock_model_info = json.load(f)
    if os.path.exists('/home/jv/crypto_model_info.json'):
        with open('/home/jv/crypto_model_info.json') as f:
            crypto_model_info = json.load(f)

    # ── Market data (use cache if available to avoid slow yf call on monitor load) ──
    market_data = None
    if os.path.exists('/home/jv/market_cache.json'):
        with open('/home/jv/market_cache.json') as f:
            market_data = json.load(f)

    # ── Portfolio history — pre-resampled per view ──
    chart_views = build_chart_views()

    # ── Recent filled orders ──
    crypto_syms = list(CRYPTOS.keys())
    _raw_orders = api.list_orders(status='filled', limit=15)
    recent_trades = []
    for o in _raw_orders:
        sym = o.symbol.replace('/USD', '').replace('USD', '')
        ts  = o.filled_at or o.submitted_at
        recent_trades.append({
            'time':   ts.strftime('%m/%d %H:%M') if ts else '—',
            'symbol': sym,
            'side':   o.side.upper(),
            'qty':    o.filled_qty or o.qty,
            'price':  f"{float(o.filled_avg_price):,.2f}" if o.filled_avg_price else '—',
        })

    stock_positions_list  = []
    crypto_positions_list = []
    stock_value  = 0.0
    crypto_value = 0.0

    def build_sparkline(yf_symbol, interval='5m', period='1d', points=20):
        """Return SVG polyline points string for a 44x22 viewport."""
        try:
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
            if df.empty:
                return ''
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            closes = df['Close'].dropna().tolist()[-points:]
            if len(closes) < 3:
                return ''
            mn, mx = min(closes), max(closes)
            if mx == mn:
                return ''
            w, h, pad = 44, 22, 2
            pts = []
            for i, c in enumerate(closes):
                x = pad + (i / (len(closes) - 1)) * (w - pad * 2)
                y = h - pad - ((c - mn) / (mx - mn)) * (h - pad * 2)
                pts.append(f"{x:.1f},{y:.1f}")
            return ' '.join(pts)
        except Exception:
            return ''

    for p in positions:
        if p.symbol in TICKERS:
            yf_sym   = p.symbol
            interval = '5m'
            period   = '1d'
        else:
            yf_sym   = p.symbol[:-3] + '-USD' if p.symbol.endswith('USD') else p.symbol
            interval = '15m'
            period   = '1d'

        row = {
            'symbol':    p.symbol,
            'name':      COMPANY_NAMES.get(p.symbol, p.symbol),
            'qty':       int(float(p.qty)) if p.symbol in TICKERS else round(float(p.qty), 4),
            'avg_price': f"{float(p.avg_entry_price):,.2f}",
            'current':   f"{float(p.current_price):,.2f}",
            'value':     f"{float(p.market_value):,.0f}",
            'pl':        float(p.unrealized_pl),
            'pl_abs':    f"{abs(float(p.unrealized_pl)):,.2f}",
            'pl_pct':    f"{float(p.unrealized_plpc)*100:+.1f}",
            'spark':     build_sparkline(yf_sym, interval=interval, period=period),
        }
        if p.symbol in TICKERS:
            stock_positions_list.append(row)
            stock_value += float(p.market_value)
        else:
            crypto_positions_list.append(row)
            crypto_value += float(p.market_value)

    # ── Per-asset-class P&L and portfolio split ──
    stock_upl  = sum(row['pl'] for row in stock_positions_list)
    crypto_upl = sum(row['pl'] for row in crypto_positions_list)
    total_invested = stock_value + crypto_value
    stock_pct  = round(stock_value  / port_val * 100, 1) if port_val else 0.0
    crypto_pct = round(crypto_value / port_val * 100, 1) if port_val else 0.0
    stock_cost  = stock_value  - stock_upl
    crypto_cost = crypto_value - crypto_upl
    stock_upl_pct  = round(stock_upl  / stock_cost  * 100, 2) if stock_cost  else 0.0
    crypto_upl_pct = round(crypto_upl / crypto_cost * 100, 2) if crypto_cost else 0.0
    # Approximate today's P&L split by weighting total daily pnl by asset-class share
    stock_pnl  = round(pnl * (stock_value  / total_invested), 2) if total_invested else 0.0
    crypto_pnl = round(pnl * (crypto_value / total_invested), 2) if total_invested else 0.0

    # ── All signals list for the sidebar ticker ──
    # Uses actual ML inference + sentiment override — same logic as trader.py / crypto_trader.py.
    # Sentiment comes from cache only (no live calls on monitor load).
    # If the stock sentiment cache is stale/missing, kick off a background refresh.
    stock_sent_cache = load_stock_sentiment_cache()
    cache_age = min(
        (time.time() - v.get('fetched', 0) for v in stock_sent_cache.values()),
        default=STOCK_SENTIMENT_TTL + 1
    )
    if cache_age > STOCK_SENTIMENT_TTL:
        threading.Thread(target=refresh_stock_sentiment_cache, daemon=True).start()

    def sent_fmt(score):
        """Format sentiment score as a short coloured label."""
        if score is None:
            return {'score': '—', 'cls': 'dim'}
        if score > 0.05:
            return {'score': f'+{score:.2f}', 'cls': 'green'}
        if score < -0.05:
            return {'score': f'{score:.2f}', 'cls': 'red'}
        return {'score': f'{score:+.2f}', 'cls': 'muted'}

    # ── Stock signals: ML inference ──
    # Fetch market indicators and sector returns once, shared across all tickers.
    try:
        mkt_row, _btc = get_market_indicators()
    except Exception:
        mkt_row = None

    all_stock_returns_mon = {}
    for t in TICKERS:
        try:
            df_r = yf.download(t, period="10d", interval="1d", progress=False)
            df_r.columns = [col[0] if isinstance(col, tuple) else col for col in df_r.columns]
            ret = df_r['Close'].pct_change()
            all_stock_returns_mon[t] = {'latest': float(ret.iloc[-1]), '5d': float(ret.iloc[-5:].mean())}
        except Exception:
            all_stock_returns_mon[t] = {'latest': 0.0, '5d': 0.0}

    all_signals = []
    for ticker in TICKERS:
        p = pos_dict.get(ticker)
        try:
            df       = get_stock_features(ticker)
            latest   = df.iloc[-1].copy()
            cur_price = float(df['Close'].iloc[-1])
            mom, mom_5d = get_sector_momentum(ticker, all_stock_returns_mon)
            latest['Sector_Momentum']    = mom
            latest['Sector_Momentum_5d'] = mom_5d
            if mkt_row is not None:
                latest['VIX']           = float(mkt_row['VIX'])
                latest['SPY_Return']    = float(mkt_row['SPY_Return'])
                latest['SPY_Trend']     = float(mkt_row['SPY_Trend'])
                latest['SPY_Return_5d'] = float(mkt_row['SPY_Return_5d'])
                latest['QQQ_Return']    = float(mkt_row['QQQ_Return'])
                latest['QQQ_Trend']     = float(mkt_row['QQQ_Trend'])
                latest['QQQ_Return_5d'] = float(mkt_row['QQQ_Return_5d'])
            else:
                for col in ['VIX','SPY_Return','SPY_Trend','SPY_Return_5d',
                            'QQQ_Return','QQQ_Trend','QQQ_Return_5d']:
                    latest[col] = 0.0
            feature_row = pd.DataFrame([latest[stock_features]])
            pred  = stock_model.predict(feature_row)[0]
            conf  = stock_model.predict_proba(feature_row)[0][pred]
            entry = stock_sent_cache.get(ticker)
            raw_score = entry['score'] if entry else None
            sent_score = raw_score if raw_score is not None else 0.0
            sig = 'BUY' if pred == 1 else 'SELL'
            if conf < 0.60:
                sig = 'HOLD'
            elif sig == 'BUY' and sent_score < -0.05:
                sig = 'HOLD'
            elif sig == 'SELL' and sent_score > 0.05:
                sig = 'HOLD'
            price = f"{cur_price:,.2f}"
        except Exception:
            entry = stock_sent_cache.get(ticker)
            raw_score = entry['score'] if entry else None
            sig   = 'HOLD'
            price = f"{float(p.current_price):,.2f}" if p else "—"
        all_signals.append({
            'ticker': ticker, 'price': price, 'signal': sig,
            'held': bool(p), 'raw_score': raw_score,
        })

    # ── Crypto signals: ML inference ──
    try:
        crypto_mkt_mon = get_crypto_market_indicators()
    except Exception:
        crypto_mkt_mon = None

    all_crypto_returns_mon = {}
    for sym, yf_sym in CRYPTOS.items():
        try:
            df_r = yf.download(yf_sym, period="5d", interval="1h", progress=False)
            df_r.columns = [col[0] if isinstance(col, tuple) else col for col in df_r.columns]
            ret = df_r['Close'].pct_change()
            all_crypto_returns_mon[sym] = {'latest': float(ret.iloc[-1]), '24h': float(ret.iloc[-24:].mean())}
        except Exception:
            all_crypto_returns_mon[sym] = {'latest': 0.0, '24h': 0.0}
    crypto_mom_mon, crypto_mom_24h_mon = get_crypto_momentum(all_crypto_returns_mon)

    for symbol, yf_symbol in CRYPTOS.items():
        alpaca_sym = symbol.replace('/', '')
        p = pos_dict.get(alpaca_sym)
        raw_score = get_crypto_sentiment_cached(symbol)
        try:
            df       = get_crypto_features(yf_symbol)
            latest   = df.iloc[-1].copy()
            cur_price = float(df['Close'].iloc[-1])
            latest['Crypto_Momentum']     = crypto_mom_mon
            latest['Crypto_Momentum_24h'] = crypto_mom_24h_mon
            if crypto_mkt_mon:
                latest['VIX']             = crypto_mkt_mon['VIX']
                latest['SPY_Return']      = crypto_mkt_mon['SPY_Return']
                latest['SPY_Trend']       = crypto_mkt_mon['SPY_Trend']
                latest['SPY_Return_24h']  = crypto_mkt_mon['SPY_Return_24h']
                if symbol == 'BTC/USD':
                    latest['BTC_Return']     = 0.0
                    latest['BTC_Trend']      = 0.0
                    latest['BTC_Return_24h'] = 0.0
                else:
                    latest['BTC_Return']     = crypto_mkt_mon['BTC_Return']
                    latest['BTC_Trend']      = crypto_mkt_mon['BTC_Trend']
                    latest['BTC_Return_24h'] = crypto_mkt_mon['BTC_Return_24h']
            else:
                for col in ['VIX','SPY_Return','SPY_Trend','SPY_Return_24h',
                            'BTC_Return','BTC_Trend','BTC_Return_24h']:
                    latest[col] = 0.0
            feature_row = pd.DataFrame([latest[crypto_features]])
            pred  = crypto_model.predict(feature_row)[0]
            conf  = crypto_model.predict_proba(feature_row)[0][pred]
            sent_score = raw_score if raw_score is not None else 0.0
            sig = 'BUY' if pred == 1 else 'SELL'
            if conf < 0.65:
                sig = 'HOLD'
            elif sig == 'BUY' and sent_score < -0.05:
                sig = 'HOLD'
            elif sig == 'SELL' and sent_score > 0.05:
                sig = 'HOLD'
            price = f"{cur_price:,.4f}"
        except Exception:
            sig   = 'HOLD'
            price = f"{float(p.current_price):,.4f}" if p else "—"
        all_signals.append({
            'ticker': symbol.replace('/USD', ''), 'price': price, 'signal': sig,
            'held': bool(p), 'raw_score': raw_score,
        })

    # ── Filter to actionable only, then attach action label ──
    def _is_actionable(item):
        if item['held']:
            return True
        if item['signal'] == 'BUY' and item['raw_score'] is not None and item['raw_score'] > 0.05:
            return True
        return False

    def _make_action(sig, held):
        if sig == 'BUY':
            return ('BUY MORE', 'buymore') if held else ('HOLD', 'hold')
        if sig == 'SELL':
            return ('SELL', 'sell')
        return ('HOLD', 'hold')

    all_signals = [item for item in all_signals if _is_actionable(item)]
    if not all_signals:  # fallback — show all open positions
        all_signals = [item for item in all_signals if item['held']]
    for item in all_signals:
        item['action'], item['action_cls'] = _make_action(item['signal'], item['held'])
        item.pop('raw_score', None)

    return render_template_string(MONITOR_HTML,
        cash=f"{cash_val:,.2f}",
        portfolio=f"{port_val:,.2f}",
        pnl=pnl,
        pnl_pct=pnl_pct,
        total_return=total_ret,
        position_count=len(positions),
        stock_position_count=len(stock_positions_list),
        crypto_position_count=len(crypto_positions_list),
        stock_value=stock_value,
        crypto_value=crypto_value,
        stock_pnl=stock_pnl,
        crypto_pnl=crypto_pnl,
        stock_pct=stock_pct,
        crypto_pct=crypto_pct,
        stock_upl=stock_upl,
        stock_upl_pct=stock_upl_pct,
        crypto_upl=crypto_upl,
        crypto_upl_pct=crypto_upl_pct,
        stock_positions=stock_positions_list,
        crypto_positions_list=crypto_positions_list,
        all_signals=all_signals,
        market=market_data,
        stock_model_info=stock_model_info,
        crypto_model_info=crypto_model_info,
        recent_trades=recent_trades,
        chart_views=json.dumps(chart_views),
        tickers=TICKERS,
        updated=datetime.now().strftime('%H:%M:%S'),
    )




# ─────────────────────────────────────────────────────────────
#  MONITOR DATA  –  JSON endpoint for AJAX refresh
#  Returns all dynamic values; skips sparklines for speed.
# ─────────────────────────────────────────────────────────────
from flask import jsonify

@app.route('/monitor/data')
def monitor_data():
    try:
        account = api.get_account()
    except Exception as e:
        return jsonify({'error': str(e)}), 503

    port_val  = float(account.portfolio_value)
    cash_val  = float(account.cash)
    last_eq   = float(account.last_equity)
    pnl       = round(float(account.equity) - last_eq, 2)
    pnl_pct   = round(pnl / last_eq * 100, 2) if last_eq else 0.0
    start_val = 100000.0
    total_ret = round(((port_val - start_val) / start_val) * 100, 2)

    positions = api.list_positions()
    pos_dict  = {p.symbol: p for p in positions}

    # Model info
    stock_model_info  = {'accuracy': '—', 'trained_at': '—'}
    crypto_model_info = {'accuracy': '—', 'trained_at': '—', 'target': '+0.5% / 6h', 'buy_rate': '—'}
    if os.path.exists('/home/jv/model_info.json'):
        with open('/home/jv/model_info.json') as f:
            stock_model_info = json.load(f)
    if os.path.exists('/home/jv/crypto_model_info.json'):
        with open('/home/jv/crypto_model_info.json') as f:
            crypto_model_info = json.load(f)

    # Portfolio history — pre-resampled per view
    chart_views = build_chart_views()

    # Recent filled orders
    _raw_orders = api.list_orders(status='filled', limit=15)
    recent_trades = []
    for o in _raw_orders:
        sym = o.symbol.replace('/USD', '').replace('USD', '')
        ts  = o.filled_at or o.submitted_at
        recent_trades.append({
            'time':   ts.strftime('%m/%d %H:%M') if ts else '—',
            'symbol': sym,
            'side':   o.side.upper(),
            'qty':    o.filled_qty or o.qty,
            'price':  f"{float(o.filled_avg_price):,.2f}" if o.filled_avg_price else '—',
        })

    # Positions (no sparklines)
    stock_positions_list  = []
    crypto_positions_list = []
    stock_value  = 0.0
    crypto_value = 0.0
    for p in positions:
        row = {
            'sym':       p.symbol.replace('/USD', '').replace('USD', ''),
            'symbol':    p.symbol,
            'qty':       int(float(p.qty)) if p.symbol in TICKERS else round(float(p.qty), 4),
            'avg_price': f"{float(p.avg_entry_price):,.2f}",
            'current':   f"{float(p.current_price):,.2f}",
            'value':     f"{float(p.market_value):,.0f}",
            'pl':        float(p.unrealized_pl),
            'pl_abs':    f"{abs(float(p.unrealized_pl)):,.2f}",
            'pl_pct':    f"{float(p.unrealized_plpc)*100:+.1f}",
            'is_stock':  p.symbol in TICKERS,
        }
        if p.symbol in TICKERS:
            stock_positions_list.append(row)
            stock_value += float(p.market_value)
        else:
            crypto_positions_list.append(row)
            crypto_value += float(p.market_value)

    all_positions = stock_positions_list + crypto_positions_list

    stock_upl  = sum(r['pl'] for r in stock_positions_list)
    crypto_upl = sum(r['pl'] for r in crypto_positions_list)
    total_inv  = stock_value + crypto_value
    stock_pct  = round(stock_value  / port_val * 100, 1) if port_val else 0.0
    crypto_pct = round(crypto_value / port_val * 100, 1) if port_val else 0.0
    stock_cost  = stock_value  - stock_upl
    crypto_cost = crypto_value - crypto_upl
    stock_upl_pct  = round(stock_upl  / stock_cost  * 100, 2) if stock_cost  else 0.0
    crypto_upl_pct = round(crypto_upl / crypto_cost * 100, 2) if crypto_cost else 0.0
    stock_pnl  = round(pnl * (stock_value  / total_inv), 2) if total_inv else 0.0
    crypto_pnl = round(pnl * (crypto_value / total_inv), 2) if total_inv else 0.0

    # Signals — P&L based + cached sentiment
    stock_sent_cache = load_stock_sentiment_cache()

    def sent_fmt(score):
        if score is None:
            return {'score': '—', 'cls': 'dim'}
        if score > 0.05:
            return {'score': f'+{score:.2f}', 'cls': 'green'}
        if score < -0.05:
            return {'score': f'{score:.2f}', 'cls': 'red'}
        return {'score': f'{score:+.2f}', 'cls': 'muted'}

    _signals_raw = []
    for ticker in TICKERS:
        p     = pos_dict.get(ticker)
        price = f"{float(p.current_price):,.2f}" if p else '—'
        sig   = 'HOLD'
        if p:
            chg = float(p.unrealized_plpc)
            sig = 'BUY' if chg > 0.01 else ('SELL' if chg < -0.02 else 'HOLD')
        entry     = stock_sent_cache.get(ticker)
        raw_score = entry['score'] if entry else None
        _signals_raw.append({'ticker': ticker, 'price': price, 'signal': sig,
                             'held': bool(p), 'raw_score': raw_score})
    for symbol in CRYPTOS:
        alpaca_sym = symbol.replace('/', '')
        p     = pos_dict.get(alpaca_sym)
        price = f"{float(p.current_price):,.2f}" if p else '—'
        sig   = 'HOLD'
        if p:
            chg = float(p.unrealized_plpc)
            sig = 'BUY' if chg > 0.01 else ('SELL' if chg < -0.02 else 'HOLD')
        raw_score = get_crypto_sentiment_cached(symbol)
        _signals_raw.append({'ticker': symbol.replace('/USD', ''), 'price': price, 'signal': sig,
                             'held': bool(p), 'raw_score': raw_score})

    # Filter: open positions always shown; unpositioned only if BUY + positive sentiment
    signals = [s for s in _signals_raw
               if s['held'] or (s['signal'] == 'BUY' and s['raw_score'] is not None and s['raw_score'] > 0.05)]
    if not signals:
        signals = [s for s in _signals_raw if s['held']]
    # Attach action label, strip internal keys
    for s in signals:
        s['action'], s['action_cls'] = (
            ('BUY MORE', 'buymore') if s['signal'] == 'BUY' and s['held'] else
            ('HOLD',     'hold')    if s['signal'] == 'BUY' else
            ('SELL',     'sell')    if s['signal'] == 'SELL' else
            ('HOLD',     'hold')
        )
        s.pop('raw_score', None)

    return jsonify(
        portfolio            = f"{port_val:,.2f}",
        cash                 = f"{cash_val:,.2f}",
        pnl                  = pnl,
        pnl_fmt              = f"{abs(pnl):,.2f}",
        pnl_pct              = pnl_pct,
        total_return         = total_ret,
        position_count       = len(positions),
        stock_value          = f"{stock_value:,.0f}",
        crypto_value         = f"{crypto_value:,.0f}",
        stock_position_count = len(stock_positions_list),
        crypto_position_count= len(crypto_positions_list),
        stock_pnl            = stock_pnl,
        stock_pnl_fmt        = f"{abs(stock_pnl):,.2f}",
        stock_pct            = stock_pct,
        stock_upl            = stock_upl,
        stock_upl_fmt        = f"{abs(stock_upl):,.2f}",
        stock_upl_pct        = stock_upl_pct,
        crypto_pnl           = crypto_pnl,
        crypto_pnl_fmt       = f"{abs(crypto_pnl):,.2f}",
        crypto_pct           = crypto_pct,
        crypto_upl           = crypto_upl,
        crypto_upl_fmt       = f"{abs(crypto_upl):,.2f}",
        crypto_upl_pct       = crypto_upl_pct,
        stock_model_accuracy = stock_model_info.get('accuracy', '—'),
        stock_model_trained  = stock_model_info.get('trained_at', '—'),
        crypto_model_accuracy= crypto_model_info.get('accuracy', '—'),
        crypto_model_trained = crypto_model_info.get('trained_at', '—'),
        crypto_buy_rate      = crypto_model_info.get('buy_rate', '—'),
        positions            = all_positions,
        signals              = signals,
        recent_trades        = recent_trades,
        chart_views          = chart_views,
        updated              = datetime.now().strftime('%H:%M:%S'),
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
    
    