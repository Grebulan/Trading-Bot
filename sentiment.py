import json
import os
import time
import requests

CACHE_FILE = '/home/jv/stock_sentiment_cache.json'
FINBERT_SERVICE_URL = 'http://127.0.0.1:5051/score'
# yfinance's .news started returning [] for every ticker (~2026-10-01), so
# headlines now come from Alpaca's news API using the existing trading keys.
ALPACA_NEWS_URL = 'https://data.alpaca.markets/v1beta1/news'

def load_env():
    env = {}
    with open(os.path.expanduser('~/.env_trading')) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                env[k] = v
    return env

ENV = load_env()

TICKERS = [
    'AAPL','GOOGL','AMZN','PLTR','CRWD',
    'NVDA','AMD','QCOM','ARM','MU',
    'TSLA','SHOP','COIN',
    'JPM','BAC','GS','MS','STT',
    'OXY',
    'GOLD',
    'UNH','ISRG',
    'CAT','VRT',
]

def score_headlines(headlines):
    """Score headlines via the local FinBERT service. Neutral on any failure."""
    try:
        resp = requests.post(FINBERT_SERVICE_URL, json={'headlines': headlines}, timeout=30)
        resp.raise_for_status()
        return resp.json()['scores']
    except Exception as e:
        print(f"  FinBERT service unavailable ({e}), using neutral sentiment")
        return [{'label': 'neutral', 'score': 0.0} for _ in headlines]

def fetch_headlines(ticker, limit=10):
    """Latest headlines tagged with this ticker from Alpaca news."""
    resp = requests.get(
        ALPACA_NEWS_URL,
        params={'symbols': ticker, 'limit': limit, 'sort': 'desc'},
        headers={'APCA-API-KEY-ID': ENV['ALPACA_KEY'],
                 'APCA-API-SECRET-KEY': ENV['ALPACA_SECRET']},
        timeout=15,
    )
    resp.raise_for_status()
    return [a['headline'] for a in resp.json().get('news', []) if a.get('headline')]

def get_sentiment(ticker):
    try:
        headlines = fetch_headlines(ticker)
        if not headlines:
            print(f"  ⚠️  {ticker}: no headlines returned — news source may be broken, using 0.0")
            return 0.0

        results = score_headlines(headlines)
        scores = []
        for r in results:
            if r['label'] == 'positive':
                scores.append(r['score'])
            elif r['label'] == 'negative':
                scores.append(-r['score'])
            else:
                scores.append(0.0)

        avg = round(sum(scores) / len(scores), 3)
        label = '🟢 Positive' if avg > 0.05 else '🔴 Negative' if avg < -0.05 else '⚪ Neutral'
        print(f"  📰 {ticker}: {label} ({avg:.2f}) from {len(scores)} headlines")
        return avg

    except Exception as e:
        print(f"  📰 {ticker}: Sentiment error ({e})")
        return 0.0

def refresh_all_sentiment():
    """Fetch fresh sentiment for every stock ticker and write the cache."""
    cache = {}
    now   = time.time()
    for ticker in TICKERS:
        score = get_sentiment(ticker)
        cache[ticker] = {'score': score, 'fetched': now}
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)
    return cache
