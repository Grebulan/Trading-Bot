import os
import json
import time
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError





analyzer   = SentimentIntensityAnalyzer()
CACHE_FILE = '/home/jv/crypto_sentiment_cache.json'
CACHE_TTL  = 1800  # 30 minutes

SYMBOL_MAP = {
    'BTC/USD':  'BTC',
    'ETH/USD':  'ETH',
    'SOL/USD':  'SOL',
    'LINK/USD': 'LINK',
    'AVAX/USD': 'AVAX',
    'XRP/USD':  'XRP',
    'DOGE/USD': 'DOGE',
    'UNI/USD':  'UNI',
    'DOT/USD':  'DOT',
    'ADA/USD':  'ADA',
}

YF_SYMBOL_MAP = {
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

def load_env():
    env = {}
    with open(os.path.expanduser('~/.env_trading')) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                env[k] = v
    return env

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def fetch_via_cryptopanic(currency):
    """Try CryptoPanic first."""
    try:
        env = load_env()
        key = env.get('CRYPTOPANIC_KEY', '')
        url = f'https://cryptopanic.com/api/developer/v2/posts/?auth_token={key}&currencies={currency}&kind=news'
        r   = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        results = r.json().get('results', [])
        if not results:
            return 0.0
        scores = [analyzer.polarity_scores(p.get('title',''))['compound']
                  for p in results[:10] if p.get('title')]
        return round(sum(scores) / len(scores), 3) if scores else 0.0
    except:
        return None


def fetch_via_yahoo(symbol):
    """Fallback to Yahoo Finance news."""
    def _fetch():
        yf_symbol = YF_SYMBOL_MAP.get(symbol, symbol)
        ticker    = yf.Ticker(yf_symbol)
        news      = ticker.news
        if not news:
            return 0.0
        scores = []
        for item in news[:10]:
            title = item.get('content', {}).get('title', '') or item.get('title', '')
            if title:
                scores.append(analyzer.polarity_scores(title)['compound'])
        return round(sum(scores) / len(scores), 3) if scores else 0.0

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch)
        try:
            return future.result(timeout=30)
        except (FuturesTimeoutError, Exception):
            return 0.0

def refresh_all_sentiment():
    """Fetch sentiment for all coins, CryptoPanic first, Yahoo fallback."""
    cache = load_cache()
    now   = time.time()

    for symbol, currency in SYMBOL_MAP.items():
        score = fetch_via_cryptopanic(currency)
        if score is None:
            score = fetch_via_yahoo(symbol)
            source = 'Yahoo'
        else:
            source = 'CryptoPanic'
            time.sleep(2)

        cache[symbol] = {'score': score, 'fetched': now}
        label = '🟢' if score > 0.05 else '🔴' if score < -0.05 else '⚪'
        print(f"  📰 {symbol}: {label} ({score:.2f}) via {source}")

    save_cache(cache)
    return cache

def get_crypto_sentiment(symbol):
    """Get sentiment, using cache if fresh enough."""
    cache = load_cache()
    now   = time.time()

    if symbol in cache:
        age = now - cache[symbol]['fetched']
        if age < CACHE_TTL:
            return cache[symbol]['score']

    cache = refresh_all_sentiment()
    return cache.get(symbol, {}).get('score', 0.0)
