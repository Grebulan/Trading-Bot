#!/usr/bin/env python3
"""
Strategy lab: stock universes and a daily price cache (2026-11-08).

Universes are chosen to limit hindsight. The S&P 500 uses each stock's join date, so a backtest
can only pick a stock once it was actually in the index. Stocks that later left the index are
missing (Wikipedia only lists today's members), so a small survivorship bias remains. The lab
offsets it by comparing every strategy with random picks from the same universe.
legacy24 is the bot's original hand-picked list and is flagged as hindsight-biased.

  python3 lab_data.py   # refresh universes and prices (the lab does this on Sundays)
"""
import io
import json
import os
import pickle
from datetime import datetime

import pandas as pd
import requests
import yfinance as yf

HOME        = '/home/jv'
UNIVERSES   = f'{HOME}/lab_universes.json'
PRICE_CACHE = f'{HOME}/lab_prices.pkl'
START       = '2013-01-01'   # backtests start 2015 (momentum needs a year of history)

ETF_ROTATION = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC',
                'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'IEF', 'GLD', 'SLV', 'DBC']
LEVERAGED_ETFS = ['TQQQ', 'UPRO', 'SOXL', 'TECL', 'FAS', 'TNA', 'CURE', 'ERX', 'TMF', 'UGL',
                  'SQQQ', 'SPXU', 'SOXS', 'TZA']
LEGACY24 = ['AAPL', 'GOOGL', 'AMZN', 'PLTR', 'CRWD', 'NVDA', 'AMD', 'QCOM', 'ARM', 'MU', 'TSLA', 'SHOP',
            'COIN', 'JPM', 'BAC', 'GS', 'MS', 'STT', 'OXY', 'GOLD', 'UNH', 'ISRG', 'CAT', 'VRT']
BENCHMARKS = ['SPY']

WIKI = {'sp500': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'sp400': 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'}


def yf_symbol(s):      # BRK.B -> BRK-B
    return s.replace('.', '-')


def alpaca_symbol(s):  # BRK-B -> BRK.B
    return s.replace('-', '.')


def fetch_index(name):
    html = requests.get(WIKI[name], headers={'User-Agent': 'Mozilla/5.0 (trading-bot research)'},
                        timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    out = {}
    for _, row in t.iterrows():
        sym = yf_symbol(str(row['Symbol']).strip())
        added = pd.to_datetime(row.get('Date added'), errors='coerce') if 'Date added' in t.columns else pd.NaT
        out[sym] = {'sector': str(row.get('GICS Sector', '')),
                    'added': None if pd.isna(added) else added.strftime('%Y-%m-%d')}
    return out


def refresh_universes():
    """Fetch index members; keep the previous lists if Wikipedia can't be reached."""
    old = load_universes()
    u = {'updated': datetime.now().strftime('%Y-%m-%d'),
         'etf_rotation': {s: {} for s in ETF_ROTATION},
         'leveraged_etfs': {s: {} for s in LEVERAGED_ETFS},
         'legacy24': {s: {} for s in LEGACY24}}
    for name in WIKI:
        try:
            u[name] = fetch_index(name)
        except Exception as e:
            print(f'  ⚠️  {name}: {e} — keeping previous list')
            u[name] = old.get(name, {})
    u['sp500_tech'] = {s: v for s, v in u.get('sp500', {}).items() if v['sector'] == 'Information Technology'}
    with open(UNIVERSES, 'w') as f:
        json.dump(u, f, indent=1)
    return u


def load_universes():
    try:
        with open(UNIVERSES) as f:
            return json.load(f)
    except Exception:
        return {}


def all_symbols(u):
    syms = set(BENCHMARKS)
    for k, v in u.items():
        if isinstance(v, dict):
            syms |= set(v)
    return sorted(syms)


def download(symbols, start=START, period=None, chunk=100):
    """Daily OHLCV as {field: DataFrame(dates x symbols)}, split/dividend adjusted."""
    frames = []
    for i in range(0, len(symbols), chunk):
        part = symbols[i:i + chunk]
        kw = {'period': period} if period else {'start': start}
        df = yf.download(part, interval='1d', auto_adjust=True, progress=False, threads=True,
                         group_by='column', **kw)
        if df.empty:
            continue
        if not isinstance(df.columns, pd.MultiIndex):   # single symbol
            df.columns = pd.MultiIndex.from_product([df.columns, part])
        frames.append(df)
    data = pd.concat(frames, axis=1)
    out = {}
    for field in ('Open', 'High', 'Low', 'Close', 'Volume'):
        f = data[field]
        f = f.loc[:, ~f.columns.duplicated()]
        out[field] = f.astype('float32')
    out['Close'] = out['Close'].dropna(how='all')
    idx = out['Close'].index
    for field in out:
        out[field] = out[field].reindex(idx)
    return out


def refresh_prices(u=None):
    u = u or load_universes()
    syms = all_symbols(u)
    print(f'  📥 Downloading {len(syms)} symbols since {START}...')
    data = download(syms)
    missing = [s for s in syms if s not in data['Close'].columns or data['Close'][s].isna().all()]
    with open(PRICE_CACHE, 'wb') as f:
        pickle.dump(data, f)
    print(f"  ✅ {data['Close'].shape[1] - len(missing)} symbols, {len(data['Close'])} days "
          f"to {data['Close'].index[-1]:%Y-%m-%d}; {len(missing)} without data")
    return data


def load_prices():
    with open(PRICE_CACHE, 'rb') as f:
        return pickle.load(f)


def universe_mask(u, name, close):
    """Boolean DataFrame (dates x symbols): which symbols are eligible on each date."""
    members = u.get(name, {})
    cols = [s for s in members if s in close.columns]
    mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    for s in cols:
        added = members[s].get('added')
        mask.loc[mask.index >= pd.Timestamp(added) if added else slice(None), s] = True
    return mask & close.notna()


if __name__ == '__main__':
    u = refresh_universes()
    print({k: len(v) for k, v in u.items() if isinstance(v, dict)})
    refresh_prices(u)
