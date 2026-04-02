#!/usr/bin/env python3
"""
Records portfolio value snapshot including stock/crypto breakdown.
Run via crontab several times per day.
"""

import os
import json
from datetime import datetime
import alpaca_trade_api as tradeapi

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

HISTORY_FILE = '/home/jv/portfolio_history.json'

def main():
    account   = api.get_account()
    total     = round(float(account.portfolio_value), 2)
    cash      = round(float(account.cash), 2)
    positions = api.list_positions()

    stock_value  = 0.0
    crypto_value = 0.0
    for p in positions:
        val = float(p.market_value)
        if p.asset_class == 'crypto':
            crypto_value += val
        else:
            stock_value += val

    stock_value  = round(stock_value, 2)
    crypto_value = round(crypto_value, 2)

    entry = {
        'time':         datetime.now().strftime('%Y-%m-%dT%H:%M'),
        'value':        total,
        'cash':         cash,
        'stock_value':  stock_value,
        'crypto_value': crypto_value,
    }

    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            history = json.load(f)

    history.append(entry)

    # Keep last 500 entries
    if len(history) > 500:
        history = history[-500:]

    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"✅ Recorded: total=${total:,.2f} | stocks=${stock_value:,.2f} | crypto=${crypto_value:,.2f} | cash=${cash:,.2f}")

if __name__ == '__main__':
    main()
