#!/usr/bin/env python3
"""
Weekly milestone recorder — runs every Sunday after the retrain.
Saves a portfolio snapshot to milestones.json for long-term review.
"""

import os
import json
import math
import alpaca_trade_api as tradeapi
from datetime import datetime

MILESTONES_FILE  = '/home/jv/milestones.json'
MODEL_INFO_FILE  = '/home/jv/model_info.json'
CRYPTO_INFO_FILE = '/home/jv/crypto_model_info.json'

RISK_FREE_ANNUAL = 0.05   # ~5% US Treasury rate
WEEKS_PER_YEAR   = 52


def calc_sharpe(milestones):
    """Weekly Sharpe ratio from milestone portfolio values. Needs ≥3 points."""
    values = [m['portfolio_value'] for m in milestones if 'portfolio_value' in m]
    if len(values) < 3:
        return None
    weekly_returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
    n = len(weekly_returns)
    mean_r = sum(weekly_returns) / n
    variance = sum((r - mean_r) ** 2 for r in weekly_returns) / (n - 1)
    std_r = math.sqrt(variance)
    if std_r == 0:
        return None
    risk_free_weekly = RISK_FREE_ANNUAL / WEEKS_PER_YEAR
    sharpe = (mean_r - risk_free_weekly) / std_r * math.sqrt(WEEKS_PER_YEAR)
    return round(sharpe, 2)


def load_env():
    env = {}
    with open(os.path.expanduser('~/.env_trading')) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                env[k] = v
    return env


def run():
    ENV = load_env()
    api = tradeapi.REST(ENV['ALPACA_KEY'], ENV['ALPACA_SECRET'],
                        'https://paper-api.alpaca.markets', api_version='v2')

    acc       = api.get_account()
    positions = api.list_positions()
    port_val  = round(float(acc.portfolio_value), 2)
    cash      = round(float(acc.cash), 2)
    total_ret = round((port_val - 100000) / 100000 * 100, 2)

    stock_info  = {}
    crypto_info = {}
    if os.path.exists(MODEL_INFO_FILE):
        with open(MODEL_INFO_FILE) as f:
            stock_info = json.load(f)
    if os.path.exists(CRYPTO_INFO_FILE):
        with open(CRYPTO_INFO_FILE) as f:
            crypto_info = json.load(f)

    milestones = []
    if os.path.exists(MILESTONES_FILE):
        with open(MILESTONES_FILE) as f:
            milestones = json.load(f)

    milestone = {
        'date':             datetime.now().strftime('%Y-%m-%d'),
        'time':             datetime.now().strftime('%H:%M'),
        'portfolio_value':  port_val,
        'cash':             cash,
        'total_return_pct': total_ret,
        'positions': [
            {
                'symbol': p.symbol,
                'qty':    round(float(p.qty), 6),
                'value':  round(float(p.market_value), 2),
                'pl_pct': round(float(p.unrealized_plpc) * 100, 2),
            }
            for p in sorted(positions, key=lambda p: float(p.market_value), reverse=True)
        ],
        'model': {
            'stock_accuracy':  stock_info.get('accuracy', '—'),
            'crypto_accuracy': crypto_info.get('accuracy', '—'),
            'trained_at':      stock_info.get('trained_at', '—'),
        },
    }

    milestones.append(milestone)

    sharpe = calc_sharpe(milestones)
    milestone['sharpe'] = sharpe

    with open(MILESTONES_FILE, 'w') as f:
        json.dump(milestones, f, indent=2)

    sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "need ≥3 milestones"
    print(f"✅ Milestone recorded — {milestone['date']}")
    print(f"   Portfolio: ${port_val:,.2f}  ({total_ret:+.2f}% total return)")
    print(f"   Sharpe ratio: {sharpe_str}")
    print(f"   Positions: {len(positions)}  |  Cash: ${cash:,.2f}")
    print(f"   Model accuracy — Stock: {milestone['model']['stock_accuracy']}  Crypto: {milestone['model']['crypto_accuracy']}")


if __name__ == '__main__':
    run()
