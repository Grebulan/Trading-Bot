#!/usr/bin/env python3
"""
Performance logger for trading bot.
Records every signal with outcome tracking.
Run after trader.py and crypto_trader.py to log results.
"""

import os
import json
import yfinance as yf
from datetime import datetime, timedelta

LOG_FILE = '/home/jv/performance.json'

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return []

def save_log(log):
    with open(LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)

def load_env():
    env = {}
    with open(os.path.expanduser('~/.env_trading')) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                env[k] = v
    return env

def log_signals(signals):
    """
    Log a batch of signals.
    signals: list of dicts with keys:
      - symbol, asset_type, signal, confidence,
        sentiment, price, acted, reason
    """
    log     = load_log()
    now     = datetime.now().isoformat()
    for s in signals:
        entry = {
            'id':         f"{s['symbol']}_{now}",
            'timestamp':  now,
            'symbol':     s['symbol'],
            'asset_type': s.get('asset_type', 'stock'),
            'signal':     s['signal'],
            'confidence': s['confidence'],
            'sentiment':  s.get('sentiment', 0.0),
            'price':      s['price'],
            'acted':      s.get('acted', False),
            'reason':     s.get('reason', ''),
            'outcome':    None,   # filled in later
            'outcome_price':  None,
            'outcome_return': None,
            'outcome_correct': None,
        }
        log.append(entry)
    save_log(log)
    print(f"  📝 Logged {len(signals)} signals")

def update_outcomes():
    """
    Check all unresolved signals and fill in outcomes.
    Stock signals resolve after 3 days.
    Crypto signals resolve after 6 hours.
    """
    log     = load_log()
    now     = datetime.now()
    updated = 0

    for entry in log:
        if entry['outcome'] is not None:
            continue

        signal_time = datetime.fromisoformat(entry['timestamp'])
        asset_type  = entry.get('asset_type', 'stock')

        # Resolution window
        if asset_type == 'crypto':
            resolve_after = timedelta(hours=6)
            yf_symbol     = {
                'BTC/USD': 'BTC-USD', 'ETH/USD': 'ETH-USD',
                'SOL/USD': 'SOL-USD', 'LINK/USD': 'LINK-USD',
                'AVAX/USD': 'AVAX-USD', 'XRP/USD': 'XRP-USD',
                'DOGE/USD': 'DOGE-USD', 'UNI/USD': 'UNI7083-USD',
                'DOT/USD': 'DOT-USD', 'ADA/USD': 'ADA-USD',
            }.get(entry['symbol'], entry['symbol'])
        else:
            resolve_after = timedelta(days=3)
            yf_symbol     = entry['symbol']

        if now < signal_time + resolve_after:
            continue  # Too early to resolve

        try:
            if asset_type == 'crypto':
                df = yf.download(yf_symbol, period='2d', interval='1h', progress=False)
            else:
                df = yf.download(yf_symbol, period='10d', interval='1d', progress=False)

            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            if df.empty:
                continue

            entry_price   = float(entry['price'])
            current_price = float(df['Close'].iloc[-1])
            ret           = (current_price - entry_price) / entry_price

            entry['outcome_price']  = round(current_price, 6)
            entry['outcome_return'] = round(ret * 100, 2)

            # Was the signal correct?
            if entry['signal'] == 'BUY':
                entry['outcome_correct'] = ret > 0.01   # Up >1%
            elif entry['signal'] == 'SELL':
                entry['outcome_correct'] = ret < -0.01  # Down >1%
            else:
                entry['outcome_correct'] = None  # HOLD — no judgment

            entry['outcome'] = 'resolved'
            updated += 1

        except Exception as e:
            print(f"  ⚠️  Could not resolve {entry['symbol']}: {e}")

    save_log(log)
    print(f"  ✅ Updated {updated} signal outcomes")

def print_summary():
    """Print performance summary."""
    log = load_log()
    if not log:
        print("No signals logged yet.")
        return

    total     = len(log)
    resolved  = [e for e in log if e['outcome'] == 'resolved']
    acted     = [e for e in log if e['acted']]
    correct   = [e for e in resolved if e['outcome_correct'] is True]
    incorrect = [e for e in resolved if e['outcome_correct'] is False]

    print(f"\n{'='*50}")
    print(f"📊 PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"Total signals logged:  {total}")
    print(f"Signals acted on:      {len(acted)}")
    print(f"Resolved signals:      {len(resolved)}")

    if resolved:
        acc = len(correct) / (len(correct) + len(incorrect)) * 100 if (correct or incorrect) else 0
        print(f"Correct signals:       {len(correct)}")
        print(f"Incorrect signals:     {len(incorrect)}")
        print(f"Signal accuracy:       {acc:.1f}%")

        returns = [e['outcome_return'] for e in resolved if e['outcome_return'] is not None]
        if returns:
            print(f"Avg return on signal:  {sum(returns)/len(returns):+.2f}%")
            print(f"Best signal:           {max(returns):+.2f}%")
            print(f"Worst signal:          {min(returns):+.2f}%")

    # By asset type
    for asset_type in ['stock', 'crypto']:
        subset   = [e for e in resolved if e.get('asset_type') == asset_type]
        correct_ = [e for e in subset if e['outcome_correct'] is True]
        wrong_   = [e for e in subset if e['outcome_correct'] is False]
        if subset:
            acc = len(correct_) / (len(correct_) + len(wrong_)) * 100 if (correct_ or wrong_) else 0
            print(f"\n  {asset_type.upper()} signals: {len(subset)} resolved, {acc:.1f}% accurate")

    # Sentiment filter effectiveness
    acted_correct   = [e for e in resolved if e['acted'] and e['outcome_correct'] is True]
    unacted_correct = [e for e in resolved if not e['acted'] and e['outcome_correct'] is True]
    acted_wrong     = [e for e in resolved if e['acted'] and e['outcome_correct'] is False]
    unacted_wrong   = [e for e in resolved if not e['acted'] and e['outcome_correct'] is False]

    if acted_correct or acted_wrong:
        acted_acc = len(acted_correct) / (len(acted_correct) + len(acted_wrong)) * 100
        print(f"\n  ACTED signals accuracy:     {acted_acc:.1f}%")
    if unacted_correct or unacted_wrong:
        unacted_acc = len(unacted_correct) / (len(unacted_correct) + len(unacted_wrong)) * 100
        print(f"  SKIPPED signals accuracy:   {unacted_acc:.1f}%")
        print(f"  (if skipped > acted, sentiment filter is helping)")

    print(f"{'='*50}\n")

if __name__ == '__main__':
    update_outcomes()
    print_summary()
