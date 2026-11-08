#!/usr/bin/env python3
"""
ML stock trades vs SPY — did each pick beat leaving the same money in SPY?

Since 2026-10-04 every ML stock buy is funded by selling SPY, so the fair comparison
for each trade is what that money would have made staying in SPY over the same period.
For each trade: stock return (entry -> exit, or -> now if still open) vs SPY return
over the same window, and the dollar difference.

Run on its own for a table, or import tally() (weekly_review.py does).
"""
import json
import os
from datetime import datetime, timezone

import alpaca_trade_api as tradeapi

SINCE = '2026-10-04'   # SPY-funded ML buys began (fund_from_spy)
SIGNAL_LOG = '/home/jv/performance.json'   # trader.py's log_signal() entries (local time)


def load_signals():
    """Stock signal log entries with UTC-aware times, so they line up with Alpaca fill times."""
    try:
        data = json.load(open(SIGNAL_LOG))
    except Exception:
        return []
    out = []
    for e in data:
        if e.get('asset_type') == 'stock':
            out.append({**e, 'when': datetime.fromisoformat(e['time']).astimezone(timezone.utc)})
    return out


def exit_details(signals, symbol, entry_time, exit_time):
    """Why a trade exited (the acted SELL logged closest to the fill) and how often the stop
    override held it while it was open."""
    end = exit_time or datetime.now(timezone.utc)
    held = [e for e in signals if e['symbol'] == symbol and entry_time <= e['when'] <= end]
    overrides = sum(e['reason'] == 'stop_override' for e in held)
    reason = None
    if exit_time:
        sells = [e for e in signals if e['symbol'] == symbol and e['signal'] == 'SELL' and e['acted']
                 and abs((e['when'] - exit_time).total_seconds()) < 6 * 3600]
        if sells:
            reason = min(sells, key=lambda e: abs((e['when'] - exit_time).total_seconds()))['reason']
        else:
            reason = 'unknown (no matching bot sell — manual or broker?)'
    return reason, overrides


def load_env():
    env = {}
    with open(os.path.expanduser('~/.env_trading')) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k] = v
    return env


def spy_price_at(bars, when):
    """SPY price at a moment: open of the 15-min bar containing it (else the last bar before)."""
    earlier = bars[bars.index <= when]
    if earlier.empty:
        return float(bars['open'].iloc[0])
    return float(earlier['open'].iloc[-1]) if (when - earlier.index[-1]).total_seconds() < 900 \
        else float(earlier['close'].iloc[-1])


def tally(api, since=SINCE):
    orders = api.list_orders(status='closed', after=f'{since}T00:00:00Z', limit=500, direction='asc')
    fills = [o for o in orders if o.filled_at and o.asset_class == 'us_equity' and o.symbol != 'SPY']
    if not fills:
        return {'since': since, 'trades': [], 'note': 'no ML stock trades yet'}

    bars = api.get_bars('SPY', '15Min', since, feed='iex').df
    spy_now = float(api.get_latest_trade('SPY', feed='iex').price)
    positions = {p.symbol: p for p in api.list_positions()}
    signals = load_signals()

    # Pair each buy with the sell that closes it (the bot always buys and sells a whole position)
    open_trades, trades = {}, []
    for o in fills:
        t, px, qty = o.filled_at, float(o.filled_avg_price), float(o.filled_qty)
        if o.side == 'buy':
            open_trades[o.symbol] = {'symbol': o.symbol, 'entry_time': t, 'entry_price': px, 'qty': qty}
        elif o.symbol in open_trades:   # sells of positions bought before `since` are ignored
            tr = open_trades.pop(o.symbol)
            trades.append({**tr, 'exit_time': t, 'exit_price': px, 'status': 'closed'})
    for sym, tr in open_trades.items():
        pos = positions.get(sym)
        px = float(pos.current_price) if pos else tr['entry_price']
        trades.append({**tr, 'exit_time': None, 'exit_price': px, 'status': 'open'})

    out = []
    for tr in sorted(trades, key=lambda x: x['entry_time']):
        exit_reason, overrides = exit_details(signals, tr['symbol'], tr['entry_time'], tr['exit_time'])
        cost = tr['entry_price'] * tr['qty']
        spy_in = spy_price_at(bars, tr['entry_time'])
        spy_out = spy_price_at(bars, tr['exit_time']) if tr['exit_time'] else spy_now
        stock_pct = (tr['exit_price'] / tr['entry_price'] - 1) * 100
        spy_pct = (spy_out / spy_in - 1) * 100
        out.append({
            'symbol': tr['symbol'], 'status': tr['status'],
            'entry': tr['entry_time'].strftime('%Y-%m-%d'),
            'exit': tr['exit_time'].strftime('%Y-%m-%d') if tr['exit_time'] else None,
            'cost': round(cost, 2),
            'stock_pct': round(stock_pct, 2), 'spy_pct': round(spy_pct, 2),
            'pl_usd': round(cost * stock_pct / 100, 2),
            'vs_spy_usd': round(cost * (stock_pct - spy_pct) / 100, 2),   # + means the pick beat SPY
            'exit_reason': exit_reason,
            'stop_override_holds': overrides,   # hourly runs where the override kept it past the stop
        })

    closed = [t for t in out if t['status'] == 'closed']
    return {
        'since': since,
        'trades': out,
        'summary': {
            'trades': len(out), 'closed': len(closed), 'open': len(out) - len(closed),
            'beat_spy': sum(t['vs_spy_usd'] > 0 for t in out),
            'total_pl_usd': round(sum(t['pl_usd'] for t in out), 2),
            'total_vs_spy_usd': round(sum(t['vs_spy_usd'] for t in out), 2),
            'closed_vs_spy_usd': round(sum(t['vs_spy_usd'] for t in closed), 2),
            'stop_override_holds': sum(t['stop_override_holds'] for t in out),
        },
        'note': 'vs_spy_usd = profit or loss compared with leaving the same money in SPY; '
                'open trades are valued at the current price. exit_reason comes from the bot\'s '
                'signal log (trailing_stop, take_profit, hard_stop); stop_override_holds counts hourly '
                'runs where the stop was hit but the override kept holding',
    }


def main():
    env = load_env()
    api = tradeapi.REST(env['ALPACA_KEY'], env['ALPACA_SECRET'],
                        'https://paper-api.alpaca.markets', api_version='v2')
    r = tally(api)
    print(f"📊 ML stock trades vs SPY since {r['since']}  ({datetime.now():%Y-%m-%d %H:%M})\n")
    print(f"{'Stock':6} {'Status':6} {'Bought':10} {'Sold':10} {'Cost':>9} {'Stock':>7} {'SPY':>7} {'P&L':>9} {'vs SPY':>9}  {'Exit':14} Overrides")
    for t in r['trades']:
        print(f"{t['symbol']:6} {t['status']:6} {t['entry']:10} {t['exit'] or '-':10} ${t['cost']:>8,.0f} "
              f"{t['stock_pct']:>+6.1f}% {t['spy_pct']:>+6.1f}% {t['pl_usd']:>+9,.0f} {t['vs_spy_usd']:>+9,.0f}  "
              f"{(t['exit_reason'] or '-')[:14]:14} {t['stop_override_holds'] or ''}")
    s = r.get('summary')
    if s:
        print(f"\n{s['beat_spy']} of {s['trades']} trades beat SPY.  Total P&L ${s['total_pl_usd']:+,.0f}, "
              f"which is ${s['total_vs_spy_usd']:+,.0f} compared with leaving the money in SPY "
              f"(closed trades only: ${s['closed_vs_spy_usd']:+,.0f}).")


if __name__ == '__main__':
    main()
