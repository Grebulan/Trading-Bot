#!/usr/bin/env python3
"""
Trades the lab's champion strategy on the paper account (2026-11-08). Runs 15:30 ET on weekdays.

champion.json names the champion (lab.py promotes them):
- 'legacy_ml': the original bot. trader.py trades; this script does nothing.
- 'spy_hold':  everything in SPY (the fallback after a demotion).
- otherwise:   a lab strategy. Its target weights come from lab_engine.Lab.weights_at, the same code
               that backtested and shadow-traded it, using prices downloaded now (today's partial
               bar stands in for the close, 30 minutes early).

Every run checks trailing stops; on rebalance days (every rebalance_days trading days) the whole
stock book is reset to the targets: anything else, including old ML positions and SPY, is sold.
Crypto positions are left to crypto_trader.py. Gross exposure is capped at MAX_GROSS_LIVE (Reg-T
overnight limit is 2x). Longs use fractional notional orders; shorts must be whole shares. A position
that flips between long and short is closed first and reopened on the next run.

  python3 champion_trader.py --dry-run   # show the orders without placing them
"""
import json
import sys
from datetime import datetime

import numpy as np
import alpaca_trade_api as tradeapi

import lab_data
from lab_engine import Lab, MAX_GROSS_LIVE
from auto_adjust import load_env

HOME          = '/home/jv'
CHAMPION_FILE = f'{HOME}/champion.json'
STATE_FILE    = f'{HOME}/champion_state.json'
MIN_TRADE     = 100     # $ — skip smaller adjustments
MIN_TRADE_PCT = 0.05    # ...or smaller than this share of the target


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def business_days_between(a, b):
    return int(np.busday_count(a, b))


def main():
    dry_run = '--dry-run' in sys.argv
    print(f"👑 Champion trader — {datetime.now():%Y-%m-%d %H:%M}{'  (DRY RUN)' if dry_run else ''}")
    champ = read_json(CHAMPION_FILE, {'id': 'legacy_ml'})
    if champ['id'] == 'legacy_ml':
        print('  Champion is the original ML bot — trader.py trades, nothing to do')
        return

    env = load_env()
    api = tradeapi.REST(env['ALPACA_KEY'], env['ALPACA_SECRET'], 'https://paper-api.alpaca.markets',
                        api_version='v2')
    if not api.get_clock().is_open and not dry_run:
        print('  Market closed — nothing to do')
        return
    acct = api.get_account()
    equity = float(acct.equity)
    state = read_json(STATE_FILE, {})
    if state.get('champion') != champ['id']:          # new champion: rebalance now, fresh stop marks
        state = {'champion': champ['id'], 'last_rebalance': None, 'marks': {}}
    today = datetime.now().strftime('%Y-%m-%d')

    for o in api.list_orders(status='open'):
        if o.asset_class == 'us_equity':
            if not dry_run:
                api.cancel_order(o.id)
            print(f'  🗑️  Cancelled open order {o.side} {o.symbol}')

    positions = {p.symbol: p for p in api.list_positions() if p.asset_class == 'us_equity'}
    current = {s: float(p.market_value) for s, p in positions.items()}   # shorts are negative
    prices = {s: float(p.current_price) for s, p in positions.items()}

    cfg = champ.get('cfg')
    due = (state['last_rebalance'] is None or
           (cfg and business_days_between(state['last_rebalance'], today) >= cfg['rebalance_days']))
    if champ['id'] == 'spy_hold':
        targets = {'SPY': 0.98 * equity} if due else dict(current)
        prices.setdefault('SPY', float(api.get_latest_trade('SPY').price))
    else:
        u = lab_data.load_universes()
        syms = sorted(set(u.get(cfg['universe'], {})) | {'SPY'})
        print(f"  📥 Prices for {len(syms)} symbols ({cfg['universe']})")
        lab = Lab(lab_data.download(syms, period='2y'), u)
        last = lab.close.iloc[-1]
        prices.update({lab_data.alpaca_symbol(s): float(v) for s, v in last.items() if np.isfinite(v)})
        if due:
            w = lab.weights_at(cfg, len(lab.close) - 1)
            gross = np.abs(w).sum()
            if gross > MAX_GROSS_LIVE:
                w *= MAX_GROSS_LIVE / gross
            targets = {lab_data.alpaca_symbol(s): float(x) * equity
                       for s, x in zip(lab.close.columns, w) if x != 0}
            print(f"  🔄 Rebalance: {len(targets)} positions, gross {np.abs(w).sum():.2f}x of ${equity:,.0f}")
        else:
            targets = dict(current)
            print(f"  Holding (next rebalance after {cfg['rebalance_days']} trading days from {state['last_rebalance']})")

        # Trailing stops on what we hold (the backtest does the same at each close)
        stop = cfg.get('trail_stop')
        for s, v in current.items():
            px = prices.get(s)
            if not stop or px is None:
                continue
            mark = state['marks'].get(s, px)
            mark = max(mark, px) if v > 0 else min(mark, px)
            state['marks'][s] = mark
            if (v > 0 and px <= mark * (1 - stop)) or (v < 0 and px >= mark * (1 + stop)):
                print(f'  🛑 Trailing stop {s}: {px:.2f} vs mark {mark:.2f}')
                targets[s] = 0.0

    # Orders: reductions first (frees buying power), then increases
    orders = []
    for s in sorted(set(targets) | set(current)):
        cur, tgt = current.get(s, 0.0), targets.get(s, 0.0)
        diff = tgt - cur
        if tgt == 0 and cur != 0:
            orders.append((0, s, 'close', cur))
        elif cur * tgt < 0:
            orders.append((0, s, 'close', cur))             # flip: close now, reopen next run
        elif abs(diff) >= max(MIN_TRADE, MIN_TRADE_PCT * abs(tgt)):
            orders.append((0 if abs(tgt) < abs(cur) else 1, s, 'adjust', diff))
    placed = 0
    for _, s, kind, amount in sorted(orders):
        px = prices.get(s)
        try:
            if kind == 'close':
                print(f'  ✖️  Close {s} (${amount:,.0f})')
                if not dry_run:
                    api.close_position(s)
                state['marks'].pop(s, None)
            else:
                tgt = targets[s]
                side = 'buy' if amount > 0 else 'sell'
                if tgt > 0:            # long: fractional
                    asset = api.get_asset(s)
                    if asset.fractionable:
                        if side == 'buy':
                            kw = {'notional': round(amount, 2)}
                        else:
                            kw = {'qty': round(min(-amount / px, float(positions[s].qty)), 6)}
                    else:
                        kw = {'qty': int(abs(amount) / px)}
                else:                  # short: whole shares, easy-to-borrow only when opening/adding
                    if side == 'sell':
                        asset = api.get_asset(s)
                        if not (asset.shortable and asset.easy_to_borrow):
                            print(f'  ⏸️  {s} not easy to borrow — skipping short')
                            continue
                    kw = {'qty': int(abs(amount) / px)}
                if kw.get('qty', 1) <= 0:
                    continue
                print(f"  {'🟢' if side == 'buy' else '🔴'} {side.upper()} {s} {kw}  (target ${tgt:,.0f}, now ${current.get(s, 0):,.0f})")
                if not dry_run:
                    api.submit_order(symbol=s, side=side, type='market', time_in_force='day', **kw)
                if s not in current:
                    state['marks'][s] = px
            placed += 1
        except Exception as e:
            print(f'  ERROR {kind} {s}: {e}')
    if due and not dry_run:
        state['last_rebalance'] = today
    print(f'  {placed} orders {"(dry run)" if dry_run else "placed"}')
    if not dry_run:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=1)


if __name__ == '__main__':
    main()
