#!/usr/bin/env python3
"""
Strategy lab: weekly champion-vs-challengers cycle (Sunday 03:00, 2026-11-08).

1. Refresh universes and prices.
2. Generate new candidate strategies from the MENU (lab_engine.py). Some are mutations of the best
   strategies so far; the rest are sampled with weights learned from past results (the lessons).
3. Backtest each on an in-sample period (2015-2022) and an out-of-sample period (2023-now), with
   random picks from the same universe as the baseline. Candidates that beat SPY and their random
   baseline in both become challengers.
4. Challengers shadow-trade: their record from the day they were accepted onwards uses only data
   that didn't exist when they were chosen, so it is an honest forward test.
5. Promotion: a challenger with 8+ weeks of forward record that beats SPY and beats the real paper
   account over the same weeks by PROMOTE_MARGIN becomes the champion (traded by champion_trader.py).
   At most one promotion per PROMOTE_GAP_DAYS. A champion that falls 25% behind SPY is replaced by
   plain SPY until a challenger earns the slot.
6. Retirement: challengers that lag badly, or that never win within 16 weeks, are retired. Every
   promotion and retirement is written to the lessons, which steer what gets generated next.

Every promotion/demotion is committed, pushed and emailed.

  python3 lab.py               # full Sunday run
  python3 lab.py --no-refresh  # reuse cached prices
  python3 lab.py --dry-run     # test and report, change nothing
  python3 lab.py --candidates 150   # a wider search than the weekly CANDIDATES_PER_WEEK
"""
import json
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd

import lab_data
from lab_engine import Lab, MENU, metrics, config_id, normalise

HOME          = '/home/jv'
STATE_FILE    = f'{HOME}/lab_state.json'
CHAMPION_FILE = f'{HOME}/champion.json'

IS_START, IS_END, OOS_START = '2015-01-02', '2022-12-30', '2023-01-03'
CANDIDATES_PER_WEEK = 24
MUTATIONS_PER_WEEK  = 8
MAX_NEW_CHALLENGERS = 3
MAX_CHALLENGERS     = 10
RANDOM_SEEDS        = (11, 12)
MIN_EDGE_OVER_RANDOM = 0.02    # CAGR points above random picks from the same universe

PROMOTE_MIN_DAYS    = 40       # trading days (~8 weeks) of forward record
PROMOTE_MARGIN      = 0.02     # must beat the account's return over the same window by this
PROMOTE_GAP_DAYS    = 28
RETIRE_AFTER_DAYS   = 80       # ~16 weeks without promotion
RETIRE_EARLY_DAYS   = 30
RETIRE_EARLY_BEHIND = -0.10    # forward return this far behind SPY
DEMOTE_BEHIND_SPY   = -0.25    # champion this far behind SPY since it started (after 8 weeks)


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1, default=str)


def default_champion():
    return {'id': 'legacy_ml', 'cfg': None, 'since': '2026-03-06',
            'note': 'original ML stock picker in trader.py (stock buys paused 2026-11-08)'}


# ------------------------------------------------------------
# Lessons: what has worked, per menu choice
# ------------------------------------------------------------
def block_scores(state):
    """Average excess CAGR vs SPY for each menu choice. In-sample backtests count once; forward
    (shadow) results count 3x, because they are the honest evidence."""
    sums = {k: {} for k in MENU}
    def add(cfg, value, weight):
        for k in MENU:
            key = str(cfg[k])
            s, w = sums[k].get(key, (0.0, 0.0))
            sums[k][key] = (s + value * weight, w + weight)
    for rec in state['tested'].values():
        if rec.get('is'):
            add(rec['cfg'], rec['is']['excess_cagr'], 1.0)
    for ch in state['challengers'] + state['retired']:
        fwd = ch.get('forward') or {}
        if fwd.get('days', 0) >= 20:
            add(ch['cfg'], fwd['vs_spy'] * 252 / fwd['days'], 3.0)   # annualised
    return {k: {v: round(s / w, 4) for v, (s, w) in d.items() if w} for k, d in sums.items()}


def sample_config(scores, rng, temperature=0.05, explore=0.3):
    cfg = {}
    for k, options in MENU.items():
        if rng.random() < explore or not scores.get(k):
            cfg[k] = rng.choice(options)
            continue
        vals = np.array([scores[k].get(str(o), 0.0) for o in options])
        p = np.exp((vals - vals.max()) / temperature)
        cfg[k] = options[rng.choices(range(len(options)), weights=p)[0]]
    return cfg


def mutate(cfg, rng):
    cfg = dict(cfg)
    k = rng.choice(list(MENU))
    cfg[k] = rng.choice([o for o in MENU[k] if o != cfg[k]])
    return cfg


def generate(state, champion, u, rng, n=CANDIDATES_PER_WEEK):
    scores = block_scores(state)
    parents = [c['cfg'] for c in state['challengers']]
    if champion.get('cfg'):
        parents.append(champion['cfg'])
    best_tested = sorted((r for r in state['tested'].values() if r.get('oos')),
                         key=lambda r: r['oos']['sharpe'], reverse=True)[:5]
    parents += [r['cfg'] for r in best_tested]
    out, seen = [], set(state['tested'])
    tries = 0
    while len(out) < n and tries < 20 * n:
        tries += 1
        if parents and len(out) < MUTATIONS_PER_WEEK:
            cfg = mutate(rng.choice(parents), rng)
        else:
            cfg = sample_config(scores, rng)
        cfg = normalise(cfg, u)
        cid = config_id(cfg)
        if cid not in seen:
            seen.add(cid)
            out.append(cfg)
    return out


# ------------------------------------------------------------
# Testing
# ------------------------------------------------------------
def random_baseline(lab, cfg, start, end, cache):
    key = (config_id({**cfg, 'signal': 'random'}), start)
    if key not in cache:
        cagrs = []
        for seed in RANDOM_SEEDS:
            m = metrics(lab.backtest(cfg, start, end, random_seed=seed), lab.spy_curve(start, end))
            cagrs.append(m.get('cagr', -1.0))
        cache[key] = float(np.mean(cagrs))
    return cache[key]


def evaluate(lab, cfg, cache):
    rec = {'cfg': cfg, 'tested': datetime.now().strftime('%Y-%m-%d')}
    rec['is'] = metrics(lab.backtest(cfg, IS_START, IS_END), lab.spy_curve(IS_START, IS_END))
    rec['oos'] = metrics(lab.backtest(cfg, OOS_START), lab.spy_curve(OOS_START))
    if not rec['is'] or not rec['oos']:
        rec['passed'] = False
        return rec
    rec['random_is_cagr'] = round(random_baseline(lab, cfg, IS_START, IS_END, cache), 4)
    rec['random_oos_cagr'] = round(random_baseline(lab, cfg, OOS_START, None, cache), 4)
    rec['checks'] = {
        'beats SPY in-sample':        rec['is']['excess_cagr'] > 0,
        'beats SPY out-of-sample':    rec['oos']['excess_cagr'] > 0,
        'beats random picks in-sample':
            rec['is']['cagr'] >= rec['random_is_cagr'] + MIN_EDGE_OVER_RANDOM,
        'beats random picks out-of-sample':
            rec['oos']['cagr'] >= rec['random_oos_cagr'] + MIN_EDGE_OVER_RANDOM,
        'not hindsight-biased universe': cfg['universe'] != 'legacy24',
    }
    rec['passed'] = all(rec['checks'].values())
    return rec


def account_curve(env):
    """The paper account's daily equity (the real champion record)."""
    import alpaca_trade_api as tradeapi
    api = tradeapi.REST(env['ALPACA_KEY'], env['ALPACA_SECRET'], 'https://paper-api.alpaca.markets',
                        api_version='v2')
    h = api.get_portfolio_history(period='1A', timeframe='1D')
    s = pd.Series(h.equity, index=pd.to_datetime(h.timestamp, unit='s').normalize(), dtype=float)
    return s[s > 0]


def window_return(curve, start):
    s = curve[curve.index >= pd.Timestamp(start)]
    return float(s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 2 else 0.0


def update_forward(lab, ch, acct):
    start = ch['forward_start']
    curve = lab.backtest(ch['cfg'], start)
    if len(curve) < 2:
        ch['forward'] = {'days': 0}
        return
    spy = lab.spy_curve(start)
    total = float(curve.iloc[-1] / curve.iloc[0] - 1)
    spy_total = float(spy.iloc[-1] / spy.iloc[0] - 1)
    ch['forward'] = {
        'days': len(curve) - 1, 'return': round(total, 4), 'spy_return': round(spy_total, 4),
        'vs_spy': round(total - spy_total, 4),
        'account_return': round(window_return(acct, start), 4) if acct is not None else None,
        'max_drawdown': round(float((curve / curve.cummax() - 1).min()), 4),
        'as_of': f'{curve.index[-1]:%Y-%m-%d}',
    }


# ------------------------------------------------------------
def main():
    dry_run = '--dry-run' in sys.argv
    print(f"🧪 Strategy lab — {datetime.now():%Y-%m-%d %H:%M}{'  (DRY RUN)' if dry_run else ''}")
    today = datetime.now().strftime('%Y-%m-%d')
    state = read_json(STATE_FILE, {'challengers': [], 'retired': [], 'tested': {}, 'events': []})
    champion = read_json(CHAMPION_FILE, None) or default_champion()

    if '--no-refresh' not in sys.argv:
        u = lab_data.refresh_universes()
        data = lab_data.refresh_prices(u)
    else:
        u, data = lab_data.load_universes(), lab_data.load_prices()
    lab = Lab(data, u)
    last_day = lab.close.index[-1]
    next_day = (last_day + pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    rng = random.Random(today)

    from auto_adjust import load_env, send_email, git_commit_push
    env = load_env()
    try:
        acct = account_curve(env)
    except Exception as e:
        print(f'  ⚠️  Could not read account history: {e}')
        acct = None

    # 1. Forward records
    for ch in state['challengers']:
        update_forward(lab, ch, acct)

    # 2. New candidates
    n = int(sys.argv[sys.argv.index('--candidates') + 1]) if '--candidates' in sys.argv else CANDIDATES_PER_WEEK
    cands = generate(state, champion, u, rng, n)
    print(f'  🔬 Testing {len(cands)} candidates...')
    cache, passed = {}, []
    for cfg in cands:
        rec = evaluate(lab, cfg, cache)
        state['tested'][config_id(cfg)] = rec
        if rec['passed']:
            passed.append(rec)
    print(f'  {len(passed)} passed all checks')
    room = MAX_CHALLENGERS - len(state['challengers'])
    for rec in sorted(passed, key=lambda r: r['oos']['sharpe'], reverse=True)[:min(MAX_NEW_CHALLENGERS, max(room, 0))]:
        ch = {'id': config_id(rec['cfg']), 'cfg': rec['cfg'], 'accepted': today, 'forward_start': next_day,
              'backtest': {'is': rec['is'], 'oos': rec['oos'], 'random_is_cagr': rec['random_is_cagr'],
                           'random_oos_cagr': rec['random_oos_cagr']}, 'forward': {'days': 0}}
        state['challengers'].append(ch)
        state['events'].append({'date': today, 'event': 'new_challenger', 'id': ch['id']})
        print(f"  ➕ Challenger: {ch['id']}  (OOS CAGR {rec['oos']['cagr']:+.1%}, Sharpe {rec['oos']['sharpe']})")

    # 3. Retirement
    keep = []
    for ch in state['challengers']:
        f = ch.get('forward') or {}
        reason = None
        if f.get('days', 0) >= RETIRE_AFTER_DAYS:
            reason = f"no promotion after {f['days']} trading days"
        elif f.get('days', 0) >= RETIRE_EARLY_DAYS and f['vs_spy'] <= RETIRE_EARLY_BEHIND:
            reason = f"{f['vs_spy']:+.1%} behind SPY after {f['days']} trading days"
        if reason:
            ch.update(status='retired', retired=today, verdict=reason)
            state['retired'].append(ch)
            state['events'].append({'date': today, 'event': 'retired', 'id': ch['id'], 'reason': reason})
            print(f"  ➖ Retired: {ch['id']} — {reason}")
        else:
            keep.append(ch)
    state['challengers'] = keep

    # 4. Champion check: demote a failing lab champion to plain SPY
    change = None
    if champion['id'] not in ('legacy_ml', 'spy_hold') and acct is not None:
        days = int(((acct.index >= pd.Timestamp(champion['since']))).sum())
        champ_vs_spy = window_return(acct, champion['since']) - window_return(lab.close['SPY'].dropna(), champion['since'])
        if days >= PROMOTE_MIN_DAYS and champ_vs_spy <= DEMOTE_BEHIND_SPY:
            change = ({'id': 'spy_hold', 'cfg': None, 'since': next_day,
                       'note': f"demoted {champion['id']}: {champ_vs_spy:+.1%} vs SPY"},
                      f"Champion demoted to plain SPY: it was {champ_vs_spy:+.1%} behind SPY since {champion['since']}.")

    # 5. Promotion
    last_promo = max((e['date'] for e in state['events'] if e['event'] in ('promoted', 'demoted')), default=None)
    gap_ok = not last_promo or (datetime.now() - datetime.fromisoformat(last_promo)).days >= PROMOTE_GAP_DAYS
    eligible = [c for c in state['challengers']
                if c['forward'].get('days', 0) >= PROMOTE_MIN_DAYS and c['forward']['vs_spy'] > 0
                and c['forward'].get('account_return') is not None
                and c['forward']['return'] >= c['forward']['account_return'] + PROMOTE_MARGIN]
    if not change and gap_ok and eligible:
        best = max(eligible, key=lambda c: c['forward']['return'])
        f = best['forward']
        change = ({'id': best['id'], 'cfg': best['cfg'], 'since': next_day, 'last_rebalance': None,
                   'note': f"promoted: forward {f['return']:+.1%} vs account {f['account_return']:+.1%}, "
                           f"SPY {f['spy_return']:+.1%} over {f['days']} trading days"},
                  f"New champion {best['id']}. Over {f['days']} trading days of shadow trading it made "
                  f"{f['return']:+.1%}, against {f['account_return']:+.1%} for the account and {f['spy_return']:+.1%} for SPY.")
        best.update(status='champion', promoted=today)
        state['challengers'] = [c for c in state['challengers'] if c is not best]
        state['retired'].append(best)

    if change and not dry_run:
        new, message = change
        old = champion
        new['previous'] = {'id': old['id'], 'since': old['since']}
        state['events'].append({'date': today, 'event': 'demoted' if new['id'] == 'spy_hold' else 'promoted',
                                'id': new['id'], 'replaced': old['id'], 'message': message})
        write_json(CHAMPION_FILE, new)
        print(f'  👑 {message}')
        sha, err = git_commit_push(f'lab: {message}', paths=['champion.json'])
        if err:
            print(f'  ERROR {err}')
        try:
            send_email(env, 'Trading bot: new champion strategy',
                       f"{message}\n\nPrevious champion: {old['id']} (since {old['since']}).\n"
                       f"New settings: {json.dumps(new.get('cfg'), indent=1)}\n\n"
                       f"champion_trader.py will start trading it on the next run. To undo, edit champion.json "
                       f"(set id to 'legacy_ml' to go back to the original bot).\nCommit: {sha or '-'}\n")
        except Exception as e:
            print(f'  ERROR sending email: {e}')
    elif change:
        print(f'  👑 would change champion: {change[1]}')

    state['lessons'] = block_scores(state)
    state['last_run'] = {'date': today, 'data_to': f'{last_day:%Y-%m-%d}', 'tested': len(cands),
                         'passed': len(passed), 'challengers': len(state['challengers'])}
    if not dry_run:
        write_json(STATE_FILE, state)
        if not read_json(CHAMPION_FILE, None):
            write_json(CHAMPION_FILE, champion)
    print(f"  Champion: {champion['id'] if not change or dry_run else change[0]['id']}; "
          f"{len(state['challengers'])} challengers shadow-trading")
    for ch in state['challengers']:
        f = ch['forward']
        if f.get('days'):
            print(f"    {ch['id']}: {f['days']}d, {f['return']:+.1%} (SPY {f['spy_return']:+.1%})")


if __name__ == '__main__':
    main()
