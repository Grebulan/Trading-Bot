#!/usr/bin/env python3
"""
Automatic setting changes. Runs Sunday after weekly_review.py ("bounded autonomy", 2026-11-08).

Changes only these trader.py settings, in two ways:
- Risk-reducing, requested by this week's review (its setting_changes): pause stock buys,
  switch off the stop override, cut MAX_POSITIONS by one (not below MIN_POSITIONS). Applied at once.
- Restarting stock buys. Never on the review's say-so: only when every RESTART rule below passes,
  and not within COOLING_OFF_DAYS of the last pause.
Never automatic: re-enabling the stop override or raising MAX_POSITIONS. Those stay the owner's call.

Every change is syntax-checked (reverted on failure), committed and pushed, recorded in
auto_adjust_state.json and emailed with the reason and how to undo it.
Suggest-only (emails what it would do, changes nothing) when AUTONOMY is False or trader.py is not
on the paper-trading API. Switch AUTONOMY off before real money.

  python3 auto_adjust.py            # normal Sunday run
  python3 auto_adjust.py --dry-run  # show what it would do; changes, commits and emails nothing
"""
import json
import os
import py_compile
import re
import smtplib
import subprocess
import sys
from datetime import datetime, timedelta
from email.mime.text import MIMEText

import numpy as np
import yfinance as yf

from retrain import simulate_live_exit, LIVE_MAX_HOLD

HOME          = '/home/jv'
TRADER        = f'{HOME}/trader.py'
STATE_FILE    = f'{HOME}/auto_adjust_state.json'
REVIEWS_FILE  = f'{HOME}/weekly_reviews.json'
SIGNALS_FILE  = f'{HOME}/performance.json'
MODEL_INFO    = f'{HOME}/model_info.json'

AUTONOMY         = True
MIN_POSITIONS    = 3
COOLING_OFF_DAYS = 28            # no loosening within 4 weeks of the last change to that setting
FIRST_PAUSE      = '2026-11-08'  # stock buys paused by hand, before this script existed

# Restart stock buys only when all of these hold
RESTART_MIN_WEEKS_PAUSED = 8     # weeks of forward-test signals since the pause
RESTART_MIN_TRADES       = 20    # hypothetical trades from 'buys_paused' signals
RESTART_MIN_CLOSED       = 10    # ...of which finished
# plus: forward trades ahead of SPY in total, retrain's live-rule test says has_edge,
# and this week's review is not red


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def load_env():
    env = {}
    with open(os.path.expanduser('~/.env_trading')) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k] = v
    return env


# ------------------------------------------------------------
# trader.py settings
# ------------------------------------------------------------
SETTING_RE = r'^({name}\s*=\s*)([^#\n]*?)(\s*(?:#.*)?)$'


def get_setting(src, name):
    m = re.search(SETTING_RE.format(name=name), src, re.M)
    return m.group(2).strip() if m else None


def set_setting(name, value, dry_run):
    """Replace one setting's value in trader.py, keeping its comment. Reverts if trader.py no longer compiles."""
    src = open(TRADER).read()
    pattern = SETTING_RE.format(name=name)
    if len(re.findall(pattern, src, re.M)) != 1:
        raise RuntimeError(f'{name} not found exactly once in trader.py')
    new_src = re.sub(pattern, lambda m: f'{m.group(1)}{value}{m.group(3)}', src, count=1, flags=re.M)
    if dry_run:
        return
    with open(TRADER, 'w') as f:
        f.write(new_src)
    try:
        py_compile.compile(TRADER, doraise=True)
    except py_compile.PyCompileError:
        with open(TRADER, 'w') as f:
            f.write(src)
        raise RuntimeError('trader.py failed to compile after the change; reverted')


def on_paper_api(src):
    m = re.search(r'^BASE_URL\s*=\s*["\']([^"\']+)', src, re.M)
    return bool(m and 'paper-api.alpaca.markets' in m.group(1))


# ------------------------------------------------------------
# Forward test of the buys the bot would have made while paused
# ------------------------------------------------------------
def forward_test(since):
    """Replay 'buys_paused' signals as trades with trader.py's exits (one position per ticker),
    each compared with holding SPY over the same days. Entries use the logged intraday price; SPY is
    measured close-to-close, so the comparison is approximate."""
    sigs = [e for e in read_json(SIGNALS_FILE, [])
            if e.get('asset_type') == 'stock' and e.get('reason') == 'buys_paused' and e['time'] >= since]
    if not sigs:
        return {'trades': 0, 'closed': 0, 'total_vs_spy_pct': 0.0, 'avg_vs_spy_pct': 0.0}
    start = (datetime.fromisoformat(since) - timedelta(days=7)).strftime('%Y-%m-%d')
    tickers = sorted({e['symbol'] for e in sigs})
    bars = yf.download(tickers + ['SPY'], start=start, interval='1d', progress=False, group_by='ticker')
    spy_c = bars['SPY']['Close'].dropna()

    trades = []
    for t in tickers:
        px = bars[t].dropna()
        o, h, l, c = (px[k].values for k in ('Open', 'High', 'Low', 'Close'))
        dates = [d.strftime('%Y-%m-%d') for d in px.index]
        held_until = None   # date the hypothetical position closes ('open' = still held)
        for e in sorted((e for e in sigs if e['symbol'] == t), key=lambda e: e['time']):
            day = e['time'][:10]
            if held_until == 'open' or (held_until and day <= held_until) or day not in dates:
                continue
            i = dates.index(day)
            j, ret, hit = simulate_live_exit(o, h, l, c, i, entry=float(e['price']))
            closed = j < len(c) - 1 or (j - i) >= LIVE_MAX_HOLD
            spy_ret = float(spy_c.asof(px.index[j]) / spy_c.asof(px.index[i]) - 1)
            trades.append({'symbol': t, 'entry': day, 'exit': dates[j] if closed else None,
                           'ret': ret, 'vs_spy': ret - spy_ret, 'hit_tp': hit, 'closed': closed})
            held_until = dates[j] if closed else 'open'
    vs = np.array([tr['vs_spy'] for tr in trades]) if trades else np.array([0.0])
    return {
        'trades': len(trades),
        'closed': sum(tr['closed'] for tr in trades),
        'hit_tp': sum(tr['hit_tp'] for tr in trades),
        'total_vs_spy_pct': round(float(vs.sum()) * 100, 1),   # sum of per-trade % ahead of SPY
        'avg_vs_spy_pct': round(float(vs.mean()) * 100, 2),
    }


# ------------------------------------------------------------
# Decisions
# ------------------------------------------------------------
def last_change(state, setting):
    for ch in reversed(state.get('changes', [])):
        if ch['setting'] == setting and ch.get('applied'):
            return ch
    return None


def plan_changes(src, review, state):
    """List of (setting, new_value, description, reason, source) to apply this run."""
    plans, notes = [], []

    for req in (review or {}).get('setting_changes', []):
        kind, reason = req['change'], req['reason']
        if kind == 'pause_stock_buys':
            if get_setting(src, 'BUYS_PAUSED') == 'True':
                notes.append('Review asked to pause stock buys; already paused.')
            else:
                plans.append(('BUYS_PAUSED', 'True', 'Paused stock buys', reason, 'review'))
        elif kind == 'disable_stop_override':
            if get_setting(src, 'STOP_OVERRIDE_CONF') == 'None':
                notes.append('Review asked to switch off the stop override; already off.')
            else:
                plans.append(('STOP_OVERRIDE_CONF', 'None', 'Switched off the stop override', reason, 'review'))
        elif kind == 'reduce_max_positions':
            cur = int(get_setting(src, 'MAX_POSITIONS'))
            if cur <= MIN_POSITIONS:
                notes.append(f'Review asked to cut max positions; already at the floor of {MIN_POSITIONS}.')
            else:
                plans.append(('MAX_POSITIONS', str(cur - 1), f'Cut max stock positions from {cur} to {cur - 1}',
                              reason, 'review'))

    # Restart stock buys: fixed rules only, and never in the same run as a tightening
    check = None
    if get_setting(src, 'BUYS_PAUSED') == 'True' and not plans:
        last = last_change(state, 'BUYS_PAUSED')
        paused_since = last['date'] if last and last['to'] == 'True' else FIRST_PAUSE
        fwd = forward_test(paused_since)
        weeks = (datetime.now() - datetime.fromisoformat(paused_since)).days / 7
        info = read_json(MODEL_INFO, {})
        rules = {
            f'paused at least {RESTART_MIN_WEEKS_PAUSED} weeks': weeks >= RESTART_MIN_WEEKS_PAUSED,
            f'at least {RESTART_MIN_TRADES} forward-test trades': fwd['trades'] >= RESTART_MIN_TRADES,
            f'at least {RESTART_MIN_CLOSED} of them finished': fwd['closed'] >= RESTART_MIN_CLOSED,
            'forward-test trades ahead of SPY in total': fwd['total_vs_spy_pct'] > 0,
            "retrain's live-rule test says has_edge": info.get('has_edge') is True,
            "this week's review is not red": bool(review) and review.get('status') != 'red',
            f'no change to stock buys in the last {COOLING_OFF_DAYS} days':
                not last or (datetime.now() - datetime.fromisoformat(last['date'])).days >= COOLING_OFF_DAYS,
        }
        check = {'date': datetime.now().strftime('%Y-%m-%d'), 'paused_since': paused_since,
                 'weeks_paused': round(weeks, 1), 'forward_test': fwd, 'rules': rules,
                 'all_pass': all(rules.values())}
        if check['all_pass']:
            plans.append(('BUYS_PAUSED', 'False', 'Restarted stock buys',
                          f"All restart rules passed: {fwd['trades']} forward-test trades "
                          f"({fwd['closed']} finished), {fwd['total_vs_spy_pct']:+.1f}% vs SPY in total.", 'rules'))
    return plans, notes, check


UNDO = {
    'BUYS_PAUSED':        lambda old: f'set BUYS_PAUSED = {old} in trader.py',
    'STOP_OVERRIDE_CONF': lambda old: f'set STOP_OVERRIDE_CONF = {old} in trader.py',
    'MAX_POSITIONS':      lambda old: f'set MAX_POSITIONS = {old} in trader.py',
}


def git_commit_push(message):
    def run(*cmd):
        return subprocess.run(cmd, cwd=HOME, capture_output=True, text=True,
                              env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    run('git', 'add', 'trader.py')
    r = run('git', 'commit', '-q', '-m', message)
    if r.returncode != 0:
        return None, f'commit failed: {r.stderr.strip() or r.stdout.strip()}'
    sha = run('git', 'rev-parse', '--short', 'HEAD').stdout.strip()
    p = run('git', 'push', '-q', 'origin', 'master')
    return sha, None if p.returncode == 0 else f'push failed: {p.stderr.strip().splitlines()[-1:]}'


def send_email(env, subject, body):
    msg = MIMEText(body)
    msg['Subject'], msg['From'], msg['To'] = subject, env['GMAIL_USER'], env['GMAIL_TO']
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(env['GMAIL_USER'], env['GMAIL_APP_PASSWORD'])
        server.sendmail(env['GMAIL_USER'], env['GMAIL_TO'], msg.as_string())


def main():
    dry_run = '--dry-run' in sys.argv
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"🤖 Auto-adjust — {datetime.now():%Y-%m-%d %H:%M}{'  (DRY RUN)' if dry_run else ''}")
    state = read_json(STATE_FILE, {'changes': []})
    src = open(TRADER).read()

    reviews = read_json(REVIEWS_FILE, [])
    latest = reviews[-1] if reviews else {}
    review = latest.get('review') if latest.get('date') == today else None
    if not review:
        print('  ⚠️  No review from today, so no review-requested changes and no restart')

    plans, notes, check = plan_changes(src, review, state)
    if check:
        state['last_restart_check'] = check
        fwd = check['forward_test']
        print(f"  Restart check: {sum(check['rules'].values())}/{len(check['rules'])} rules pass — "
              f"{fwd['trades']} forward trades ({fwd['closed']} finished), {fwd['total_vs_spy_pct']:+.1f}% vs SPY")
        for rule, ok in check['rules'].items():
            print(f"    {'✅' if ok else '❌'} {rule}")
    for n in notes:
        print(f'  ℹ️  {n}')

    autonomous = AUTONOMY and on_paper_api(src)
    if plans and not autonomous:
        print('  ⚠️  Suggest-only mode (AUTONOMY off or not on the paper API): changing nothing')

    email_lines = []
    for setting, new, desc, reason, source in plans:
        old = get_setting(src, setting)
        rec = {'date': today, 'setting': setting, 'from': old, 'to': new, 'description': desc,
               'reason': reason, 'source': source, 'applied': False}
        if autonomous and not dry_run:
            try:
                set_setting(setting, new, dry_run)
                rec['applied'] = True
                sha, err = git_commit_push(f'auto_adjust: {desc}\n\n{reason}\n\nSource: '
                                           f"{'weekly review' if source == 'review' else 'restart rules'}")
                rec['commit'] = sha
                if err:
                    rec['git_error'] = err
                    print(f'  ERROR {err}')
            except Exception as e:
                rec['error'] = str(e)
                print(f'  ERROR applying {setting}: {e}')
        if not dry_run:
            state['changes'].append(rec)
        status = 'APPLIED' if rec['applied'] else ('would apply' if dry_run else 'SUGGESTED ONLY')
        print(f"  {'✅' if rec['applied'] else '➡️ '} {status}: {desc} ({setting} {old} → {new})")
        email_lines += [f"{desc}  [{status}]", f"  Setting: {setting} {old} -> {new}",
                        f"  Why: {reason}", f"  Decided by: {'the weekly review' if source == 'review' else 'the fixed restart rules'}",
                        f"  To undo: {UNDO[setting](old)}", f"  Commit: {rec.get('commit') or '-'}", '']
    if not plans:
        print('  No changes this week')

    if dry_run:
        return
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=1)
    if email_lines:
        body = ('The trading bot changed its own settings this week.\n\n' if autonomous else
                'The weekly review wants these setting changes. Nothing was changed (suggest-only mode).\n\n')
        body += '\n'.join(email_lines) + '\nDetails: auto_adjust.log and the /review page.\n'
        try:
            send_email(load_env(), f"Trading bot: {len(plans)} setting change(s) — {today}", body)
            print('  📧 Email sent')
        except Exception as e:
            print(f'  ERROR sending email: {e}')


if __name__ == '__main__':
    main()
