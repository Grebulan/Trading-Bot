#!/usr/bin/env python3
"""
Weekly review agent — runs Sunday after record_milestone.py.

Gathers facts about the bot (performance vs SPY, model health, log errors, Pi health),
asks Claude to review them, and saves the report to weekly_reviews.json for the
dashboard's /review page. Read-only: it never places trades or changes settings itself. Its
setting_changes (risk-reducing only) are applied by auto_adjust.py, which runs after it.
"""
import os
import re
import json
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import anthropic
import yfinance as yf
import alpaca_trade_api as tradeapi

from trade_tally import tally

HOME         = '/home/jv'
REVIEWS_FILE = f'{HOME}/weekly_reviews.json'
LOG_STATE_FILE = f'{HOME}/weekly_review_logstate.json'   # how far each log was read by the last review
MODEL        = 'claude-opus-5-5'
SPY_STRATEGY_START = datetime(2026, 6, 30).date()   # first SPY-default buy
LOGS = {   # log file -> how often cron should write to it (hours), for staleness checks
    'trader.log':          24 * 3,   # weekdays only, so allow a weekend gap
    'crypto.log':          2,
    'portfolio.log':       1,
    'email.log':           24,
    'retrain.log':         24 * 7,
    'milestone.log':       24 * 7,
    'sentiment.log':       1,
    'stock_sentiment.log': 1,
    'performance.log':     24,
    'auto_commit.log':     24,      # nightly commit + push to GitHub
    'auto_adjust.log':     24 * 7,  # Sunday automatic setting changes
    'lab.log':             24 * 7,  # Sunday strategy lab
    'champion.log':        24 * 3,  # weekdays 15:30 ET (idle while legacy_ml is champion)
}


def load_env():
    env = {}
    with open(os.path.expanduser('~/.env_trading')) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k] = v
    return env


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def tail(path, n):
    try:
        with open(path, errors='replace') as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def pct(a, b):
    return round((b / a - 1) * 100, 2) if a else None


def spy_return_since(start_date):
    spy = yf.download('SPY', start=start_date.isoformat(), interval='1d',
                      progress=False, auto_adjust=True)
    closes = spy['Close'].squeeze().dropna()
    return pct(float(closes.iloc[0]), float(closes.iloc[-1]))


# ------------------------------------------------------------
# FACT GATHERING
# ------------------------------------------------------------
def gather_performance(api):
    acct = api.get_account()
    equity = float(acct.equity)
    hist = api.get_portfolio_history(period='all', timeframe='1D')
    pts = [(datetime.fromtimestamp(t, timezone.utc).date(), e)
           for t, e in zip(hist.timestamp, hist.equity) if e]
    start_date, start_eq = pts[0]

    def bot_vs_spy(days=None, since=None):
        since = since or datetime.now(timezone.utc).date() - timedelta(days=days)
        base = next(((d, e) for d, e in pts if d >= since), pts[0])
        return {'bot_pct': pct(base[1], equity), 'spy_pct': spy_return_since(base[0]),
                'from': base[0].isoformat()}

    positions = []
    for p in api.list_positions():
        mv = float(p.market_value)
        positions.append({'symbol': p.symbol, 'value': round(mv, 2),
                          'weight_pct': round(mv / equity * 100, 1),
                          'unrealized_pl_pct': round(float(p.unrealized_plpc) * 100, 2)})
    positions.sort(key=lambda x: -x['value'])

    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    orders = [{'time': str(o.filled_at or o.submitted_at)[:16], 'symbol': o.symbol, 'side': o.side,
               'qty': o.filled_qty or o.qty, 'avg_price': o.filled_avg_price, 'status': o.status}
              for o in api.list_orders(status='all', after=week_ago, limit=100)]

    return {
        'equity': round(equity, 2), 'cash': round(float(acct.cash), 2),
        'since_start': {'bot_pct': pct(start_eq, equity), 'spy_pct': spy_return_since(start_date),
                        'from': start_date.isoformat(), 'start_equity': start_eq},
        # SPY-default strategy began 2026-06-30; before that the portfolio was ML picks + cash
        'since_spy_strategy': bot_vs_spy(since=SPY_STRATEGY_START),
        'last_4_weeks': bot_vs_spy(28),
        'last_week': bot_vs_spy(7),
        'positions': positions,
        'orders_last_7_days': orders,
    }


def gather_models():
    lines = tail(f'{HOME}/retrain.log', 400)
    # keep only the most recent run (from the last stock retrain header onward)
    starts = [i for i, l in enumerate(lines) if 'RETRAINING STOCK MODEL' in l]
    last_run = lines[starts[-1]:] if starts else lines[-80:]
    last_run = [re.sub(r'^\S+ \S+ ', '', l).rstrip() for l in last_run]
    ms = read_json(f'{HOME}/milestones.json', [])
    return {
        'stock_model_info': read_json(f'{HOME}/model_info.json'),
        'crypto_model_info': read_json(f'{HOME}/crypto_model_info.json'),
        'last_retrain_log': last_run[-70:],
        'model_accuracy_history': [{'date': m['date'], **m.get('model', {}), 'sharpe': m.get('sharpe'),
                                    'portfolio_value': m.get('portfolio_value')} for m in ms[-10:]],
    }


def log_start_offsets():
    """Byte offset in each log where this review should start reading errors.

    Normally where the previous review stopped. If the review is re-run on the same
    day, reuse that day's starting points so the re-run sees the same errors."""
    state = read_json(LOG_STATE_FILE, {})
    today = datetime.now().strftime('%Y-%m-%d')
    return state.get('start', {}) if state.get('saved_on') == today else state.get('end', {})


def save_log_offsets(start, end):
    with open(LOG_STATE_FILE, 'w') as f:
        json.dump({'saved_on': datetime.now().strftime('%Y-%m-%d'), 'start': start, 'end': end}, f, indent=2)


def gather_logs(start_offsets, end_offsets):
    """Staleness and errors per log. Only errors written since the last review are
    counted, so an old error isn't reported again every week. Fills end_offsets with
    where reading stopped."""
    now = datetime.now().timestamp()
    out = {}
    for name, expected_hours in LOGS.items():
        path = f'{HOME}/{name}'
        if not os.path.exists(path):
            out[name] = {'missing': True}
            continue
        age_h = round((now - os.path.getmtime(path)) / 3600, 1)
        size = os.path.getsize(path)
        start = start_offsets.get(name)
        with open(path, 'rb') as f:
            if start is None or start > size:   # first run, or the log was truncated
                new_lines = f.read().decode(errors='replace').splitlines()[-3000:]
            else:
                f.seek(start)
                new_lines = f.read(size - start).decode(errors='replace').splitlines()
        end_offsets[name] = size
        errs = [l.strip()[:200] for l in new_lines if re.search(r'error|traceback|exception|❌', l, re.I)]
        out[name] = {
            'hours_since_last_write': age_h,
            'expected_every_hours': expected_hours,
            'looks_stale': age_h > expected_hours * 2 + 1,
            'error_lines_since_last_review': len(errs),
            'sample_errors': list(dict.fromkeys(errs))[-8:],   # distinct, most recent
        }
    return out


def gather_system():
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as e:
            return f'unavailable ({e})'
    du = shutil.disk_usage('/')
    return {
        'uptime': run(['uptime', '-p']),
        'disk_root_used_pct': round(du.used / du.total * 100, 1),
        'disk_root_free_gb': round(du.free / 1e9, 1),
        'cpu_temp': run(['vcgencmd', 'measure_temp']),
        'throttled': run(['vcgencmd', 'get_throttled']),
        'failed_services': run(['systemctl', '--failed', '--no-legend', '--plain']),
        'dashboard_service': run(['systemctl', 'is-active', 'dashboard.service']),
    }


def gather_settings():
    src = open(f'{HOME}/crypto_trader.py').read()
    m = re.search(r'^BUYS_PAUSED\s*=\s*(\w+)', src, re.M)
    # All-zero scores means the news feed is returning nothing (sentiment filter silently off)
    sent = {}
    for name in ('stock_sentiment_cache.json', 'crypto_sentiment_cache.json'):
        cache = read_json(f'{HOME}/{name}')
        if isinstance(cache, dict) and cache:
            scores = [v.get('score', 0) for v in cache.values() if isinstance(v, dict)]
            sent[name] = {'tickers': len(scores), 'nonzero_scores': sum(1 for s in scores if s)}
    stock_src = open(f'{HOME}/trader.py').read()
    def stock_setting(name):
        sm = re.search(rf'^{name}\s*=\s*([^#\n]+)', stock_src, re.M)
        return sm.group(1).strip() if sm else None
    return {'crypto_buys_paused': m.group(1) == 'True' if m else False,
            'stock_buys_paused': stock_setting('BUYS_PAUSED') == 'True',
            'stock_stop_override_conf': stock_setting('STOP_OVERRIDE_CONF'),   # 'None' = override off
            'stock_max_positions': stock_setting('MAX_POSITIONS'),
            'news_sentiment_caches': sent}


def gather_auto_adjust():
    """Recent automatic setting changes and where the restart-buys evidence stands."""
    state = read_json(f'{HOME}/auto_adjust_state.json', {})
    return {'recent_changes': state.get('changes', [])[-6:], 'restart_check': state.get('last_restart_check')}


def gather_lab():
    """Strategy lab: the champion, challengers' forward (shadow) records and what has been learned."""
    state = read_json(f'{HOME}/lab_state.json', {})
    lessons = state.get('lessons', {})
    best = {k: sorted(v.items(), key=lambda kv: kv[1], reverse=True)[:2] for k, v in lessons.items()}
    return {
        'champion': read_json(f'{HOME}/champion.json', {'id': 'legacy_ml'}),
        'last_run': state.get('last_run'),
        'challengers': [{'id': c['id'], 'accepted': c['accepted'], 'forward': c.get('forward'),
                         'oos_backtest': (c.get('backtest') or {}).get('oos')} for c in state.get('challengers', [])],
        'recent_events': state.get('events', [])[-10:],
        'best_choices_so_far': best,   # avg excess CAGR vs SPY per menu choice (forward results weigh 3x)
    }


def gather_facts(env, log_start, log_end):
    api = tradeapi.REST(env['ALPACA_KEY'], env['ALPACA_SECRET'],
                        'https://paper-api.alpaca.markets', api_version='v2')
    facts = {'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M')}
    for key, fn in [('performance', lambda: gather_performance(api)), ('models', gather_models),
                    ('ml_trades_vs_spy', lambda: tally(api)),
                    ('logs', lambda: gather_logs(log_start, log_end)), ('system', gather_system), ('settings', gather_settings),
                    ('auto_adjust', gather_auto_adjust), ('lab', gather_lab)]:
        try:
            facts[key] = fn()
        except Exception as e:
            facts[key] = {'error': f'{type(e).__name__}: {e}'}
    return facts


# ------------------------------------------------------------
# CLAUDE REVIEW
# ------------------------------------------------------------
SYSTEM_PROMPT = """You are the weekly reviewer for an algorithmic paper-trading bot on a Raspberry Pi.
The owner is not an engineer. They are deciding whether the bot is worth trusting with real money,
and they want a frank, plain-English assessment, not cheerleading.

Your job:
- Compare the bot to simply holding SPY over the same periods. Since 2026-06-30 spare cash has been
  parked in SPY by default (since_spy_strategy); before that the portfolio was ML picks plus cash, so
  separate gaps that opened before vs after that change. Most of the portfolio sits in SPY by design, so ask whether the ML-driven part (stocks/crypto picks) is adding anything.
- Check model health. A model accuracy is only meaningful against its baseline: if the info file has
  baseline_accuracy / buy_precision (crypto) or pick_hit_rate (stocks) / random_buy_hit_rate / has_edge,
  judge it on those. For stocks, live_rule_test is the honest measure (real trades under the bot's own
  exit rules, vs SPY, against random buys of the same tickers); pick_hit_rate counts every flagged day
  and overstates the edge. Stock has_edge comes from live_rule_test. Flag sudden
  changes in the accuracy history.
- Look for operational problems: stale logs (cron jobs not running), repeated errors, failed
  services, disk space, Pi throttling or overheating. Log errors are only those written since the
  previous review, so an error mentioned last week that is absent now has not recurred.
- Note anything that changed since the previous review and whether earlier recommendations were acted on.
- Flag small samples honestly (e.g. a few weeks of data say little).
- ml_trades_vs_spy tallies every ML stock trade since SPY-funded buys began (2026-10-04) against
  leaving the same money in SPY. It is the most direct evidence on whether the picks earn their place:
  report the running total and how it changed since the previous review. Each trade also has
  exit_reason (from the bot's signal log) and stop_override_holds (hourly runs where the 12% stop was
  hit but the override kept holding); use them to explain losses beyond -12%.
- lab is the strategy lab (lab.py): weekly it backtests new strategy variants, shadow-trades the best
  as challengers, and promotes one to champion (it then trades the account via champion_trader.py)
  when its forward record beats SPY and the account. The owner chose to let it take big risks on
  paper (leverage, shorts, any US stock or ETF) so it can learn. Judge it on forward records, not
  backtests; flag if it is promoting on thin evidence, churning champions, or if a champion is
  losing heavily. Its promotions are automatic: don't recommend manual tinkering unless something
  is broken.
- settings shows whether stock buys are paused and whether the stop override is on. While stock buys
  are paused, the tally only changes through the remaining open positions.

Recommend actions. You are advisory, with one exception: setting_changes. auto_adjust.py applies these
automatically, without the owner, so request one only for a critical risk finding. Only risk-reducing
changes exist (pause stock buys, switch off the stop override, cut max positions by one). Each must
not already be in place: see settings. Loosening (e.g. restarting stock buys) is never yours to request; fixed evidence rules in
auto_adjust.py decide that, and facts.auto_adjust shows their current state and recent changes.
Leave setting_changes empty in most weeks. Everything else remains the owner's decision. Prefer the smallest,
safest action. Keep each finding to a few sentences, avoid jargon, and explain any number you cite.
The project guide below may contain outdated figures; trust the live facts over it."""

REVIEW_SCHEMA = {
    'type': 'object',
    'properties': {
        'status':   {'type': 'string', 'enum': ['green', 'amber', 'red']},
        'headline': {'type': 'string', 'description': 'One sentence summary of the week'},
        'benchmark_summary': {'type': 'string', 'description': '2-4 sentences: bot vs SPY'},
        'findings': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'severity': {'type': 'string', 'enum': ['critical', 'warning', 'info', 'good']},
                    'area':     {'type': 'string', 'enum': ['performance', 'models', 'operations', 'risk']},
                    'title':    {'type': 'string'},
                    'detail':   {'type': 'string'},
                },
                'required': ['severity', 'area', 'title', 'detail'],
                'additionalProperties': False,
            },
        },
        'recommended_actions': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'priority': {'type': 'string', 'enum': ['now', 'this week', 'when convenient']},
                    'action':   {'type': 'string'},
                    'why':      {'type': 'string'},
                },
                'required': ['priority', 'action', 'why'],
                'additionalProperties': False,
            },
        },
        'setting_changes': {   # applied automatically by auto_adjust.py — risk-reducing only
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'change': {'type': 'string', 'enum': ['pause_stock_buys', 'disable_stop_override',
                                                          'reduce_max_positions']},
                    'reason': {'type': 'string'},
                },
                'required': ['change', 'reason'],
                'additionalProperties': False,
            },
        },
    },
    'required': ['status', 'headline', 'benchmark_summary', 'findings', 'recommended_actions', 'setting_changes'],
    'additionalProperties': False,
}


def ask_claude(env, facts, previous):
    # User-level keys (sk-ant-usr...) aren't tied to a workspace, so the request must name one
    headers = ({'anthropic-workspace-id': env['ANTHROPIC_WORKSPACE_ID']}
               if env.get('ANTHROPIC_WORKSPACE_ID') else None)
    client = anthropic.Anthropic(api_key=env['ANTHROPIC_API_KEY'], default_headers=headers)
    guide = open(f'{HOME}/CLAUDE.md').read() if os.path.exists(f'{HOME}/CLAUDE.md') else ''
    prev = ({k: previous['review'].get(k) for k in ('date', 'headline', 'recommended_actions')}
            if previous and previous.get('review') else None)
    prompt = (f"<project_guide>\n{guide}\n</project_guide>\n\n"
              f"<previous_review>\n{json.dumps(prev, indent=1)}\n</previous_review>\n\n"
              f"<this_weeks_facts>\n{json.dumps(facts, indent=1, default=str)}\n</this_weeks_facts>\n\n"
              "Write this week's review.")
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=['server-side-fallback-2026-07-01'],
        fallbacks='default',
        system=SYSTEM_PROMPT,
        thinking={'type': 'adaptive'},
        output_config={'effort': 'high', 'format': {'type': 'json_schema', 'schema': REVIEW_SCHEMA}},
        messages=[{'role': 'user', 'content': prompt}],
    )
    if response.stop_reason != 'end_turn':
        raise RuntimeError(f'Claude stopped early: {response.stop_reason}')
    text = next(b.text for b in response.content if b.type == 'text')
    usage = response.usage
    cost = (usage.input_tokens * 4 + usage.output_tokens * 20) / 1e6
    return json.loads(text), {'model': response.model, 'input_tokens': usage.input_tokens,
                              'output_tokens': usage.output_tokens, 'approx_cost_usd': round(cost, 3)}


def main():
    print(f"🧐 Weekly review — {datetime.now():%Y-%m-%d %H:%M}")
    env = load_env()
    reviews = read_json(REVIEWS_FILE, [])
    log_start, log_end = log_start_offsets(), {}
    facts = gather_facts(env, log_start, log_end)
    entry = {'date': datetime.now().strftime('%Y-%m-%d'), 'facts': facts}

    if not env.get('ANTHROPIC_API_KEY'):
        entry['error'] = 'ANTHROPIC_API_KEY is not set in ~/.env_trading'
    else:
        try:
            review, usage = ask_claude(env, facts, reviews[-1] if reviews else None)
            entry['review'] = {'date': entry['date'], **review}
            entry['usage'] = usage
            print(f"✅ {review['status'].upper()}: {review['headline']}  (~${usage['approx_cost_usd']})")
        except anthropic.AuthenticationError:
            entry['error'] = 'Claude API key was rejected — check ANTHROPIC_API_KEY'
        except anthropic.APIStatusError as e:
            entry['error'] = f'Claude API error {e.status_code}: {e.message}'
            if 'workspace' in str(e.message):
                entry['error'] += (' — add ANTHROPIC_WORKSPACE_ID=<workspace id> to ~/.env_trading,'
                                   ' or use a workspace API key (sk-ant-api...)')
        except anthropic.APIConnectionError:
            entry['error'] = 'Could not reach the Claude API (network error)'
        except Exception as e:
            entry['error'] = f'{type(e).__name__}: {e}'
    if 'error' in entry:
        print(f"❌ {entry['error']}")

    reviews = [r for r in reviews if r.get('date') != entry['date']] + [entry]
    with open(REVIEWS_FILE, 'w') as f:
        json.dump(reviews[-52:], f, indent=1, default=str)   # keep a year
    if 'error' not in entry:   # if the review failed, next week's run re-reads these errors
        save_log_offsets(log_start, log_end)


if __name__ == '__main__':
    main()
