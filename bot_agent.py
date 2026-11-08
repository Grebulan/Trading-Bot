#!/usr/bin/env python3
"""
Self-improvement agent (Sunday 05:30, 2026-11-08). The owner has delegated decisions to the bot:
this agent reads the weekly review and carries out its recommendations by editing the bot's code,
then reports what it changed in the weekly email.

It runs on the Claude API with its own small set of tools, so it can only:
- read the bot's files, data and logs (never ~/.env_trading),
- edit the files in EDITABLE and create new .py files in /home/jv,
- run backtests and a fixed set of checks.

Guardrails, all enforced here rather than trusted to the model:
- PROTECTED files are the bot's scorekeeping and safety rails (the review, the trade tally, the lab's
  promotion rules, auto-adjust, the weekly email, this file). The bot must not grade its own homework,
  or "improvements" could just be changes to how it is measured. PROTECTED_LINES pin paper trading,
  backtest costs, the leverage cap and the model's has_edge bar.
- After the agent finishes, CHECKS run: compile, new pylint errors, trader/crypto_trader import,
  lab/champion/auto-adjust dry runs, protected lines. If any fail, every change is reverted.
- Passing changes are committed and pushed, and recorded in agent_history.json for the weekly email.
- Budget: MAX_TURNS and MAX_COST_USD per run.

  python3 bot_agent.py            # Sunday run
  python3 bot_agent.py --dry-run  # agent works, then everything is reverted (nothing committed)
"""
import json
import os
import py_compile
import re
import shutil
import subprocess
import sys
from datetime import datetime

import anthropic

HOME         = '/home/jv'
HISTORY_FILE = f'{HOME}/agent_history.json'
CREATED_FILE = f'{HOME}/agent_created_files.json'
BACKUP_ROOT  = f'{HOME}/.agent_backup'
MODEL        = 'claude-opus-5-5'
MAX_TURNS    = 60
MAX_COST_USD = 10.0
PRICE        = {'input': 4.0, 'output': 20.0, 'cache_write': 5.0, 'cache_read': 0.20}   # $ per MTok

EDITABLE = ['trader.py', 'crypto_trader.py', 'retrain.py', 'sentiment.py', 'crypto_sentiment.py',
            'finbert_service.py', 'performance_log.py', 'record_milestone.py', 'dashboard.py',
            'lab_engine.py', 'lab_data.py', 'champion_trader.py', 'CLAUDE.md', 'README.md']
PROTECTED = ['weekly_review.py', 'trade_tally.py', 'lab.py', 'auto_adjust.py', 'weekly_email.py',
             'bot_agent.py', 'auto_commit.sh', 'run_trader_et.sh', 'logts.sh', 'champion.json',
             '.gitignore', '.env_trading']
READ_ONLY_DATA = ['weekly_reviews.json', 'lab_state.json', 'champion.json', 'model_info.json',
                  'crypto_model_info.json', 'milestones.json', 'auto_adjust_state.json',
                  'agent_history.json', 'performance.json', 'stock_highwater.json', 'champion_state.json']
LOGS = ['trader.log', 'crypto.log', 'portfolio.log', 'retrain.log', 'lab.log', 'champion.log',
        'auto_adjust.log', 'weekly_review.log', 'email.log', 'auto_commit.log', 'sentiment.log',
        'stock_sentiment.log', 'milestone.log', 'agent.log']
PROTECTED_LINES = {
    'trader.py':          ['BASE_URL = "https://paper-api.alpaca.markets"'],
    'crypto_trader.py':   ['BASE_URL = "https://paper-api.alpaca.markets"'],
    'champion_trader.py': ["api = tradeapi.REST(env['ALPACA_KEY'], env['ALPACA_SECRET'], 'https://paper-api.alpaca.markets',"],
    'lab_engine.py':      ['COST_BPS        = 5', 'MARGIN_RATE     = 0.065', 'BORROW_FEE      = 0.005',
                           'MAX_GROSS_LIVE  = 1.95'],
    'retrain.py':         ["'has_edge':            bool(len(m) >= 50 and extra >= 2 * se),"],
}


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


def sh(cmd, timeout=900):
    r = subprocess.run(cmd, cwd=HOME, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, 'GIT_TERMINAL_PROMPT': '0'})
    return r.returncode, (r.stdout + r.stderr)


def clip(text, n=12000):
    return text if len(text) <= n else text[:n // 2] + f'\n... [{len(text) - n} characters cut] ...\n' + text[-n // 2:]


# ------------------------------------------------------------
# Workspace: what may be read/edited, backups and revert
# ------------------------------------------------------------
class Workspace:
    def __init__(self):
        self.created = read_json(CREATED_FILE, [])          # files the agent made in earlier runs
        self.new_files, self.touched = [], set()
        self.backup = f"{BACKUP_ROOT}/{datetime.now():%Y%m%d-%H%M%S}"
        os.makedirs(self.backup, exist_ok=True)
        for f in self.editable():
            if os.path.exists(f'{HOME}/{f}'):
                shutil.copy2(f'{HOME}/{f}', f'{self.backup}/{f}')

    def editable(self):
        return EDITABLE + [f for f in self.created if f not in PROTECTED]

    @staticmethod
    def name(path):
        p = path.replace(HOME + '/', '').strip()
        if '/' in p or p.startswith('.') or not p:
            raise ValueError(f'{path}: only files directly in {HOME} are allowed')
        return p

    def readable(self, p):
        return (p in self.editable() or p in PROTECTED or p in READ_ONLY_DATA or p in LOGS) and p != '.env_trading'

    def revert(self):
        for f in self.touched:
            src = f'{self.backup}/{f}'
            if os.path.exists(src):
                shutil.copy2(src, f'{HOME}/{f}')
        for f in self.new_files:
            if os.path.exists(f'{HOME}/{f}'):
                os.remove(f'{HOME}/{f}')
        self.new_files.clear()
        self.touched.clear()


# ------------------------------------------------------------
# Tools
# ------------------------------------------------------------
TOOLS = [
    {'name': 'list_files', 'description': 'List the files you can edit, read-only files, data files and logs.',
     'input_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False}, 'strict': True},
    {'name': 'read_file', 'description': 'Read a bot file (code, CLAUDE.md, JSON data). 1-based line range; '
                                         'pass 0 and 0 for the whole file. '
                                         'Large JSON files are better summarised with a backtest or by reading a range.',
     'input_schema': {'type': 'object', 'properties': {
         'path': {'type': 'string'}, 'start_line': {'type': 'integer'}, 'end_line': {'type': 'integer'}},
         'required': ['path', 'start_line', 'end_line'], 'additionalProperties': False}, 'strict': True},
    {'name': 'read_log', 'description': 'Read the last N lines of a log file.',
     'input_schema': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'lines': {'type': 'integer'}},
                      'required': ['name', 'lines'], 'additionalProperties': False}, 'strict': True},
    {'name': 'search', 'description': 'Search the bot\'s code files for a regular expression; returns file:line matches.',
     'input_schema': {'type': 'object', 'properties': {'pattern': {'type': 'string'}},
                      'required': ['pattern'], 'additionalProperties': False}, 'strict': True},
    {'name': 'edit_file', 'description': 'Replace one exact, unique occurrence of old_text with new_text in an editable file.',
     'input_schema': {'type': 'object', 'properties': {
         'path': {'type': 'string'}, 'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
         'required': ['path', 'old_text', 'new_text'], 'additionalProperties': False}, 'strict': True},
    {'name': 'create_file', 'description': 'Create a new .py file in /home/jv (not overwriting anything). '
                                           'Nothing runs it unless existing code imports it.',
     'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}, 'content': {'type': 'string'}},
                      'required': ['path', 'content'], 'additionalProperties': False}, 'strict': True},
    {'name': 'run_backtest', 'description': 'Backtest a strategy-lab config (keys as lab_engine.MENU; values may be new '
                                            'ones you added to the engine). Returns in-sample (2015-2022) and '
                                            'out-of-sample (2023-now) metrics vs SPY and vs random picks.',
     'input_schema': {'type': 'object', 'properties': {'config_json': {'type': 'string'}},
                      'required': ['config_json'], 'additionalProperties': False}, 'strict': True},
    {'name': 'run_checks', 'description': 'Run the safety checks that decide whether your changes are kept: compile, '
                                          'new pylint errors, imports, lab/champion/auto-adjust dry runs, protected '
                                          'lines. Run before finishing; fix or undo anything that fails.',
     'input_schema': {'type': 'object', 'properties': {}, 'additionalProperties': False}, 'strict': True},
    {'name': 'undo_file', 'description': 'Put a file back exactly as it was at the start of this run '
                                         '(deletes it if you created it).',
     'input_schema': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                      'required': ['path'], 'additionalProperties': False}, 'strict': True},
    {'name': 'finish', 'description': 'End the run with your report. Call exactly once, after run_checks passes.',
     'input_schema': {'type': 'object', 'properties': {
         'summary': {'type': 'string', 'description': 'Two or three plain-English sentences for a non-expert owner.'},
         'changes': {'type': 'array', 'items': {'type': 'object', 'properties': {
             'title': {'type': 'string'}, 'plain_english': {'type': 'string'},
             'why': {'type': 'string'}, 'files': {'type': 'array', 'items': {'type': 'string'}}},
             'required': ['title', 'plain_english', 'why', 'files'], 'additionalProperties': False}},
         'not_done': {'type': 'array', 'items': {'type': 'object', 'properties': {
             'recommendation': {'type': 'string'}, 'reason': {'type': 'string'}},
             'required': ['recommendation', 'reason'], 'additionalProperties': False}},
         'commit_message': {'type': 'string'}},
         'required': ['summary', 'changes', 'not_done', 'commit_message'], 'additionalProperties': False},
     'strict': True},
]


class Tools:
    def __init__(self, ws, baseline_lint):
        self.ws, self.baseline_lint = ws, baseline_lint
        self.lab = None
        self.report = None

    def run(self, name, args):
        return getattr(self, name)(**args)

    def list_files(self):
        return json.dumps({'editable': self.ws.editable(), 'read_only_code': PROTECTED[:-1],
                           'data': READ_ONLY_DATA, 'logs': [l for l in LOGS if os.path.exists(f'{HOME}/{l}')]}, indent=1)

    def read_file(self, path, start_line, end_line):
        p = self.ws.name(path)
        if not self.ws.readable(p) or not os.path.exists(f'{HOME}/{p}'):
            return f'Error: {p} is not readable (see list_files)'
        lines = open(f'{HOME}/{p}', errors='replace').read().splitlines()
        a = max(1, start_line or 1)
        b = min(len(lines), end_line if end_line and end_line > 0 else len(lines))
        body = '\n'.join(f'{i}\t{lines[i - 1]}' for i in range(a, b + 1))
        return clip(f'{p}: lines {a}-{b} of {len(lines)}\n{body}', 40000)

    def read_log(self, name, lines):
        p = self.ws.name(name)
        if p not in LOGS or not os.path.exists(f'{HOME}/{p}'):
            return f'Error: unknown log {p}'
        return clip('\n'.join(open(f'{HOME}/{p}', errors='replace').read().splitlines()[-min(max(lines, 1), 400):]))

    def search(self, pattern):
        rx = re.compile(pattern)
        out = []
        for f in sorted(set(self.ws.editable() + PROTECTED[:-1])):
            if f.endswith(('.py', '.sh', '.md')) and os.path.exists(f'{HOME}/{f}'):
                for i, line in enumerate(open(f'{HOME}/{f}', errors='replace'), 1):
                    if rx.search(line):
                        out.append(f'{f}:{i}: {line.rstrip()}')
        return clip('\n'.join(out[:300]) or 'no matches')

    def edit_file(self, path, old_text, new_text):
        p = self.ws.name(path)
        if p in PROTECTED or p not in self.ws.editable():
            return f'Error: {p} is protected or not editable. If it needs changing, list it under not_done for the owner.'
        src = open(f'{HOME}/{p}').read()
        n = src.count(old_text)
        if n != 1:
            return f'Error: old_text found {n} times in {p}; it must match exactly once'
        with open(f'{HOME}/{p}', 'w') as f:
            f.write(src.replace(old_text, new_text))
        self.ws.touched.add(p)
        return f'Edited {p}'

    def create_file(self, path, content):
        p = self.ws.name(path)
        if not re.fullmatch(r'[a-z][a-z0-9_]*\.py', p) or os.path.exists(f'{HOME}/{p}') or p in PROTECTED:
            return 'Error: new files must be new lowercase .py names in /home/jv'
        with open(f'{HOME}/{p}', 'w') as f:
            f.write(content)
        self.ws.new_files.append(p)
        return f'Created {p}'

    def undo_file(self, path):
        p = self.ws.name(path)
        if p in self.ws.new_files:
            os.remove(f'{HOME}/{p}')
            self.ws.new_files.remove(p)
            return f'Deleted {p}'
        if p in self.ws.touched and os.path.exists(f'{self.ws.backup}/{p}'):
            shutil.copy2(f'{self.ws.backup}/{p}', f'{HOME}/{p}')
            self.ws.touched.discard(p)
            return f'Restored {p}'
        return f'{p} was not changed this run'

    def run_backtest(self, config_json):
        # Runs in a subprocess so it always uses the engine code as currently edited
        code = ("import json,sys; sys.path.insert(0,'/home/jv'); import lab, lab_data\n"
                "from lab_engine import Lab, normalise\n"
                "L=Lab(lab_data.load_prices(), lab_data.load_universes())\n"
                "cfg=normalise(json.loads(sys.argv[1]), L.u)\n"
                "r=lab.evaluate(L,cfg,{})\nprint(json.dumps(r, default=str))")
        rc, out = sh(['python3', '-c', code, config_json], timeout=900)
        return clip(out.strip().splitlines()[-1] if rc == 0 and out.strip() else f'Error (exit {rc}):\n{out}')

    def run_checks(self):
        ok, lines = run_checks(self.ws, self.baseline_lint)
        return ('ALL CHECKS PASSED' if ok else 'CHECKS FAILED — fix or undo before finishing') + '\n' + '\n'.join(lines)

    def finish(self, summary, changes, not_done, commit_message):
        self.report = {'summary': summary, 'changes': changes, 'not_done': not_done, 'commit_message': commit_message}
        return 'Report received.'


# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------
def lint_errors(files):
    counts = {}
    for f in files:
        if f.endswith('.py') and os.path.exists(f'{HOME}/{f}'):
            rc, out = sh(['python3', '-m', 'pylint', '-E', '--disable=import-error,no-member', '--score=n', f], timeout=300)
            counts[f] = len([l for l in out.splitlines() if re.search(r':\d+:\d+: E\d+', l)])
    return counts


def run_checks(ws, baseline_lint):
    lines, ok = [], True
    files = sorted(set(ws.editable() + ws.new_files))
    for f in files:
        if f.endswith('.py') and os.path.exists(f'{HOME}/{f}'):
            try:
                py_compile.compile(f'{HOME}/{f}', doraise=True)
            except py_compile.PyCompileError as e:
                ok = False
                lines.append(f'❌ compile {f}: {e.msg[:500]}')
    lint = lint_errors(sorted(ws.touched | set(ws.new_files)))
    for f, n in lint.items():
        if n > baseline_lint.get(f, 0):
            ok = False
            _, out = sh(['python3', '-m', 'pylint', '-E', '--disable=import-error,no-member', '--score=n', f], timeout=300)
            lines.append(f'❌ pylint {f}: {n} errors (was {baseline_lint.get(f, 0)})\n{clip(out, 3000)}')
    for f, must in PROTECTED_LINES.items():
        src = open(f'{HOME}/{f}').read()
        for line in must:
            if line not in src:
                ok = False
                lines.append(f'❌ protected line changed in {f}: {line}')
    for label, cmd in [('import trader', ['python3', '-c', 'import trader']),
                       ('import crypto_trader', ['python3', '-c', 'import crypto_trader']),
                       ('lab dry run', ['python3', 'lab.py', '--no-refresh', '--dry-run']),
                       ('champion dry run', ['python3', 'champion_trader.py', '--dry-run']),
                       ('champion test-strategy dry run', ['python3', 'champion_trader.py', '--test-champion']),
                       ('auto-adjust dry run', ['python3', 'auto_adjust.py', '--dry-run'])]:
        rc, out = sh(cmd, timeout=1500)
        errs = [l for l in out.splitlines() if 'Traceback' in l or re.search(r'\bERROR\b', l)]
        if rc != 0 or errs:
            ok = False
            lines.append(f'❌ {label} (exit {rc}):\n{clip(out, 3000)}')
        else:
            lines.append(f'✅ {label}')
    return ok, lines


# ------------------------------------------------------------
SYSTEM = """You are the self-improvement agent for an algorithmic trading bot on a Raspberry Pi, trading a
PAPER account (no real money). The owner is not a trader and has delegated decisions to you: each
Sunday you carry out the weekly review's recommendations yourself by changing the bot's code.

How to work:
- Start from this week's review (below) and your past runs (below, including any that failed checks).
  Look at the code and data before changing anything. Skip recommendations that are already done.
- Prefer changes you can check. For strategy ideas, use run_backtest and compare against SPY and random
  picks; a change that only looks good in one period is probably luck. The strategy lab (lab.py) already
  searches strategy combinations every week; you can widen what it can try by adding new signals,
  universes or options to lab_engine.py's MENU and compute_signal, and new universes to lab_data.py.
- Risk is allowed on paper: the owner wants the bot to try things and learn. But never break trading:
  every change must keep the scheduled jobs working.
- Keep changes focused. Update CLAUDE.md when you change behaviour, so the next run knows.
- Protected files and lines (the scorekeeping, promotion rules, paper-trading URL, backtest costs) are
  not yours to change. The bot must not grade its own homework. If one needs changing, put it in
  not_done with your reasoning; the owner will see it.
- When done, call run_checks. If anything fails, fix it or undo_file it. Then call finish once.
  If nothing is worth changing this week, finish with no changes and say why.
- Write the report for a non-expert: what changed, in plain English, and why."""


def build_context():
    reviews = read_json(f'{HOME}/weekly_reviews.json', [])
    latest = reviews[-1] if reviews else {}
    history = read_json(HISTORY_FILE, [])[-6:]
    guide = open(f'{HOME}/CLAUDE.md').read()
    lab = read_json(f'{HOME}/lab_state.json', {})
    lab_brief = {'champion': read_json(f'{HOME}/champion.json', {}), 'last_run': lab.get('last_run'),
                 'challengers': [{'id': c['id'], 'forward': c.get('forward')} for c in lab.get('challengers', [])],
                 'lessons': lab.get('lessons'), 'recent_events': lab.get('events', [])[-10:]}
    return (f"<project_guide>\n{guide}\n</project_guide>\n\n"
            f"<this_weeks_review date=\"{latest.get('date')}\">\n"
            f"{json.dumps(latest.get('review') or {'error': latest.get('error')}, indent=1)}\n</this_weeks_review>\n\n"
            f"<this_weeks_facts>\n{clip(json.dumps(latest.get('facts', {}), indent=1, default=str), 30000)}\n</this_weeks_facts>\n\n"
            f"<strategy_lab>\n{json.dumps(lab_brief, indent=1, default=str)}\n</strategy_lab>\n\n"
            f"<your_past_runs>\n{json.dumps(history, indent=1, default=str)}\n</your_past_runs>\n\n"
            f"Today is {datetime.now():%Y-%m-%d}. Carry out this week's recommendations.")


def cost_of(usage):
    return (usage.input_tokens * PRICE['input'] + usage.output_tokens * PRICE['output']
            + (usage.cache_creation_input_tokens or 0) * PRICE['cache_write']
            + (usage.cache_read_input_tokens or 0) * PRICE['cache_read']) / 1e6


def agent_loop(env, tools):
    headers = ({'anthropic-workspace-id': env['ANTHROPIC_WORKSPACE_ID']}
               if env.get('ANTHROPIC_WORKSPACE_ID') else None)
    client = anthropic.Anthropic(api_key=env['ANTHROPIC_API_KEY'], default_headers=headers, timeout=900)
    messages = [{'role': 'user', 'content': build_context()}]
    cost, turns = 0.0, 0
    while turns < MAX_TURNS and tools.report is None:
        turns += 1
        with client.beta.messages.stream(
            model=MODEL, max_tokens=64000, system=SYSTEM, tools=TOOLS, messages=messages,
            betas=['server-side-fallback-2026-07-01'], fallbacks='default',
            thinking={'type': 'adaptive'}, output_config={'effort': 'high'},
            cache_control={'type': 'ephemeral'},
        ) as stream:
            resp = stream.get_final_message()
        cost += cost_of(resp.usage)
        messages.append({'role': 'assistant', 'content': resp.content})
        if resp.stop_reason == 'refusal':
            raise RuntimeError('the model declined the task')
        uses = [b for b in resp.content if b.type == 'tool_use']
        for b in resp.content:
            if b.type == 'text' and b.text.strip():
                print(f'  💬 {b.text.strip()[:300]}')
        if not uses:
            if tools.report is None:
                messages.append({'role': 'user', 'content': 'Please call run_checks and then finish with your report.'})
            continue
        results = []
        for u in uses:
            print(f"  🔧 {u.name} {json.dumps(u.input)[:160]}")
            try:
                out = tools.run(u.name, u.input)
                results.append({'type': 'tool_result', 'tool_use_id': u.id, 'content': out})
            except Exception as e:
                results.append({'type': 'tool_result', 'tool_use_id': u.id, 'content': f'Error: {e}', 'is_error': True})
        if cost > MAX_COST_USD * 0.8 and tools.report is None:
            results.append({'type': 'text', 'text': 'Budget nearly used: run_checks and finish now.'})
        messages.append({'role': 'user', 'content': results})
        if cost > MAX_COST_USD:
            break
    return turns, round(cost, 2)


def main():
    dry_run = '--dry-run' in sys.argv
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"🤖 Bot agent — {datetime.now():%Y-%m-%d %H:%M}{'  (DRY RUN)' if dry_run else ''}")
    reviews = read_json(f'{HOME}/weekly_reviews.json', [])
    if not reviews or not reviews[-1].get('review'):
        print('  ⚠️  No usable weekly review — nothing to do')
        return
    env = load_env()
    ws = Workspace()
    baseline_lint = lint_errors(ws.editable())
    tools = Tools(ws, baseline_lint)
    record = {'date': today, 'review_date': reviews[-1]['date']}
    try:
        turns, cost = agent_loop(env, tools)
        record.update(turns=turns, cost_usd=cost)
    except Exception as e:
        print(f'  ERROR agent failed: {type(e).__name__}: {e}')
        ws.revert()
        record.update(outcome='failed', error=f'{type(e).__name__}: {e}')
        if not dry_run:
            history = read_json(HISTORY_FILE, []) + [record]
            json.dump(history[-52:], open(HISTORY_FILE, 'w'), indent=1)
        return

    report = tools.report or {'summary': 'The agent did not finish its report.', 'changes': [], 'not_done': [],
                              'commit_message': ''}
    record['report'] = report
    changed = sorted(ws.touched | set(ws.new_files))
    if not changed:
        record['outcome'] = 'no_changes'
    else:
        ok, lines = run_checks(ws, baseline_lint)
        record['checks'] = lines
        if not ok or tools.report is None or dry_run:
            ws.revert()
            record['outcome'] = 'dry_run_reverted' if dry_run and ok else 'reverted_checks_failed'
            print(f"  ↩️  Reverted: {'dry run' if dry_run else 'checks failed'}")
            for l in lines:
                print(f'    {l[:300]}')
        else:
            if ws.new_files:
                json.dump(sorted(set(ws.created + ws.new_files)), open(CREATED_FILE, 'w'), indent=1)
            msg = (report['commit_message'] or 'bot_agent: weekly self-improvement').strip()
            rc, out = sh(['git', 'add', *changed])
            rc, out = sh(['git', 'commit', '-q', '-m', f'{msg}\n\nMade by bot_agent.py from the {record["review_date"]} '
                                                        f'weekly review.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'])
            if rc != 0:
                print(f'  ERROR commit failed: {out[-300:]}')
            else:
                record['commit'] = sh(['git', 'rev-parse', '--short', 'HEAD'])[1].strip()
                prc, pout = sh(['git', 'push', '-q', 'origin', 'master'])
                if prc != 0:
                    print(f'  ERROR push failed: {pout.strip().splitlines()[-1:]}')
            record['outcome'] = 'applied'
            record['files'] = changed
            print(f"  ✅ Applied {len(report['changes'])} change(s) in {', '.join(changed)} ({record.get('commit')})")
    print(f"  {report['summary']}")
    print(f"  Turns {record['turns']}, cost ~${record['cost_usd']}")
    if not dry_run:
        history = read_json(HISTORY_FILE, []) + [record]
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history[-52:], f, indent=1)


if __name__ == '__main__':
    main()
