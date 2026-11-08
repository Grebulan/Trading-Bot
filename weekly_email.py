#!/usr/bin/env python3
"""
Weekly summary email, Sunday 06:30, after the lab, the review, auto-adjust and bot_agent (2026-11-08).
Replaces the daily email_summary.py.

Covers: performance vs SPY, the weekly review's verdict and urgent actions, the changes the
self-improvement agent (bot_agent.py) made, which strategy is
trading the account, any automatic setting changes, the strategy lab's challengers and their
shadow-trading records, this week's lab activity, and current positions.

  python3 weekly_email.py            # send
  python3 weekly_email.py --preview  # write weekly_email_preview.html instead of sending
"""
import json
import smtplib
import sys
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from html import escape

import alpaca_trade_api as tradeapi

from weekly_review import load_env, gather_performance, read_json
from lab import PROMOTE_MIN_DAYS, PROMOTE_MARGIN

HOME = '/home/jv'

UNIVERSE_NAMES = {'sp500': 'S&P 500', 'sp500_tech': 'S&P 500 tech', 'sp400': 'S&P 400 mid-caps',
                  'etf_rotation': 'ETF rotation', 'leveraged_etfs': 'leveraged ETFs', 'legacy24': 'old 24-stock list'}
SIGNAL_NAMES = {'mom_12_1': '12-month momentum', 'mom_6m': '6-month momentum', 'mom_3m': '3-month momentum',
                'mom_1m': '1-month momentum', 'reversal_1w': '1-week reversal', 'rsi_oversold': 'most oversold (RSI)',
                'near_52w_high': 'nearest 52-week high', 'low_vol': 'lowest volatility',
                'risk_adj_mom': 'risk-adjusted momentum'}
COLOURS = {'green': '#1a7f37', 'amber': '#b7791f', 'red': '#c53030'}


def describe(cfg):
    """Plain-English strategy description."""
    if not cfg:
        return ''
    parts = [UNIVERSE_NAMES.get(cfg['universe'], cfg['universe']),
             f"top {cfg['top_k']} by {SIGNAL_NAMES.get(cfg['signal'], cfg['signal'])}",
             f"rebalance every {cfg['rebalance_days']} trading days"]
    if cfg['weighting'] == 'inv_vol':
        parts.append('calmer stocks weighted more')
    if cfg['leverage'] > 1:
        parts.append(f"{cfg['leverage']:g}× leverage")
    if cfg['shorts']:
        parts.append(f"{cfg['shorts']:.0%} short")
    if cfg['regime'] == 'spy200_cash':
        parts.append('cash when SPY is below its 200-day average')
    if cfg['trail_stop']:
        parts.append(f"{cfg['trail_stop']:.0%} trailing stop")
    return ' · '.join(parts)


def champion_name(champ):
    if champ['id'] == 'legacy_ml':
        return 'Original ML stock picker (trader.py)'
    if champ['id'] == 'spy_hold':
        return 'Plain SPY (fallback after a demotion)'
    return describe(champ.get('cfg'))


def pct(x, signed=True):
    return '—' if x is None else (f'{x:+.1f}%' if signed else f'{x:.1f}%')


def coloured(x, text=None):
    text = text or pct(x)
    if x is None:
        return text
    return f'<span style="color:{"#1a7f37" if x >= 0 else "#c53030"}">{text}</span>'


def table(headers, rows, align=None):
    th = ''.join(f'<th style="text-align:left;padding:6px 8px;border-bottom:2px solid #ddd;font-size:13px">{h}</th>'
                 for h in headers)
    body = ''
    for r in rows:
        body += '<tr>' + ''.join(f'<td style="padding:6px 8px;border-bottom:1px solid #eee;font-size:13px;'
                                 f'vertical-align:top">{c}</td>' for c in r) + '</tr>'
    return f'<table style="border-collapse:collapse;width:100%;margin:8px 0 18px">{th and "<tr>" + th + "</tr>"}{body}</table>'


def section(title, inner):
    return f'<h2 style="font-size:17px;margin:22px 0 6px;color:#222">{title}</h2>{inner}'


def build(env):
    api = tradeapi.REST(env['ALPACA_KEY'], env['ALPACA_SECRET'], 'https://paper-api.alpaca.markets',
                        api_version='v2')
    perf = gather_performance(api)
    today = datetime.now().strftime('%Y-%m-%d')
    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    reviews = read_json(f'{HOME}/weekly_reviews.json', [])
    entry = reviews[-1] if reviews and reviews[-1].get('date', '') >= week_ago else {}
    review = entry.get('review')
    ms = read_json(f'{HOME}/milestones.json', [])
    sharpe = ms[-1].get('sharpe') if ms else None
    champ = read_json(f'{HOME}/champion.json', {'id': 'legacy_ml', 'since': '2026-03-06'})
    lab = read_json(f'{HOME}/lab_state.json', {})
    adjust = read_json(f'{HOME}/auto_adjust_state.json', {})

    lw, l4, ss = perf['last_week'], perf['last_4_weeks'], perf['since_start']
    html = ['<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:720px;'
            'margin:0 auto;color:#222;line-height:1.45">',
            f'<h1 style="font-size:21px;margin:0 0 4px">Trading bot — week to {datetime.now():%d %b %Y}</h1>',
            '<p style="color:#666;margin:0 0 14px;font-size:13px">Paper account · no real money</p>']

    # Performance
    rows = [[label, coloured(p['bot_pct']), coloured(p['spy_pct']),
             coloured(None if p['bot_pct'] is None or p['spy_pct'] is None else p['bot_pct'] - p['spy_pct'])]
            for label, p in (('This week', lw), ('Last 4 weeks', l4), (f"Since start ({ss['from']})", ss))]
    html.append(section('Performance',
        f'<p style="margin:4px 0"><b style="font-size:20px">${perf["equity"]:,.0f}</b> &nbsp; '
        f'Sharpe ratio {sharpe if sharpe is not None else "—"} <span style="color:#666">(target above 1.0)</span></p>'
        + table(['', 'Bot', 'SPY', 'Difference'], rows)))

    # Review verdict
    if review:
        colour = COLOURS.get(review['status'], '#444')
        now = [a for a in review.get('recommended_actions', []) if a['priority'] == 'now']
        inner = (f'<p><span style="background:{colour};color:#fff;padding:2px 8px;border-radius:4px;'
                 f'font-size:12px;font-weight:600">{review["status"].upper()}</span> {escape(review["headline"])}</p>')
        if now:
            inner += ('<p style="margin:8px 0 2px"><b>Its most urgent recommendations</b> '
                      '<span style="color:#666;font-size:13px">(the bot carries these out itself — see below)</span></p>'
                      '<ul style="margin:0;padding-left:20px">' +
                      ''.join(f'<li style="margin:3px 0">{escape(a["action"])}</li>' for a in now) + '</ul>')
        inner += '<p style="font-size:13px;color:#666">Full review on the dashboard\'s /review page.</p>'
        html.append(section('Weekly review', inner))
    elif entry.get('error'):
        html.append(section('Weekly review', f'<p style="color:#c53030">The review failed: {escape(entry["error"])}</p>'))

    # What the bot changed in itself
    runs = [r for r in read_json(f'{HOME}/agent_history.json', []) if r['date'] >= week_ago]
    if runs:
        r = runs[-1]
        rep = r.get('report') or {}
        outcome = r.get('outcome')
        if outcome == 'applied':
            inner = f'<p style="margin:4px 0">{escape(rep.get("summary", ""))}</p>'
            inner += '<ul style="margin:4px 0;padding-left:20px">' + ''.join(
                f'<li style="margin:6px 0"><b>{escape(c["title"])}</b><br>{escape(c["plain_english"])}'
                f'<br><span style="color:#666;font-size:13px">Why: {escape(c["why"])}</span></li>'
                for c in rep.get('changes', [])) + '</ul>'
            inner += f'<p style="font-size:13px;color:#666">Saved to GitHub as {r.get("commit", "—")}. All safety checks passed.</p>'
        elif outcome == 'no_changes':
            inner = f'<p style="margin:4px 0">No changes this week. {escape(rep.get("summary", ""))}</p>'
        elif outcome == 'reverted_checks_failed':
            inner = ('<p style="margin:4px 0;color:#b7791f">The bot tried to change itself, but its changes failed the '
                     'safety checks, so they were all undone. It will see this next week and try again.</p>'
                     f'<p style="font-size:13px;color:#666">What it attempted: {escape(rep.get("summary", ""))}</p>')
        else:
            inner = (f'<p style="color:#c53030">The self-improvement run failed: {escape(r.get("error") or outcome or "unknown")}. '
                     'Nothing was changed.</p>')
        if rep.get('not_done'):
            inner += ('<p style="margin:8px 0 2px"><b>Not done</b></p><ul style="margin:0;padding-left:20px">' + ''.join(
                f'<li style="margin:3px 0">{escape(n["recommendation"])} — <span style="color:#666">{escape(n["reason"])}</span></li>'
                for n in rep['not_done']) + '</ul>')
        html.append(section('Changes the bot made to itself', inner))

    # Who is trading
    inner = f'<p style="margin:4px 0"><b>Champion:</b> {escape(champion_name(champ))} <span style="color:#666">(since {champ.get("since", "—")})</span></p>'
    if champ['id'] == 'legacy_ml':
        settings = (entry.get('facts') or {}).get('settings', {})
        if settings.get('stock_buys_paused'):
            inner += '<p style="margin:4px 0;font-size:13px;color:#666">Its stock buys are paused, so the account is mostly SPY.</p>'
    changes = [c for c in adjust.get('changes', []) if c['date'] >= week_ago]
    if changes:
        inner += '<p style="margin:8px 0 2px"><b>Automatic setting changes this week</b></p><ul style="margin:0;padding-left:20px">' + \
                 ''.join(f'<li>{escape(c["description"])} — {escape(c["reason"])}'
                         f'{"" if c.get("applied") else " (suggested only)"}</li>' for c in changes) + '</ul>'
    rc = adjust.get('last_restart_check')
    if rc and champ['id'] == 'legacy_ml':
        passed = sum(rc['rules'].values())
        inner += (f'<p style="font-size:13px;color:#666;margin:6px 0">Restarting its stock buys: {passed} of '
                  f'{len(rc["rules"])} rules pass ({rc["forward_test"]["trades"]} forward-test trades so far).</p>')
    html.append(section('Who is trading the account', inner))

    # Challengers
    chs = lab.get('challengers', [])
    if chs:
        rows = []
        for c in sorted(chs, key=lambda c: -(c['forward'].get('return') or 0)):
            f = c.get('forward') or {}
            days = f.get('days', 0)
            if days < PROMOTE_MIN_DAYS:
                status = f'{PROMOTE_MIN_DAYS - days} trading days until it can be promoted'
            elif f.get('vs_spy', 0) <= 0:
                status = 'behind SPY'
            elif f.get('account_return') is not None and f['return'] < f['account_return'] + PROMOTE_MARGIN:
                status = 'not far enough ahead of the account'
            else:
                status = '<b>qualifies for promotion</b>'
            ret = None if not days else f['return'] * 100
            spy = None if not days else f['spy_return'] * 100
            acct = None if not days or f.get('account_return') is None else f['account_return'] * 100
            oos = (c.get('backtest') or {}).get('oos', {})
            rows.append([escape(describe(c['cfg'])) + f'<br><span style="color:#888;font-size:12px">since {c["forward_start"]} · '
                         f'backtest 2023–now {oos.get("cagr", 0) * 100:+.0f}%/yr, worst fall {oos.get("max_drawdown", 0) * 100:.0f}%</span>',
                         f'{days}d', coloured(ret), coloured(spy), coloured(acct), status])
        inner = table(['Challenger', 'Shadow days', 'Return', 'SPY', 'Account', 'Status'], rows)
        inner += (f'<p style="font-size:13px;color:#666">Returns are from shadow trading since each challenger was chosen. '
                  f'Promotion needs {PROMOTE_MIN_DAYS}+ trading days, beating SPY, and beating the account by '
                  f'{PROMOTE_MARGIN:.0%}.</p>')
    else:
        inner = '<p>No challengers yet.</p>'
    html.append(section('Challengers (shadow trading)', inner))

    # Lab activity this week
    lr = lab.get('last_run') or {}
    events = [e for e in lab.get('events', []) if e['date'] >= week_ago]
    inner = (f'<p style="margin:4px 0">Tested {lr.get("tested", 0)} new strategies; {lr.get("passed", 0)} passed every check.</p>'
             if lr.get('date', '') >= week_ago else '<p style="color:#c53030">The lab did not run this week.</p>')
    labels = {'new_challenger': '➕ New challenger', 'retired': '➖ Retired', 'promoted': '👑 Promoted',
              'demoted': '⬇️ Champion demoted'}
    if events:
        by_id = {c['id']: c for c in lab.get('challengers', []) + lab.get('retired', [])}
        inner += '<ul style="margin:0;padding-left:20px">' + ''.join(
            f'<li>{labels.get(e["event"], e["event"])}: {escape(describe(by_id.get(e["id"], {}).get("cfg")) or e["id"])}'
            f'{" — " + escape(e["reason"]) if e.get("reason") else ""}</li>' for e in events) + '</ul>'
    html.append(section('Strategy lab this week', inner))

    # Positions
    rows = [[p['symbol'], f'${p["value"]:,.0f}', pct(p['weight_pct'], signed=False), coloured(p['unrealized_pl_pct'])]
            for p in perf['positions']]
    html.append(section('Positions', table(['Symbol', 'Value', 'Weight', 'Gain/loss'], rows) +
                        f'<p style="font-size:13px;color:#666">Cash ${perf["cash"]:,.0f}</p>'))

    html.append('</div>')
    subject = (f"🤖 Weekly — {datetime.now():%d %b} — bot {pct(lw['bot_pct'])} vs SPY {pct(lw['spy_pct'])}"
               + (f" — {review['status'].upper()}" if review else ''))
    return subject, ''.join(html)


def send(env, subject, html):
    msg = MIMEMultipart('alternative')
    msg['Subject'], msg['From'], msg['To'] = subject, env['GMAIL_USER'], env['GMAIL_TO']
    msg.attach(MIMEText(html, 'html'))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(env['GMAIL_USER'], env['GMAIL_APP_PASSWORD'])
        server.sendmail(env['GMAIL_USER'], env['GMAIL_TO'], msg.as_string())


def main():
    env = load_env()
    subject, html = build(env)
    if '--preview' in sys.argv:
        with open(f'{HOME}/weekly_email_preview.html', 'w') as f:
            f.write(f'<!-- {subject} -->\n{html}')
        print(f'Preview written: {subject}')
        return
    send(env, subject, html)
    print(f'✅ Weekly email sent: {subject}')


if __name__ == '__main__':
    main()
