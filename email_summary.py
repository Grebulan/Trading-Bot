import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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

ENV = load_env()
api = tradeapi.REST(ENV['ALPACA_KEY'], ENV['ALPACA_SECRET'],
                    "https://paper-api.alpaca.markets", api_version='v2')

STOCK_TICKERS = [
    'AAPL','GOOGL','MSFT','META','AMZN','TSLA','NVDA',
    'JPM','BAC','GS','PFE','MRK','RIVN','NIO',
    'WMT','TGT','AMD','QCOM','MS'
]
CRYPTO_SYMBOLS = [
    'BTC/USD','ETH/USD','SOL/USD','LINK/USD','AVAX/USD',
    'XRP/USD','DOGE/USD','UNI/USD','DOT/USD','ADA/USD'
]

def get_days_orders():
    orders = api.list_orders(status='all', limit=100)
    today  = datetime.now().strftime('%Y-%m-%d')
    stocks, crypto = [], []
    for o in orders:
        if not o.submitted_at or o.submitted_at.strftime('%Y-%m-%d') != today:
            continue
        row = {
            'symbol': o.symbol,
            'side':   o.side.upper(),
            'qty':    float(o.qty),
            'price':  float(o.filled_avg_price) if o.filled_avg_price else 0.0,
            'status': o.status,
            'time':   o.submitted_at.strftime('%H:%M')
        }
        if o.symbol in CRYPTO_SYMBOLS:
            crypto.append(row)
        else:
            stocks.append(row)
    return stocks, crypto

def get_positions():
    positions = api.list_positions()
    stocks, crypto = [], []
    for p in positions:
        row = {
            'symbol':    p.symbol,
            'qty':       float(p.qty),
            'avg_price': float(p.avg_entry_price),
            'current':   float(p.current_price),
            'pl':        round(float(p.unrealized_pl), 2),
            'pl_pct':    round(float(p.unrealized_plpc) * 100, 2),
            'value':     round(float(p.market_value), 2),
        }
        if p.symbol in STOCK_TICKERS:
            stocks.append(row)
        else:
            crypto.append(row)
    return stocks, crypto

def get_portfolio_history():
    if not os.path.exists('/home/jv/portfolio_history.json'):
        return []
    with open('/home/jv/portfolio_history.json') as f:
        return json.load(f)

def orders_table(orders, is_crypto=False):
    unit = 'coins' if is_crypto else 'shares'
    if not orders:
        return f"""<tr><td colspan="6" style="padding:20px;text-align:center;color:#888">
            No {'crypto' if is_crypto else 'stock'} trades today</td></tr>"""
    rows = ""
    for o in orders:
        side_color = '#00c853' if o['side'] == 'BUY' else '#ff1744'
        value      = round(o['qty'] * o['price'], 2)
        qty_str    = f"{o['qty']:.6f}" if is_crypto and o['qty'] < 1 else f"{o['qty']:g}"
        rows += f"""
        <tr>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a">{o['time']}</td>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a"><strong>{o['symbol']}</strong></td>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a;color:{side_color}"><strong>{o['side']}</strong></td>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a">{qty_str} {unit}</td>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a">${o['price']:,.4f}</td>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a">${value:,.2f}</td>
        </tr>"""
    return rows

def positions_table(positions, is_crypto=False):
    unit = 'coins' if is_crypto else 'shares'
    if not positions:
        return f"""<tr><td colspan="6" style="padding:20px;text-align:center;color:#888">
            No open {'crypto' if is_crypto else 'stock'} positions</td></tr>"""
    rows = ""
    for p in positions:
        pl_color = '#00c853' if p['pl'] >= 0 else '#ff1744'
        pl_arrow = '▲' if p['pl'] >= 0 else '▼'
        qty_str  = f"{p['qty']:.6f}" if is_crypto and p['qty'] < 1 else f"{p['qty']:g}"
        rows += f"""
        <tr>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a"><strong>{p['symbol']}</strong></td>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a">{qty_str} {unit}</td>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a">${p['avg_price']:,.4f}</td>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a">${p['current']:,.4f}</td>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a">${p['value']:,.2f}</td>
            <td style="padding:10px 15px;border-bottom:1px solid #2a2a2a;color:{pl_color}">
                {pl_arrow} ${abs(p['pl']):,.2f} ({p['pl_pct']:+.2f}%)</td>
        </tr>"""
    return rows

def table_header(cols):
    ths = "".join(f'<th style="padding:12px 15px;text-align:left;font-size:0.8em;color:#888">{c}</th>' for c in cols)
    return f'<tr style="background:#252840">{ths}</tr>'

def build_email_html(account, stock_orders, crypto_orders, stock_positions, crypto_positions, history):
    pnl       = round(float(account.equity) - float(account.last_equity), 2)
    pnl_color = '#00c853' if pnl >= 0 else '#ff1744'
    pnl_arrow = '▲' if pnl >= 0 else '▼'
    cash      = float(account.cash)
    portfolio = float(account.portfolio_value)
    start_val = 100000.0
    total_return       = round(((portfolio - start_val) / start_val) * 100, 2)
    total_return_color = '#00c853' if total_return >= 0 else '#ff1744'

    stock_value  = sum(p['value'] for p in stock_positions)
    crypto_value = sum(p['value'] for p in crypto_positions)

    # 7-day trend
    trend_html = ""
    if len(history) >= 2:
        recent     = history[-7:]
        week_start = recent[0]['value']
        week_end   = recent[-1]['value']
        week_chg   = round(((week_end - week_start) / week_start) * 100, 2)
        week_color = '#00c853' if week_chg >= 0 else '#ff1744'
        trend_html = f"""<p style="margin:5px 0;color:#888">7-day trend:
            <span style="color:{week_color}">{'▲' if week_chg >= 0 else '▼'} {abs(week_chg):.2f}%</span></p>"""

    total_trades = len(stock_orders) + len(crypto_orders)
    total_pos    = len(stock_positions) + len(crypto_positions)

    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0;padding:0;background:#0f1117;font-family:-apple-system,sans-serif;color:#ffffff">
    <div style="max-width:650px;margin:0 auto;padding:30px 20px">

      <div style="text-align:center;margin-bottom:30px">
        <h1 style="margin:0;font-size:1.8em">🤖 Trading Bot</h1>
        <p style="margin:5px 0 0 0;color:#888">Daily Summary — {datetime.now().strftime('%A, %d %B %Y')}</p>
      </div>

      <!-- Top stats -->
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:15px;margin-bottom:20px">
        <div style="background:#1a1d2e;border-radius:12px;padding:18px;text-align:center">
          <div style="font-size:0.75em;color:#888;margin-bottom:6px">💰 PORTFOLIO</div>
          <div style="font-size:1.4em;font-weight:bold">${portfolio:,.2f}</div>
        </div>
        <div style="background:#1a1d2e;border-radius:12px;padding:18px;text-align:center">
          <div style="font-size:0.75em;color:#888;margin-bottom:6px">🎯 TODAY P&L</div>
          <div style="font-size:1.4em;font-weight:bold;color:{pnl_color}">{pnl_arrow} ${abs(pnl):,.2f}</div>
        </div>
        <div style="background:#1a1d2e;border-radius:12px;padding:18px;text-align:center">
          <div style="font-size:0.75em;color:#888;margin-bottom:6px">📈 TOTAL RETURN</div>
          <div style="font-size:1.4em;font-weight:bold;color:{total_return_color}">{total_return:+.2f}%</div>
        </div>
      </div>

      <!-- Breakdown -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-bottom:20px">
        <div style="background:#1a1d2e;border-radius:12px;padding:18px">
          <div style="font-size:0.75em;color:#888;margin-bottom:8px">📈 STOCKS</div>
          <div style="font-size:1.2em;font-weight:bold;color:#448aff">${stock_value:,.2f}</div>
          <div style="font-size:0.8em;color:#888;margin-top:4px">{len(stock_positions)} positions &bull; {len(stock_orders)} trades today</div>
        </div>
        <div style="background:#1a1d2e;border-radius:12px;padding:18px">
          <div style="font-size:0.75em;color:#888;margin-bottom:8px">🪙 CRYPTO</div>
          <div style="font-size:1.2em;font-weight:bold;color:#ffd600">${crypto_value:,.2f}</div>
          <div style="font-size:0.8em;color:#888;margin-top:4px">{len(crypto_positions)} positions &bull; {len(crypto_orders)} trades today</div>
        </div>
      </div>

      <!-- Summary row -->
      <div style="background:#1a1d2e;border-radius:12px;padding:18px;margin-bottom:30px">
        <p style="margin:5px 0;color:#888">Cash available: <span style="color:#fff;font-weight:bold">${cash:,.2f}</span></p>
        <p style="margin:5px 0;color:#888">Open positions: <span style="color:#fff;font-weight:bold">{total_pos}</span></p>
        <p style="margin:5px 0;color:#888">Total trades today: <span style="color:#fff;font-weight:bold">{total_trades}</span></p>
        {trend_html}
      </div>

      <!-- Stock trades -->
      <h2 style="font-size:1.1em;margin:0 0 15px 0">📋 Stock Trades Today</h2>
      <table style="width:100%;border-collapse:collapse;background:#1a1d2e;border-radius:12px;overflow:hidden;margin-bottom:30px">
        {table_header(['TIME','STOCK','ACTION','QTY','PRICE','VALUE'])}
        {orders_table(stock_orders, is_crypto=False)}
      </table>

      <!-- Crypto trades -->
      <h2 style="font-size:1.1em;margin:0 0 15px 0">🪙 Crypto Trades Today</h2>
      <table style="width:100%;border-collapse:collapse;background:#1a1d2e;border-radius:12px;overflow:hidden;margin-bottom:30px">
        {table_header(['TIME','COIN','ACTION','QTY','PRICE','VALUE'])}
        {orders_table(crypto_orders, is_crypto=True)}
      </table>

      <!-- Stock positions -->
      <h2 style="font-size:1.1em;margin:0 0 15px 0">📊 Open Stock Positions</h2>
      <table style="width:100%;border-collapse:collapse;background:#1a1d2e;border-radius:12px;overflow:hidden;margin-bottom:30px">
        {table_header(['STOCK','QTY','BOUGHT AT','CURRENT','VALUE','UNREALISED P&L'])}
        {positions_table(stock_positions, is_crypto=False)}
      </table>

      <!-- Crypto positions -->
      <h2 style="font-size:1.1em;margin:0 0 15px 0">🪙 Open Crypto Positions</h2>
      <table style="width:100%;border-collapse:collapse;background:#1a1d2e;border-radius:12px;overflow:hidden;margin-bottom:30px">
        {table_header(['COIN','QTY','BOUGHT AT','CURRENT','VALUE','UNREALISED P&L'])}
        {positions_table(crypto_positions, is_crypto=True)}
      </table>

      <div style="text-align:center;color:#555;font-size:0.8em;margin-top:20px">
        <p>🤖 Raspberry Pi Trading Bot — Paper Trading Only</p>
        <p>Generated at {datetime.now().strftime('%H:%M:%S')} on {datetime.now().strftime('%d/%m/%Y')}</p>
      </div>

    </div>
    </body>
    </html>
    """
    return html

def send_email(html, subject):
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = ENV['GMAIL_USER']
    msg['To']      = ENV['GMAIL_TO']
    msg.attach(MIMEText(html, 'html'))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(ENV['GMAIL_USER'], ENV['GMAIL_APP_PASSWORD'])
        server.sendmail(ENV['GMAIL_USER'], ENV['GMAIL_TO'], msg.as_string())

def run():
    print(f"📧 Generating daily summary email...")
    account                              = api.get_account()
    stock_orders, crypto_orders          = get_days_orders()
    stock_positions, crypto_positions    = get_positions()
    history                              = get_portfolio_history()
    pnl                                  = round(float(account.equity) - float(account.last_equity), 2)
    pnl_str                              = f"+${pnl:,.2f}" if pnl >= 0 else f"-${abs(pnl):,.2f}"
    subject = f"🤖 Trading Bot — {datetime.now().strftime('%d %b %Y')} — P&L: {pnl_str}"
    html    = build_email_html(account, stock_orders, crypto_orders, stock_positions, crypto_positions, history)
    send_email(html, subject)
    print(f"✅ Email sent to {ENV['GMAIL_TO']}")
    print(f"   Trades today — stocks: {len(stock_orders)}, crypto: {len(crypto_orders)}")
    print(f"   Open positions — stocks: {len(stock_positions)}, crypto: {len(crypto_positions)}")

if __name__ == "__main__":
    run()
