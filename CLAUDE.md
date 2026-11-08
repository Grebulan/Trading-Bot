# Trading Bot — Project Guide

## What this is
An algorithmic trading bot running on a Raspberry Pi. Uses the **Alpaca paper trading API** (`https://paper-api.alpaca.markets`). Started March 2026 with $100,000 paper capital. Not yet running real money. The September 2026 review was pushed back on 2026-11-08: real money is **2027 at the earliest**. To pass, the bot must beat SPY over the period, have a positive trade tally (`trade_tally.py`) and reach a Sharpe ratio above 1.0.

## Key files
| File | Purpose |
|---|---|
| `trader.py` | Main trading logic — runs on cron |
| `dashboard.py` | Flask web dashboard on port 5000 |
| `email_summary.py` | Portfolio email summary |
| `record_milestone.py` | Weekly Sunday portfolio snapshot |
| `weekly_review.py` | Sunday review agent — gathers facts, asks Claude (needs `ANTHROPIC_API_KEY`), writes `weekly_reviews.json` for `/review`. Advisory only |
| `trade_tally.py` | Every ML stock trade since 2026-10-04 vs leaving the money in SPY, with each exit reason and how many times the stop override held it (from `performance.json`). Run it directly for a table; also feeds the weekly review |
| `milestones.json` | Portfolio history (Sharpe, returns, positions) |
| `stock_highwater.json` | Peak prices for trailing stops |
| `model_info.json` | Stock ML model metadata |
| `crypto_model_info.json` | Crypto ML model metadata |
| `~/.env_trading` | API keys, `DASHBOARD_SECRET_KEY` |

## Cron schedule
- **Stocks:** 09:05–15:05 **US Eastern** hourly, plus 15:55 ET (final trade before close). Pi cron can't do timezones, so cron calls `run_trader_et.sh` at both possible UK times and the wrapper only runs `trader.py` when the ET time matches (changed 2026-10-12).
- **Crypto:** `:07` past even hours
- **Weekly milestone:** Sunday 4am (`record_milestone.py`)
- **Weekly review agent:** Sunday 4:30am (`weekly_review.py`)
- **Weekly retrain:** Sunday (runs before milestone)
- **Log timestamps (2026-10-20):** cron jobs pipe output through `logts.sh`, which adds a time to each line (`PYTHONUNBUFFERED=1` is set in crontab so the times are accurate). `retrain.log` already has its own timestamps. `weekly_review.py` only reports log errors written since the previous review. It keeps its read position in `weekly_review_logstate.json`.

## Trading strategy
- **ML models** predict BUY signals for stocks and crypto. SELL signals disabled — model only predicts BUY outcomes.
- **SPY default position:** spare cash deployed into S&P 500 ETF when no ML signals fire. Was ~97% of the portfolio by Oct 2026 because it starved stock buys of cash (fixed 2026-10-04).
- **Trailing stops:** SPY 5%, Stocks 12% (trailing from peak), Crypto 15%
- **Stock buys paused (2026-11-08):** `BUYS_PAUSED = True` in `trader.py`, after the weekly review found the ML picks were $8,900 behind SPY, with only 1 of 12 closed trades beating it. BUY signals are still logged to `performance.json` with reason `buys_paused`, so the model can be judged without risking money. Open positions still exit via the trailing stop or take-profit, and the spare cash goes into SPY. Before restarting, find out why live picks hit the target only 1 time in 12 when the backtest says ~55%.
- **Stock stop override (2026-10-20, DISABLED 2026-11-08):** if the model still says BUY at ≥60% when the 12% trailing stop is hit, the stock is held. It always sells at −30% from entry (`STOP_OVERRIDE_BACKSTOP`), and take-profit still applies. The bot logs this as `stop_override`. A walk-forward sim (2019–2026) gave +$99k vs SPY, against +$88k when the stop always sells. The gain mostly comes from avoiding stop-out-then-rebuy cycles, and the sim is flattered by a hindsight-picked ticker list. Now `STOP_OVERRIDE_CONF = None`: live, the override held VRT 8 times near −10% before it sold at −24.7%, and CAT 11 times before it sold at −15%. Set it to `0.60` to re-enable.
- **Stock hard stop:** disabled 2026-10-04 (`HARD_STOP = None`). It was added 2026-08-18 at 6% from entry, based on a 23-trade backtest. A walk-forward sim over 5,142 out-of-sample picks showed it lowered average return per trade (+6.0% vs +6.9%).
- **ML stock buys are funded from SPY (2026-10-04):** if cash is short, `fund_from_spy()` sells just enough whole SPY shares to fund an 8%-of-equity buy (`STOCK_POSITION_PCT`). The SPY high-water mark is left alone. Before this, SPY absorbed all cash and 282 BUY signals were skipped between 2026-08-25 and 2026-10-02.
- **Max 6 stock positions (2026-10-20, was 8):** `MAX_POSITIONS` in `trader.py`, which caps ML stocks at ~48% of the portfolio (6 × 8%). The count now includes only stocks. Before, crypto holdings used up stock slots too.
- **Model evaluation:** `retrain.py` scores both models on a time split and logs HAS EDGE/NO EDGE (`has_edge` in `model_info.json` / `crypto_model_info.json`). Ignore raw accuracy, which sits at or below the always-no baseline. Stocks (2026-11-08): `has_edge` comes from `live_rule_test`, which replays the test period as real trades (one position per ticker, 12% trailing stop, +20% TP, SPY/VIX filters) and compares them with SPY and with random buys of the same tickers. It needs ≥50 trades and a gain over random buys of at least 2 standard errors. First result: 154 trades, +3.6% vs SPY per trade, but random buys of the same stocks also make +1.8%, so the model adds +1.9% ±1.0%, giving NO EDGE (just short of the bar). The old label test (`pick_hit_rate` ~55% vs ~35%) counts every flagged day and overstates the edge.
- **Crypto buys paused (2026-10-04):** `BUYS_PAUSED = True` in `crypto_trader.py`. Time-split test showed no edge. Existing positions still exit via stops. `retrain.py` now evaluates crypto on recent data and logs `HAS EDGE`/`NO EDGE` (also `has_edge` in `crypto_model_info.json`); re-enable only on evidence.
- **SPY crash exits:** 5% trailing stop OR VIX > 40 OR SPY below 200-day MA (bear regime)

## Alpaca API quirks
- Fractional orders **must** use `time_in_force='day'` — GTC is rejected.
- `account.cash` does **not** reflect pending open orders — deduct manually before sizing SPY.
- SPY uses **whole shares only** (`qty = int(available / spy_price)`).

## Dashboard
- Flask app, port 5000, session-based auth (30-day cookie).
- Password and `DASHBOARD_SECRET_KEY` in `~/.env_trading`.
- Routes: `/` (main), `/milestones`, `/review`, `/login`, `/logout`
- `/monitor` was removed — do not re-add it.

## Milestones & Sharpe ratio
- `milestones.json` stores weekly snapshots. Sharpe ratio calculated from weekly portfolio values — needs ≥3 entries before it displays.
- Risk-free rate assumed: 5% annual (US Treasury).
- Target Sharpe > 1.0 before considering real money.

## Current performance (last updated 2026-11-08)
- Portfolio: **$94,262** (−5.7% since March vs SPY +10.2%); Sharpe −1.78
- ML stock trades since 2026-10-04: −$8,904 vs leaving the money in SPY (4 of 18 beat SPY)
- Stock model: picks ≥60% hit TP 55.8% vs 35.2% random in the backtest. That edge is unproven live. Crypto: no edge, buys paused
- Positions: SPY (~50%), GOLD, GOOGL, MU, PLTR, NVDA, CAT, a little BTC
