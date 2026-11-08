# Trading Bot — Project Guide

## What this is
An algorithmic trading bot running on a Raspberry Pi. Uses the **Alpaca paper trading API** (`https://paper-api.alpaca.markets`). Started March 2026 with $100,000 paper capital. Not yet running real money. The September 2026 review was pushed back on 2026-11-08: real money is **2027 at the earliest**. To pass, the bot must beat SPY over the period, have a positive trade tally (`trade_tally.py`) and reach a Sharpe ratio above 1.0.

## Key files
| File | Purpose |
|---|---|
| `trader.py` | Main trading logic — runs on cron |
| `dashboard.py` | Flask web dashboard on port 5000 |
| `bot_agent.py` | Sunday 05:20: implements the weekly review's recommendations by editing the bot's code (see Self-improvement agent). History in `agent_history.json` |
| `weekly_email.py` | Sunday 06:30 email: performance vs SPY, review verdict, champion, challengers' shadow records, lab activity, positions. `--preview` writes HTML instead of sending |
| `email_summary.py` | Old daily email, retired 2026-11-08 (cron line commented out) |
| `record_milestone.py` | Weekly Sunday portfolio snapshot |
| `weekly_review.py` | Sunday review agent — gathers facts, asks Claude (needs `ANTHROPIC_API_KEY`), writes `weekly_reviews.json` for `/review`. Advisory only |
| `auto_adjust.py` | Sunday 5am, after the review: applies the review's risk-reducing `setting_changes` to `trader.py`, and restarts stock buys only when fixed evidence rules pass. Commits, pushes and emails every change. State in `auto_adjust_state.json`. `--dry-run` to preview |
| `lab.py`, `lab_engine.py`, `lab_data.py` | Strategy lab: weekly search, backtests, shadow trading, promotion (see below) |
| `champion_trader.py` | Trades the lab's champion strategy (leverage/shorts allowed) |
| `champion.json`, `lab_state.json` | Current champion; challengers, everything tested, lessons |
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
- **Auto-adjust:** Sunday 5:00am (`auto_adjust.py`, logs to `auto_adjust.log`)
- **Self-improvement agent:** Sunday 5:20am (`bot_agent.py`, `agent.log`)
- **Weekly email:** Sunday 6:30am (`weekly_email.py`, `email.log`). The daily 10pm email was retired 2026-11-08
- **Strategy lab:** Sunday 3:00am (`lab.py`, `lab.log`)
- **Champion trader:** weekdays 15:30 ET via `run_trader_et.sh` (`champion_trader.py`, `champion.log`). Does nothing while `legacy_ml` is champion
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

## Self-improvement agent (2026-11-08)
The owner delegated decisions to the bot: they are not a trader and want it to act on its own recommendations. `bot_agent.py` runs Claude (Opus 5.5) through the API with its own narrow tools: read code, data and logs; edit `EDITABLE` files; create new `.py` files; `run_backtest`; `run_checks`; `finish`.
- **Protected, the scorekeeping:** `weekly_review.py`, `trade_tally.py`, `lab.py` (promotion rules), `auto_adjust.py`, `weekly_email.py`, `bot_agent.py`, the shell scripts and `champion.json`. These stop the bot grading its own homework.
- **Protected lines:** the paper-trading URLs, backtest costs, the 1.95× leverage cap and the `has_edge` bar.
- **Off limits:** it can never read `~/.env_trading`.
- **Before anything is kept,** compile, new pylint errors, imports of `trader`/`crypto_trader`, and dry runs of lab, champion (current champion and a leveraged long/short test strategy) and auto-adjust must all pass. Otherwise every change is reverted from `.agent_backup/`.
- **Passing changes** are committed and pushed with the agent's message and reported in the weekly email. Anything it thinks needs a protected change goes in its `not_done` list for the owner.
- **Budget:** 60 turns and $10 per run. The first dry run cost $0.49.
- `python3 bot_agent.py --dry-run` does the work, runs the checks, then reverts everything.

## Strategy lab: champion vs challengers (2026-11-08)
The owner wants the bot to evolve by itself on paper, taking as much risk as it likes, including leverage and shorts. Before any real money, the best version found is frozen for ~3 months as a final untouched test.
- **`lab_data.py`:** universes are the S&P 500 (a stock is eligible only after its index join date, which limits hindsight), S&P 500 tech, S&P 400, ETF rotation, leveraged ETFs, and `legacy24`. `legacy24` is hindsight-biased and can never be promoted. Prices are cached in `lab_prices.pkl`.
- **`lab_engine.py`:** the `MENU` of building blocks (universe, signal, top_k, rebalance, weighting, leverage up to 2×, shorts, SPY regime, trailing stop) and the daily backtester. It charges costs, margin interest and borrow fees. The same `weights_at` drives backtests, shadow records and live trading.
- **`lab.py`, weekly:** generates 24 candidates (mutations of the best, plus samples weighted by the lessons) and backtests each on in-sample 2015–2022 and out-of-sample 2023–now, against random picks from the same universe. A candidate must beat SPY and random in both periods to become a challenger (max 10). Challengers shadow-trade from the day after acceptance.
- **Promotion:** after ≥40 trading days, a challenger must beat SPY and beat the real account by 2% over the same window. At most one promotion per 28 days.
- **Demotion and retirement:** a champion 25% behind SPY is demoted to `spy_hold`. Challengers retire after 80 days without promotion, or after 30 days if 10% behind SPY. Results feed `lessons` in `lab_state.json`.
- **`champion.json`:** the active champion. `legacy_ml` means `trader.py` trades as before; `spy_hold` means all SPY; anything else, `champion_trader.py` trades it and `trader.py` stands aside. Promotions are committed, pushed and emailed. To go back by hand, set `id` to `legacy_ml`.
- `python3 lab.py --dry-run`, `python3 champion_trader.py --dry-run` to preview.

## Bounded autonomy (2026-11-08)
The bot can change a few of its own settings in `trader.py`, through `auto_adjust.py`.
- **Tightening, at the weekly review's request:** pause stock buys, switch off the stop override, cut `MAX_POSITIONS` by one (minimum 3). The review's schema has no way to ask for anything else.
- **Restarting stock buys, by fixed rules only:** paused ≥8 weeks; ≥20 forward-test trades from `buys_paused` signals, ≥10 of them finished; those trades ahead of SPY in total; `has_edge` true in `model_info.json`; this week's review not red; no change to buys in the last 28 days.
- **Never automatic:** re-enabling the stop override or raising `MAX_POSITIONS`.
- **Every change** is committed and pushed, recorded in `auto_adjust_state.json` and emailed with an undo line.
- **Suggest-only** if `AUTONOMY = False` in `auto_adjust.py` or `trader.py` isn't on the paper API. Turn `AUTONOMY` off before going live.
- If you edit a setting by hand, the guardrails still apply. They read the setting's current value from `trader.py`.

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
