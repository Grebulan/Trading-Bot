# Raspberry Pi Trading Bot

An algorithmic trading bot that runs on a Raspberry Pi 5. It trades US stocks and crypto through
[Alpaca's](https://alpaca.markets) **paper-trading** API, using machine-learning models (XGBoost) to
pick buys and parking spare cash in the S&P 500 ETF (SPY).

> **No real money.** This is a paper-trading experiment that started in March 2026 with $100,000 of
> simulated capital. Nothing here is investment advice.

## Status (November 2026)

So far the bot has not beaten simply holding SPY:

| | Since March 2026 |
|---|---|
| Bot | −5.7% |
| SPY | +10.2% |
| Sharpe ratio | −1.78 (target: above 1.0) |

- **Stock buys are paused** (since 2026-11-08). The ML stock picks finished $8,900 behind leaving the
  same money in SPY, and only 1 of 12 closed trades beat SPY. The bot still logs the buys it *would*
  make, which gives a free forward test of the model.
- **Crypto buys are paused** (since 2026-10-04). Tested on recent data, the crypto model's picks are no
  better than chance.
- **A more honest model test.** The weekly retrain now replays the test period as real trades, under
  the bot's own exit rules, and compares them with SPY and with random buys of the same stocks. On that
  test the stock model adds about +1.9% per trade (±1.0%) over random buys. That falls just short of
  the "has edge" bar.
- **It now evolves by itself.** Each week a strategy lab invents and tests new strategy variants:
  different stocks and ETFs, signals, leverage and short-selling. Promising ones are "shadow-traded"
  (tracked without risking the account), and one that beats both SPY and the live bot over 8+ weeks
  takes over the account. See [Strategy lab](#strategy-lab).
- **Real money:** 2027 at the earliest. To get there, the bot must beat SPY over the period, show a
  positive trade tally and reach a Sharpe ratio above 1.0.

## How it works

1. **Signals.** An XGBoost model scores 24 stocks every trading hour. It is trained on 15 years of
   daily data with technical, sector and market features (VIX, SPY and QQQ trends). It predicts
   whether a stock will rise +20% before it falls −12% within 60 trading days. It buys at ≥60%
   confidence, and only when SPY is above its 200-day average, VIX is ≤28 and recent news isn't
   negative. News sentiment comes from FinBERT and VADER.
2. **Position sizing.** Each stock buy is 8% of equity, with at most 6 stocks at a time. If there isn't
   enough cash, the bot sells just enough SPY to fund the buy.
3. **Exits.** A 12% trailing stop from the stock's peak, or a +20% take-profit. The model only predicts
   buys, so there are no ML sell signals.
4. **SPY default.** Spare cash sits in SPY. SPY is sold on a 5% trailing stop, if VIX goes above 40, or
   if SPY falls below its 200-day average.
5. **Crypto.** The same approach on hourly data for 10 coins. Buys are currently paused.

## Strategy lab

The bot is free to take big risks on paper so it can learn. The lab is built so that it learns from
real results, not from luck or hindsight:

1. **Menu.** Strategies are built from blocks: universe (S&P 500, S&P 500 tech, S&P 400 mid-caps,
   sector/asset ETFs, leveraged ETFs), signal (several momentum, reversal, 52-week-high and
   low-volatility variants), number of holdings, how often to rebalance, weighting, leverage (up to
   2×), share of the book that is short, a market-regime filter and a trailing stop.
2. **Weekly search.** 24 new combinations a week. Some are tweaks of the best so far; the rest are
   sampled using what has worked before.
3. **Two backtests and a fair baseline.** Each candidate is tested on 2015–2022 and again on
   2023–now. It must beat SPY and random picks from the same stocks in both. A stock only counts
   from the date it joined the S&P 500, so the test can't pick tomorrow's winners with hindsight.
4. **Shadow trading.** Survivors are tracked forward from the day they were chosen, using prices
   that didn't exist when they were picked. That is the honest test.
5. **Promotion.** A challenger with 8+ weeks of forward record that beats SPY and beats the real
   account by 2% becomes the champion and trades the account. A champion that falls 25% behind SPY
   is replaced by plain SPY until something better earns the slot.
6. **Lessons.** Every promotion and retirement updates a score for each building block, which steers
   the next week's search.

## Files

| File | Purpose |
|---|---|
| `trader.py` | Stock trading run: signals, buys, stops, SPY default |
| `crypto_trader.py` | Crypto trading run |
| `retrain.py` | Weekly model retrain and evaluation (writes `model_info.json`) |
| `sentiment.py`, `crypto_sentiment.py`, `finbert_service.py` | News sentiment scoring |
| `dashboard.py` | Flask web dashboard (port 5000, password login) |
| `trade_tally.py` | Every ML stock trade vs leaving the money in SPY, with exit reasons |
| `weekly_review.py` | Sunday review: gathers facts and asks Claude for a plain-English assessment |
| `record_milestone.py` | Weekly portfolio snapshot (used for the Sharpe ratio) |
| `weekly_email.py` | Sunday summary email: performance, review verdict, champion and challengers |
| `record_portfolio.py`, `performance_log.py` | Portfolio history, signal outcomes |
| `run_trader_et.sh` | Runs `trader.py` at fixed US Eastern times, despite UK/US clock changes |
| `logts.sh` | Adds timestamps to cron log output |
| `auto_commit.sh` | Nightly commit and push of the source files to this repo |
| `lab.py`, `lab_engine.py`, `lab_data.py` | Strategy lab: search, backtests, shadow trading, promotion |
| `champion_trader.py` | Trades the current champion strategy |
| `champion.json` | Which strategy is champion |
| `auto_adjust.py` | Applies the weekly review's risk-reducing setting changes to the original bot |
| `CLAUDE.md` | Detailed project notes and change history |

## Schedule (cron, UK time on the Pi)

| When | Job |
|---|---|
| Hourly 09:05–15:05 ET, plus 15:55 ET | `trader.py` (via `run_trader_et.sh`) |
| Every 2 hours at :07 | `crypto_trader.py` |
| Hourly | News sentiment refresh |
| Every 30 min | `record_portfolio.py` |
| Daily 23:00 | `auto_commit.sh` |
| Sunday 02:15 | `retrain.py` |
| Sunday 04:00 | `record_milestone.py` |
| Sunday 03:00 | `lab.py` (strategy lab) |
| Sunday 04:30 | `weekly_review.py` |
| Sunday 05:00 | `auto_adjust.py` |
| Sunday 05:15 | `weekly_email.py` |
| Weekdays 15:30 ET | `champion_trader.py` |

## Setup

Requires Python 3.11 and these packages: `alpaca-trade-api`, `yfinance`, `xgboost`, `scikit-learn`,
`pandas`, `numpy`, `joblib`, `flask`, `newsapi-python`, `vaderSentiment`, `transformers`, `torch`,
`anthropic`, `requests`.

Keys go in `~/.env_trading` (never committed):

```
ALPACA_KEY=...
ALPACA_SECRET=...
NEWS_KEY=...              # NewsAPI
CRYPTOPANIC_KEY=...
GMAIL_USER=...
GMAIL_APP_PASSWORD=...
GMAIL_TO=...
DASHBOARD_PASSWORD=...
DASHBOARD_SECRET_KEY=...
ANTHROPIC_API_KEY=...     # weekly review only
```

Run `python3 retrain.py` once to build the models, then add the cron jobs above. File paths in the
scripts assume the code lives in `/home/jv`.

## Lessons so far

- **Check accuracy against a baseline.** The crypto model once reported 80% accuracy. Tested properly,
  it scores below simply always guessing "no".
- **Test on the future, not on other tickers.** An early split tested on held-out *stocks*, not on
  later *dates*, which made the models look far better than they were.
- **Count trades, not flagged days.** The first stock test counted every day a stock was flagged as a
  separate pick, so 1,000 "picks" were really about 130 trades. Scored as real trades, the claimed
  edge halved.
- **Beware a hindsight-picked watchlist.** Random buys of the same 24 stocks also beat SPY, because the
  list was chosen in 2026 from stocks that had already done well.
- **Compare everything with SPY.** It's the bar the bot has to clear, and so far it hasn't.
