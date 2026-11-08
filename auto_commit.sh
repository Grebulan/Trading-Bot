#!/bin/bash
# Daily auto-commit of trading bot source files, then push to GitHub.
# Writes one dated line per step to auto_commit.log; weekly_review.py flags lines containing "ERROR".
cd /home/jv || exit 1
LOG=/home/jv/auto_commit.log
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

# Each file separately: one missing file would otherwise make git add skip them all
for f in crypto_features.json crypto_sentiment.py crypto_trader.py dashboard.py features.json finbert_service.py performance_log.py retrain.py sentiment.py trader.py weekly_review.py trade_tally.py CLAUDE.md auto_commit.sh README.md record_milestone.py run_trader_et.sh logts.sh auto_adjust.py lab.py lab_data.py lab_engine.py champion_trader.py champion.json weekly_email.py bot_agent.py; do
    [ -e "$f" ] && git add "$f"
done

# Only commit if there are staged changes
if ! git diff --cached --quiet; then
    git commit -q -m "Auto-commit: $(date '+%Y-%m-%d %H:%M')" && log "committed $(git rev-parse --short HEAD)"
else
    log "nothing to commit"
fi

# Push anything GitHub doesn't have yet (also catches up after an earlier failed push).
# GIT_TERMINAL_PROMPT=0 makes a bad/expired token fail instead of waiting for a password.
if [ -n "$(git log origin/master..master --oneline 2>/dev/null)" ]; then
    if out=$(GIT_TERMINAL_PROMPT=0 git push -q origin master 2>&1); then
        log "pushed to GitHub"
    else
        log "ERROR push failed (GitHub token expired?): $(echo "$out" | tail -1)"
    fi
fi
