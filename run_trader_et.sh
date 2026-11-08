#!/bin/bash
# Runs trader.py only at fixed US Eastern times, so the schedule follows the
# NYSE across UK/US daylight-saving mismatches. Cron calls this at both
# possible UK times (ET+4 and ET+5); only the one matching ET actually runs.
NOW_ET=$(TZ=America/New_York date +%H:%M)
case "$NOW_ET" in
  09:05|10:05|11:05|12:05|13:05|14:05|15:05|15:55)
    exec python3 /home/jv/trader.py ;;
esac
