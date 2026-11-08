#!/bin/bash
# Prefixes each line of a cron job's output with the time it was written.
# Usage in crontab:  python3 job.py 2>&1 | /home/jv/logts.sh >> job.log
exec awk '{ print strftime("%Y-%m-%d %H:%M:%S"), $0; fflush() }'
