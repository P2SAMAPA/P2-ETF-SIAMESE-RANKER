#!/usr/bin/env bash
# cron_setup.sh — Install daily 22:00 cron job (Mon–Fri)
# Usage: bash cron_setup.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=$(which python3 || which python)
LOG="$SCRIPT_DIR/outputs/cron.log"

CRON_LINE="0 22 * * 1-5 cd $SCRIPT_DIR && bash run_all.sh >> $LOG 2>&1"

# Add only if not already present
( crontab -l 2>/dev/null | grep -v "P2-ETF-SIAMESE-RANKER"; echo "$CRON_LINE" ) | crontab -

echo "Cron job installed:"
echo "  $CRON_LINE"
echo ""
echo "View with:   crontab -l"
echo "Remove with: crontab -e  (delete the line)"
echo "Logs at:     $LOG"
