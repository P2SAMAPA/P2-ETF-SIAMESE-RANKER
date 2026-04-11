#!/usr/bin/env bash
# run_equity.sh — Equity Sectors module only
# Usage:
#   bash run_equity.sh
#   bash run_equity.sh --lgbm
#   bash run_equity.sh --no-backtest

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================="
echo " P2 ETF Siamese Ranker — Equity Module"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

[ -f .env ] && export $(grep -v '^#' .env | xargs)
mkdir -p outputs/models outputs/backtest outputs/rankings outputs/features data

echo "▶ Training Equity..."
python equity_train.py "$@"

echo "▶ Predicting Equity..."
python equity_predict.py "$@"

echo "========================================="
echo " Equity complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
