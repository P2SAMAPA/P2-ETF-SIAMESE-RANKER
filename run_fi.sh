#!/usr/bin/env bash
# run_fi.sh — FI / Commodities module only
# Usage:
#   bash run_fi.sh
#   bash run_fi.sh --lgbm
#   bash run_fi.sh --no-backtest

set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================"
echo " P2 ETF Siamese Ranker — FI Module"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"

[ -f .env ] && export $(grep -v '^#' .env | xargs)
mkdir -p outputs/models outputs/backtest outputs/rankings outputs/features data

echo "▶ Training FI..."
python fi_train.py "$@"

echo "▶ Predicting FI..."
python fi_predict.py "$@"

echo "======================================"
echo " FI complete — $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"
