#!/bin/bash

cd /usr/local/google/home/marcoscarlata/strat/Strategy-Arena

mkdir -p side_by_side_backtest/reports

echo "Starting Card Strategy Backtest..."
python3 -m side_by_side_backtest.run_today --mode card --all --use-atr > side_by_side_backtest/reports/card_out.txt &
PID1=$!

echo "Starting SxS Strategy Backtest..."
python3 -m side_by_side_backtest.run_today --mode sbs --all --use-atr > side_by_side_backtest/reports/sbs_out.txt &
PID2=$!

echo "Waiting for both to finish..."
wait $PID1
wait $PID2

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Card Strategy:"
grep "Total trades:" side_by_side_backtest/reports/card_out.txt
echo ""
echo "SxS Strategy:"
grep "Total trades:" side_by_side_backtest/reports/sbs_out.txt

rm side_by_side_backtest/reports/card_out.txt side_by_side_backtest/reports/sbs_out.txt
