#!/bin/bash
# Strategy Arena v3 - General Stocks Evolution
# Optimized for large-cap tech stocks (no oil features)

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Strategy Arena v3 - General Stocks Evolution               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📈 General Stocks Configuration:"
echo "  Tickers: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA"
echo "  Date Range: 2020-01-01 to present"
echo "  Population: 50"
echo "  Generations: 30"
echo "  Timeframe: weekly"
echo "  Walk-Forward: 12mo train / 3mo test / 3mo step"
echo "  Parallel Workers: Auto-detect"
echo ""
echo "🎯 Advanced Features:"
echo "  ✓ Smart Money Concepts"
echo "  ✓ Support/Resistance"
echo "  ✓ Regime Detection"
echo ""
echo "🛡️ Risk Management:"
echo "  ✓ Trailing Volatility Stops"
echo "  ✓ Kelly Criterion Sizing"
echo "  ✓ Calmar Ratio Fitness"
echo "  ✓ Partial Rebalancing (20%)"
echo ""
echo "⏱️  Expected Runtime: 30-60 minutes"
echo ""

# Confirmation prompt
read -p "Start general stocks evolution? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🚀 Starting evolution..."
echo ""

python3 arena_runner_v3.py \
    --tickers AAPL MSFT GOOGL AMZN NVDA META TSLA \
    --start 2020-01-01 \
    --population 50 \
    --generations 30 \
    --timeframe weekly \
    --max-depth 5 \
    --train-months 12 \
    --test-months 3 \
    --step-months 3 \
    --parallel-workers -1 \
    --enable-smc \
    --enable-sr \
    --enable-regime \
    --use-stops \
    --use-kelly \
    --use-calmar-fitness \
    --rebalance-threshold 0.20

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ General Stocks Evolution Complete!                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Next Steps:"
echo "  1. View results: python3 ui/strategy_picker.py"
echo "  2. Dashboard: streamlit run ui/dashboard.py"
echo "  3. Database: sqlite3 data/gp_strategies.db"
echo ""
