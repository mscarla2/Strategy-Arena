#!/bin/bash
# Strategy Arena v3 - Oil Stocks Evolution
# Optimized for oil & gas stocks with all oil-specific features

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Strategy Arena v3 - Oil Stocks Evolution                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🛢️  Oil Stocks Configuration:"
echo "  Universe: oil (EONR, TPET, USEG, STAK, PRSO, BATL, USO, BNO)"
echo "  Date Range: 2025-01-01 to present (March 17, 2026)"
echo "  Population: 50"
echo "  Generations: 30"
echo "  Timeframe: swing"
echo ""
echo "🎯 Oil-Specific Features:"
echo "  ✓ WTI/Brent Correlation & Beta"
echo "  ✓ EIA Inventory Analysis"
echo "  ✓ Crack Spreads (3-2-1, 5-3-2)"
echo "  ✓ Seasonal Patterns"
echo "  ✓ WTI-Brent Spread Dynamics"
echo ""
echo "🎯 Additional Features:"
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
read -p "Start oil stocks evolution? (y/n) " -n 1 -r
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
    --universe oil \
    --start 2018-01-01 \
    --population 50 \
    --generations 30 \
    --timeframe swing \
    --train-months 3 \
    --test-months 2 \
    --step-months 1 \
    --enable-oil \
    --enable-smc \
    --enable-sr \
    --enable-regime \
    --use-stops \
    --use-kelly \
    --use-calmar-fitness \
    --rebalance-threshold 0.20

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Oil Stocks Evolution Complete!                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Next Steps:"
echo "  1. View results: python3 ui/strategy_picker.py"
echo "  2. Dashboard: streamlit run ui/dashboard.py"
echo "  3. Database: sqlite3 data/gp_strategies.db"
echo ""
