#!/bin/bash
# Strategy Arena v3 - Oil Stocks Evolution
# Optimized for oil & gas stocks with all oil-specific features
# Updated with RC-3 (Fitness v2), RC-4 (Expanded Universe), RC-5 (Feature Cache)

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Strategy Arena v3 - Oil Stocks Evolution (RC-3/4/5)        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🛢️  Oil Stocks Configuration:"
echo "  Universe: oil (expanded — microcaps + reference panel + benchmarks)"
echo "  Tradeable: EONR, TPET, USEG, STAK, PRSO, BATL"
echo "  Reference: XOM, CVX, COP, OXY, EOG, DVN, APA, FANG, SLB, HAL, BKR, MPC, PSX, VLO, XLE, XOP"
echo "  Benchmarks: USO, BNO, XLE (primary), XOP"
echo "  Date Range: 2018-01-01 to present"
echo "  Population: 100"
echo "  Generations: 30"
echo "  Timeframe: weekly (optimized for microcaps)"
echo "  Walk-Forward: 6mo train / 3mo test / 3mo step + anchored recent period"
echo "  Parallel Workers: Auto-detect"
echo ""
echo "🎯 RC-3: Fitness v2 Enhancements:"
echo "  ✓ Recency-weighted period scoring (half-life: 4 periods)"
echo "  ✓ Universe-adaptive drawdown penalties (oil_microcap calibration)"
echo "  ✓ Blended Sharpe + Calmar multi-objective scoring"
echo ""
echo "🎯 RC-4: Oil Universe Specialization:"
echo "  ✓ Expanded universe with 16-ticker reference panel"
echo "  ✓ Cross-sectional features computed on full panel"
echo "  ✓ Portfolio restricted to 6 tradeable microcaps"
echo "  ✓ XLE as primary benchmark (not equal-weight)"
echo ""
echo "🎯 RC-5: Feature Pre-computation:"
echo "  ✓ Features cached for all rebalance dates"
echo "  ✓ 10-50× speedup for population evaluation"
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
echo "  ✓ Partial Rebalancing (20%)"
echo ""
echo "🚀 Performance Optimizations:"
echo "  ✓ Feature Pre-computation & Caching (RC-5)"
echo "  ✓ Parallel Strategy Evaluation"
echo "  ✓ Anchored Recent Period (covers March 2026)"
echo "  ✓ Oil-Specific Benchmarks (XLE/USO/BNO/XOP)"
echo ""
echo "⏱️  Expected Runtime: 15-25 minutes (faster with feature cache)"
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
    --universe-type oil_microcap \
    --start 2018-01-01 \
    --population 100 \
    --generations 30 \
    --timeframe weekly \
    --max-depth 5 \
    --train-months 6 \
    --test-months 3 \
    --step-months 3 \
    --parallel-workers -1 \
    --enable-oil \
    --enable-smc \
    --enable-sr \
    --enable-regime \
    --use-stops \
    --use-kelly \
    --use-fitness-v2 \
    --recency-half-life 4 \
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
