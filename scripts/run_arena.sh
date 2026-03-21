#!/bin/bash
# Strategy Arena v3 - Production Evolution Run
# Optimized configuration with all advanced features

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Strategy Arena v3 - Genetic Programming Evolution          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# CONFIGURATION
# ============================================================================

# Basic Parameters
UNIVERSE="oil"
START_DATE="2013-01-01"
POPULATION=50
GENERATIONS=10
TIMEFRAME="weekly"

# Walk-Forward Configuration
TRAIN_MONTHS=12
TEST_MONTHS=3
STEP_MONTHS=3

# Evolution Parameters
MAX_DEPTH=7
TOURNAMENT=3
CROSSOVER=0.7
MUTATION=0.2
ELITE=2
PARSIMONY=0.001

# Performance
PARALLEL_WORKERS=-1  # Auto-detect (use -1), or set specific number, or 0 for sequential

# Advanced Features (comment out to disable)
ENABLE_SMC="--enable-smc"
ENABLE_SR="--enable-sr"
ENABLE_OIL="--enable-oil"
ENABLE_REGIME="--enable-regime"

# Risk Management (comment out to disable)
USE_STOPS="--use-stops"
USE_KELLY="--use-kelly"
USE_CALMAR="--use-calmar-fitness"
REBALANCE_THRESHOLD="--rebalance-threshold 0.20"

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "📊 Configuration:"
echo "  Universe: $UNIVERSE"
echo "  Date Range: $START_DATE to present"
echo "  Population: $POPULATION"
echo "  Generations: $GENERATIONS"
echo "  Timeframe: $TIMEFRAME"
echo "  Parallel Workers: $PARALLEL_WORKERS (auto-detect)"
echo ""
echo "🎯 Advanced Features:"
echo "  ✓ Smart Money Concepts (Order Blocks, FVG, Liquidity Sweeps)"
echo "  ✓ Support/Resistance (Volume Profile, Pivot Points)"
echo "  ✓ Oil-Specific Features (WTI/Brent correlation, Crack Spreads)"
echo "  ✓ Regime Detection (Volatility, Trend, Market regimes)"
echo ""
echo "🛡️ Risk Management:"
echo "  ✓ Trailing Volatility Stops (ATR-based)"
echo "  ✓ Kelly Criterion Position Sizing"
echo "  ✓ Calmar Ratio Fitness (drawdown-aware)"
echo "  ✓ Partial Rebalancing (20% threshold)"
echo ""
echo "⏱️  Expected Runtime: 30-60 minutes"
echo ""

# Confirmation prompt
read -p "Start evolution? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 0
fi

# ============================================================================
# RUN EVOLUTION
# ============================================================================

echo ""
echo "🚀 Starting evolution..."
echo ""

python3 arena_runner_v3.py \
    --universe $UNIVERSE \
    --start $START_DATE \
    --population $POPULATION \
    --generations $GENERATIONS \
    --timeframe $TIMEFRAME \
    --train-months $TRAIN_MONTHS \
    --test-months $TEST_MONTHS \
    --step-months $STEP_MONTHS \
    --max-depth $MAX_DEPTH \
    --tournament $TOURNAMENT \
    --crossover $CROSSOVER \
    --mutation $MUTATION \
    --elite $ELITE \
    --parsimony $PARSIMONY \
    --parallel-workers $PARALLEL_WORKERS \
    $ENABLE_SMC \
    $ENABLE_SR \
    $ENABLE_OIL \
    $ENABLE_REGIME \
    $USE_STOPS \
    $USE_KELLY \
    $USE_CALMAR \
    $REBALANCE_THRESHOLD

# ============================================================================
# POST-RUN INSTRUCTIONS
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Evolution Complete!                                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Next Steps:"
echo ""
echo "  1. View Results (Terminal UI):"
echo "     python3 ui/strategy_picker.py"
echo ""
echo "  2. View Dashboard (Web UI):"
echo "     streamlit run ui/dashboard.py"
echo ""
echo "  3. Check Database:"
echo "     sqlite3 data/gp_strategies.db"
echo ""
echo "  4. Run Another Evolution:"
echo "     bash run_arena_quick.sh    # Quick test (5 gen)"
echo "     bash run_arena_oil.sh       # Oil stocks (30 gen)"
echo "     bash run_arena_general.sh   # General stocks (30 gen)"
echo ""
