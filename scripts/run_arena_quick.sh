#!/bin/bash
# Strategy Arena v3 - Quick Test Run
# Fast validation run (5 generations, 20 population, ~5 minutes)

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Strategy Arena v3 - Quick Test Run                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "🚀 Quick Test Configuration:"
echo "  Universe: oil"
echo "  Date Range: 2024-01-01 to present"
echo "  Population: 20"
echo "  Generations: 5"
echo "  Timeframe: weekly"
echo "  Parallel Workers: Auto-detect"
echo ""
echo "⏱️  Expected Runtime: ~5 minutes"
echo ""

python3 arena_runner_v3.py \
    --universe oil \
    --start 2024-01-01 \
    --population 20 \
    --generations 5 \
    --timeframe weekly \
    --train-months 6 \
    --test-months 2 \
    --step-months 2 \
    --parallel-workers -1 \
    --enable-smc \
    --enable-sr \
    --enable-oil \
    --use-stops \
    --use-kelly \
    --use-calmar-fitness \
    --rebalance-threshold 0.20

echo ""
echo "✅ Quick test complete!"
echo ""
echo "📊 View results:"
echo "   python3 ui/strategy_picker.py"
echo ""
echo "🚀 Run full evolution:"
echo "   bash run_arena_oil.sh"
echo ""
