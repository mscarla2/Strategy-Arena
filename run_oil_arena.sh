#!/bin/bash
# Quick test run for oil-focused arena with optimized parameters

echo "🛢️  Starting Oil-Focused Strategy Arena..."
echo "================================================"
echo ""
echo "Configuration:"
echo "  - Stocks: EONR, TPET, USEG, STAK, PRSO, BATL, USO, BNO"
echo "  - Date Range: 2025-01-01 to present"
echo "  - Walk-Forward: 6 train / 2 test / 1 step months"
echo "  - Population: 20 (quick test)"
echo "  - Generations: 5 (quick test)"
echo ""

python arena_runner_v3.py \
    --tickers oil \
    --start 2025-01-01 \
    --population 20 \
    --generations 5 \
    --train-months 6 \
    --test-months 2 \
    --step-months 1

echo ""
echo "✅ Oil arena run complete!"
echo ""
echo "Next steps:"
echo "  1. Check results in data/gp_strategies.db"
echo "  2. Run dashboard: python -m ui.dashboard"
echo "  3. For full run, increase --population to 100 and --generations to 20+"
