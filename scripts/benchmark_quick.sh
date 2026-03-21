#!/bin/bash
# Quick Performance Benchmark
# Fast comparison of sequential vs parallel (2 tests only, ~5-10 minutes)

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Strategy Arena - Quick Performance Benchmark               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "This script will run 2 quick tests:"
echo "  1. Sequential (0 workers)"
echo "  2. Parallel (auto-detect workers)"
echo ""
echo "⏱️  Total Runtime: ~5-10 minutes"
echo ""

# Confirmation
read -p "Start quick benchmark? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 0
fi

# Create results directory
RESULTS_DIR="benchmark_quick_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "📊 Results will be saved to: $RESULTS_DIR"
echo ""

# Smaller test parameters for speed
COMMON_ARGS="--universe oil --start 2024-06-01 --population 20 --generations 2 --timeframe weekly --train-months 3 --test-months 1 --step-months 1"

# ============================================================================
# TEST 1: Sequential
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Sequential (--parallel-workers 0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

START1=$(date +%s)
python3 arena_runner_v3.py \
    $COMMON_ARGS \
    --parallel-workers 0 \
    2>&1 | tee "$RESULTS_DIR/sequential.log"
END1=$(date +%s)
TIME1=$((END1 - START1))

echo ""
echo "✅ Sequential completed in ${TIME1}s"
echo ""

# ============================================================================
# TEST 2: Parallel Auto
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Parallel Auto-detect (--parallel-workers -1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

START2=$(date +%s)
python3 arena_runner_v3.py \
    $COMMON_ARGS \
    --parallel-workers -1 \
    2>&1 | tee "$RESULTS_DIR/parallel.log"
END2=$(date +%s)
TIME2=$((END2 - START2))

echo ""
echo "✅ Parallel completed in ${TIME2}s"
echo ""

# ============================================================================
# RESULTS
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Quick Benchmark Results                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Calculate speedup
SPEEDUP=$(echo "scale=2; $TIME1 / $TIME2" | bc)

echo "📊 Performance Summary:" | tee "$RESULTS_DIR/summary.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"
echo "Sequential (0 workers):  ${TIME1}s" | tee -a "$RESULTS_DIR/summary.txt"
echo "Parallel (auto-detect):  ${TIME2}s" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"
echo "Speedup: ${SPEEDUP}x" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"

# Interpretation
if (( $(echo "$SPEEDUP < 1.5" | bc -l) )); then
    echo "⚠️  Low speedup - parallel overhead may be too high for this workload" | tee -a "$RESULTS_DIR/summary.txt"
elif (( $(echo "$SPEEDUP < 2.5" | bc -l) )); then
    echo "✓  Moderate speedup - typical for Python multiprocessing" | tee -a "$RESULTS_DIR/summary.txt"
else
    echo "✅ Excellent speedup - parallel workers are very effective!" | tee -a "$RESULTS_DIR/summary.txt"
fi

echo "" | tee -a "$RESULTS_DIR/summary.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"

# Check for caching
echo "🔍 Feature Caching:" | tee -a "$RESULTS_DIR/summary.txt"
CACHE_HITS_SEQ=$(grep -c "Loaded from cache" "$RESULTS_DIR/sequential.log" 2>/dev/null || echo "0")
CACHE_HITS_PAR=$(grep -c "Loaded from cache" "$RESULTS_DIR/parallel.log" 2>/dev/null || echo "0")
echo "  Sequential: $CACHE_HITS_SEQ cache hits" | tee -a "$RESULTS_DIR/summary.txt"
echo "  Parallel:   $CACHE_HITS_PAR cache hits" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"

# Worker count detection
WORKERS=$(grep "parallel workers" "$RESULTS_DIR/parallel.log" | head -1 | grep -oE '[0-9]+' | head -1 || echo "unknown")
echo "💡 Detected Workers: $WORKERS" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"

echo "📁 Detailed logs: $RESULTS_DIR/" | tee -a "$RESULTS_DIR/summary.txt"
echo ""
echo "✅ Quick benchmark complete!"
echo ""
