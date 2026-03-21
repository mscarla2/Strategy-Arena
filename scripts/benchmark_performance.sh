#!/bin/bash
# Performance Benchmarking Script
# Tests parallel workers effectiveness and feature caching impact

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Strategy Arena - Performance Benchmark                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "This script will run multiple tests to measure:"
echo "  1. Sequential vs Parallel performance"
echo "  2. Speedup from parallel workers"
echo "  3. Feature caching effectiveness (if implemented)"
echo ""
echo "⏱️  Total Runtime: ~15-20 minutes"
echo ""

# Confirmation
read -p "Start benchmark? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 0
fi

# Create results directory
RESULTS_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo ""
echo "📊 Results will be saved to: $RESULTS_DIR"
echo ""

# Common parameters for all tests
COMMON_ARGS="--universe oil --start 2024-01-01 --population 30 --generations 3 --timeframe weekly --train-months 6 --test-months 2 --step-months 2"

# ============================================================================
# TEST 1: Sequential (0 workers)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Sequential Evaluation (--parallel-workers 0)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

time python3 arena_runner_v3.py \
    $COMMON_ARGS \
    --parallel-workers 0 \
    2>&1 | tee "$RESULTS_DIR/test1_sequential.log"

echo ""
echo "✅ Test 1 complete"
echo ""

# ============================================================================
# TEST 2: Parallel with 2 workers
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Parallel Evaluation (--parallel-workers 2)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

time python3 arena_runner_v3.py \
    $COMMON_ARGS \
    --parallel-workers 2 \
    2>&1 | tee "$RESULTS_DIR/test2_parallel_2.log"

echo ""
echo "✅ Test 2 complete"
echo ""

# ============================================================================
# TEST 3: Parallel with 4 workers
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Parallel Evaluation (--parallel-workers 4)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

time python3 arena_runner_v3.py \
    $COMMON_ARGS \
    --parallel-workers 4 \
    2>&1 | tee "$RESULTS_DIR/test3_parallel_4.log"

echo ""
echo "✅ Test 3 complete"
echo ""

# ============================================================================
# TEST 4: Auto-detect workers
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: Auto-detect Workers (--parallel-workers -1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

time python3 arena_runner_v3.py \
    $COMMON_ARGS \
    --parallel-workers -1 \
    2>&1 | tee "$RESULTS_DIR/test4_parallel_auto.log"

echo ""
echo "✅ Test 4 complete"
echo ""

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Benchmark Complete - Analyzing Results                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Extract timing information
echo "📊 Performance Summary:" | tee "$RESULTS_DIR/summary.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"

for test in test1_sequential test2_parallel_2 test3_parallel_4 test4_parallel_auto; do
    if [ -f "$RESULTS_DIR/${test}.log" ]; then
        # Extract real time from the log
        REAL_TIME=$(grep "^real" "$RESULTS_DIR/${test}.log" | tail -1 || echo "N/A")
        USER_TIME=$(grep "^user" "$RESULTS_DIR/${test}.log" | tail -1 || echo "N/A")
        SYS_TIME=$(grep "^sys" "$RESULTS_DIR/${test}.log" | tail -1 || echo "N/A")
        
        echo "Test: $test" | tee -a "$RESULTS_DIR/summary.txt"
        echo "  Real: $REAL_TIME" | tee -a "$RESULTS_DIR/summary.txt"
        echo "  User: $USER_TIME" | tee -a "$RESULTS_DIR/summary.txt"
        echo "  Sys:  $SYS_TIME" | tee -a "$RESULTS_DIR/summary.txt"
        echo "" | tee -a "$RESULTS_DIR/summary.txt"
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"
echo "📁 Detailed logs saved to: $RESULTS_DIR/" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"
echo "💡 Analysis Tips:" | tee -a "$RESULTS_DIR/summary.txt"
echo "  - Compare 'real' time to see wall-clock speedup" | tee -a "$RESULTS_DIR/summary.txt"
echo "  - 'user' time shows total CPU time used" | tee -a "$RESULTS_DIR/summary.txt"
echo "  - Ideal speedup: real_time(sequential) / real_time(parallel)" | tee -a "$RESULTS_DIR/summary.txt"
echo "  - Check for 'Loaded from cache' messages in logs" | tee -a "$RESULTS_DIR/summary.txt"
echo "" | tee -a "$RESULTS_DIR/summary.txt"

# Create a Python analysis script
cat > "$RESULTS_DIR/analyze.py" << 'EOF'
#!/usr/bin/env python3
"""Analyze benchmark results and calculate speedup metrics."""

import re
import os
from pathlib import Path

def parse_time(time_str):
    """Parse time string like '1m23.456s' to seconds."""
    if 'N/A' in time_str:
        return None
    
    match = re.search(r'(\d+)m([\d.]+)s', time_str)
    if match:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    
    match = re.search(r'([\d.]+)s', time_str)
    if match:
        return float(match.group(1))
    
    return None

def main():
    results_dir = Path(__file__).parent
    
    tests = {
        'Sequential (0 workers)': 'test1_sequential.log',
        'Parallel (2 workers)': 'test2_parallel_2.log',
        'Parallel (4 workers)': 'test3_parallel_4.log',
        'Auto-detect': 'test4_parallel_auto.log',
    }
    
    times = {}
    
    for name, logfile in tests.items():
        logpath = results_dir / logfile
        if logpath.exists():
            with open(logpath) as f:
                content = f.read()
                # Look for time output
                match = re.search(r'real\s+(\d+m[\d.]+s)', content)
                if match:
                    times[name] = parse_time(match.group(1))
    
    if not times:
        print("❌ Could not parse timing information")
        return
    
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70 + "\n")
    
    # Print times
    print("Execution Times:")
    print("-" * 70)
    for name, time_sec in times.items():
        if time_sec:
            print(f"  {name:30s}: {time_sec:7.2f}s ({time_sec/60:5.2f}m)")
    print()
    
    # Calculate speedups
    if 'Sequential (0 workers)' in times:
        baseline = times['Sequential (0 workers)']
        print("Speedup vs Sequential:")
        print("-" * 70)
        for name, time_sec in times.items():
            if name != 'Sequential (0 workers)' and time_sec:
                speedup = baseline / time_sec
                efficiency = (speedup / int(name.split('(')[1].split()[0])) * 100 if 'workers)' in name else 0
                print(f"  {name:30s}: {speedup:5.2f}x", end='')
                if efficiency > 0:
                    print(f" (efficiency: {efficiency:.1f}%)")
                else:
                    print()
        print()
    
    # Check for caching
    print("Feature Caching Analysis:")
    print("-" * 70)
    for name, logfile in tests.items():
        logpath = results_dir / logfile
        if logpath.exists():
            with open(logpath) as f:
                content = f.read()
                cache_hits = content.count('Loaded from cache')
                cache_misses = content.count('Fetching') + content.count('Computing')
                print(f"  {name:30s}: {cache_hits} cache hits, {cache_misses} misses")
    print()
    
    print("="*70)
    print("\n💡 Recommendations:")
    if 'Sequential (0 workers)' in times and 'Parallel (4 workers)' in times:
        speedup = times['Sequential (0 workers)'] / times['Parallel (4 workers)']
        if speedup < 2:
            print("  ⚠️  Low speedup - check for I/O bottlenecks or GIL contention")
        elif speedup < 3:
            print("  ✓  Moderate speedup - typical for Python multiprocessing")
        else:
            print("  ✅ Excellent speedup - parallel workers are very effective!")
    print()

if __name__ == '__main__':
    main()
EOF

chmod +x "$RESULTS_DIR/analyze.py"

echo "🔍 Running detailed analysis..."
echo ""
python3 "$RESULTS_DIR/analyze.py"

echo ""
echo "✅ Benchmark complete!"
echo ""
echo "📁 All results saved to: $RESULTS_DIR/"
echo "🔍 Re-run analysis: python3 $RESULTS_DIR/analyze.py"
echo ""
