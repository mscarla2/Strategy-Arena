# Strategy Arena Scripts

This directory contains shell scripts for running different evolution configurations and performance benchmarks.

## Evolution Scripts

### 🚀 Quick Test (`run_arena_quick.sh`)
Fast validation run for testing the system.
- **Population**: 20 strategies
- **Generations**: 5
- **Universe**: Oil stocks
- **Runtime**: ~5 minutes
- **Use case**: Quick testing, debugging, validation

```bash
bash scripts/run_arena_quick.sh
```

### 🛢️ Oil Stocks Evolution (`run_arena_oil.sh`)
Optimized for oil & gas microcap stocks with oil-specific features.
- **Population**: 100 strategies
- **Generations**: 30
- **Universe**: Oil stocks (EONR, TPET, USEG, STAK, PRSO, BATL, USO, BNO)
- **Timeframe**: Weekly (optimal for microcaps)
- **Features**: WTI/Brent correlation, crack spreads, EIA inventory
- **Runtime**: ~20-30 minutes

```bash
bash scripts/run_arena_oil.sh
```

### 📈 General Stocks Evolution (`run_arena_general.sh`)
Optimized for large-cap tech stocks.
- **Population**: 50 strategies
- **Generations**: 30
- **Tickers**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
- **Timeframe**: Weekly
- **Features**: Smart Money Concepts, Support/Resistance, Regime Detection
- **Runtime**: ~30-60 minutes

```bash
bash scripts/run_arena_general.sh
```

### ⚙️ Custom Configuration (`run_arena.sh`)
Fully configurable production run with all advanced features.
- **Configurable**: All parameters can be edited in the script
- **Default**: Oil stocks with all features enabled
- **Runtime**: ~30-60 minutes

```bash
bash scripts/run_arena.sh
```

## Performance Benchmarking Scripts

### 🔍 Quick Benchmark (`benchmark_quick.sh`)
Fast performance comparison (2 tests, ~5-10 minutes).

**Tests:**
1. Sequential evaluation (0 workers)
2. Parallel evaluation (auto-detect workers)

**Outputs:**
- Execution times for each test
- Speedup calculation
- Feature caching statistics
- Worker count detection

```bash
bash scripts/benchmark_quick.sh
```

**Use this to**: Quickly verify parallel workers are working and measure speedup.

### 📊 Full Benchmark (`benchmark_performance.sh`)
Comprehensive performance analysis (4 tests, ~15-20 minutes).

**Tests:**
1. Sequential (0 workers)
2. Parallel with 2 workers
3. Parallel with 4 workers
4. Auto-detect workers

**Outputs:**
- Detailed timing for all configurations
- Speedup and efficiency calculations
- Feature caching analysis
- Python analysis script for detailed metrics

```bash
bash scripts/benchmark_performance.sh
```

**Use this to**: 
- Find optimal worker count for your system
- Measure parallel efficiency
- Validate feature caching effectiveness
- Generate detailed performance reports

### Understanding Benchmark Results

**Speedup**: How much faster parallel is vs sequential
- `< 1.5x`: Low speedup - check for bottlenecks
- `1.5-2.5x`: Moderate - typical for Python multiprocessing
- `> 2.5x`: Excellent - parallel workers very effective

**Efficiency**: Speedup divided by number of workers
- `> 80%`: Excellent parallelization
- `50-80%`: Good parallelization
- `< 50%`: Poor parallelization - overhead too high

**Cache Hits**: Number of times data was loaded from cache
- More cache hits = better performance
- Compare first vs second run to see caching impact

## Parallel Workers

All scripts now support the `--parallel-workers` flag:
- `-1`: Auto-detect (uses CPU count - 1) - **Default**
- `0`: Sequential evaluation (single worker)
- `N > 0`: Use exactly N parallel workers

The scripts are configured with `-1` (auto-detect) for optimal performance.

## Features Enabled

All scripts include:
- ✅ **Smart Money Concepts**: Order blocks, Fair Value Gaps, liquidity sweeps
- ✅ **Support/Resistance**: Volume profile, pivot points
- ✅ **Regime Detection**: Volatility, trend, market regime adaptation
- ✅ **Trailing Stops**: ATR-based volatility stops
- ✅ **Kelly Sizing**: Kelly Criterion position sizing
- ✅ **Calmar Fitness**: Drawdown-aware fitness function
- ✅ **Partial Rebalancing**: 20% deviation threshold

Oil-specific scripts also include:
- ✅ **Oil Features**: WTI/Brent correlation, crack spreads, EIA inventory

## Viewing Results

After evolution completes:

```bash
# Terminal UI
python3 ui/strategy_picker.py

# Web Dashboard
streamlit run ui/dashboard.py

# Direct database access
sqlite3 data/gp_strategies.db
```

## Customization

To customize a script:
1. Copy the script to a new name
2. Edit the configuration variables at the top
3. Run your custom script

Example:
```bash
cp scripts/run_arena_oil.sh scripts/run_arena_custom.sh
# Edit scripts/run_arena_custom.sh
bash scripts/run_arena_custom.sh
```

## Tips for Performance Testing

1. **Run benchmarks when system is idle** - Close other applications
2. **Run multiple times** - First run may be slower due to cold cache
3. **Check CPU usage** - Use `htop` or Activity Monitor during runs
4. **Compare cache hits** - Second run should show more cache hits
5. **Test different worker counts** - Optimal count varies by system
6. **Monitor memory usage** - Each worker needs memory for data
