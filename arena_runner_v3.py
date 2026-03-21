#!/usr/bin/env python3
"""
ALPHAGENE Arena Runner v3.0
Genetic Programming - Evolves expression trees instead of fixed genes.
OPTIMIZED: Feature caching, parallel evaluation, oil-specific benchmarks
"""

import os
import sys
import random
import argparse
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from config import (
    TRADING_DAYS_PER_YEAR,
    DEFAULT_INITIAL_CAPITAL,
    RISK_FREE_RATE,
    UNIVERSE_TYPE,
    RECENCY_HALF_LIFE,
    USE_EXPANDING_WINDOW,
    OIL_USE_EXPANDED_UNIVERSE,
    OIL_PRIMARY_BENCHMARK,
    ENABLE_FEATURE_CACHE,
    FEATURE_CACHE_MAX_DATES,
)

from data import DataFetcher, get_universe_for_period

from evolution.gp import (
    GPStrategy,
    GPOperators,
    TreeGenerator,
    FeatureLibrary,
    WalkForwardEvaluator,
    evaluate_strategy_parallel,
    calculate_fitness_v2,
)

from evolution.gp_storage import GPDatabase


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL STYLING
# ═══════════════════════════════════════════════════════════════════════════════

class Theme:
    PRIMARY = "\033[38;5;75m"
    SECONDARY = "\033[38;5;245m"
    SUCCESS = "\033[38;5;78m"
    WARNING = "\033[38;5;220m"
    ERROR = "\033[38;5;203m"
    ACCENT = "\033[38;5;183m"
    MUTED = "\033[38;5;240m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def style(text: str, *styles) -> str:
    if not sys.stdout.isatty():
        return str(text)
    return f"{''.join(styles)}{text}{Theme.RESET}"


def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗  ██████╗ ███████╗███╗   ██╗███████╗  ║
    ║ ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗██╔════╝ ██╔════╝████╗  ██║██╔════╝  ║
    ║ ███████║██║     ██████╔╝███████║███████║██║  ███╗█████╗  ██╔██╗ ██║█████╗    ║
    ║ ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║██║   ██║██╔══╝  ██║╚██╗██║██╔══╝    ║
    ║ ██║  ██║███████╗██║     ██║  ██║██║  ██║╚██████╔╝███████╗██║ ╚████║███████╗  ║
    ║ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚══════╝  ║
    ║                    GENETIC PROGRAMMING ARENA v3.0                            ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(style(banner, Theme.PRIMARY))


def print_section(title: str, icon: str = "═"):
    print(f"\n  {style(icon, Theme.ACCENT)}  {style(title, Theme.PRIMARY, Theme.BOLD)}")
    print(f"  {style('─' * 60, Theme.MUTED)}")


def print_status(message: str, status: str = "info"):
    icons = {
        "info": ("ℹ", Theme.PRIMARY),
        "success": ("✓", Theme.SUCCESS),
        "warning": ("⚠", Theme.WARNING),
        "error": ("✗", Theme.ERROR),
        "progress": ("◐", Theme.ACCENT),
    }
    icon, color = icons.get(status, ("•", Theme.MUTED))
    print(f"  {style(icon, color)} {message}")


def print_progress(current: int, total: int, prefix: str = "", width: int = 30):
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r  {prefix} [{bar}] {pct*100:.0f}% ({current}/{total})", end='', flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_benchmark(
    prices: pd.DataFrame,
    start_date: str,
    end_date: str,
    method: str = "equal_weight",
    lookback: int = 252,
    top_pct: int = 20,
    rebalance: bool = True,
    benchmark_ticker: str = None
) -> Dict[str, float]:
    """
    Calculate benchmark returns for a period.
    
    Args:
        prices: DataFrame of stock prices
        start_date: Period start date (YYYY-MM-DD)
        end_date: Period end date (YYYY-MM-DD)
        method: Benchmark type:
            - "equal_weight": Equal-weight portfolio (default, passive baseline)
            - "momentum": Top 20% by 12-month momentum (active strategy)
            - "single_ticker": Single ticker benchmark (e.g., USO, BNO)
        lookback: Lookback period for momentum calculation (days)
        top_pct: Percentage of stocks to select for momentum
        rebalance: If False, true buy & hold (no rebalancing)
        benchmark_ticker: Ticker to use for single_ticker method
    
    Returns:
        Dict with total_return, sharpe_ratio, max_drawdown
    """
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    if method == "equal_weight":
        mask = (prices.index >= start_dt) & (prices.index <= end_dt)
        period_prices = prices.loc[mask]
        
        if len(period_prices) < 2:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        if rebalance:
            # Equal-weight with daily rebalancing
            daily_returns = period_prices.pct_change().dropna()
            portfolio_returns = daily_returns.mean(axis=1)
        else:
            # True buy & hold: equal-weight at start, let it drift
            initial_weights = 1.0 / len(period_prices.columns)
            
            # Calculate cumulative returns for each stock
            stock_returns = period_prices / period_prices.iloc[0]
            
            # Portfolio value = weighted sum of stock values
            portfolio_value = (stock_returns * initial_weights).sum(axis=1)
            
            # Portfolio returns
            portfolio_returns = portfolio_value.pct_change().dropna()
    
    elif method == "momentum":
        lookback_start = start_dt - pd.Timedelta(days=int(lookback * 1.5))
        mask = (prices.index >= lookback_start) & (prices.index <= end_dt)
        all_prices = prices.loc[mask]
        
        if len(all_prices) < lookback + 1:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        start_idx = all_prices.index.get_indexer([start_dt], method='nearest')[0]
        if start_idx < lookback:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        mom_end = all_prices.iloc[start_idx]
        mom_start = all_prices.iloc[start_idx - lookback]
        momentum = ((mom_end / mom_start) - 1).dropna()
        
        if len(momentum) == 0:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        n_select = max(1, int(len(momentum) * top_pct / 100))
        top_stocks = momentum.nlargest(n_select).index.tolist()
        
        period_mask = (prices.index >= start_dt) & (prices.index <= end_dt)
        period_prices = prices.loc[period_mask, top_stocks]
        
        if len(period_prices) < 2:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        daily_returns = period_prices.pct_change().dropna()
        portfolio_returns = daily_returns.mean(axis=1)
    
    elif method == "single_ticker":
        # Single ticker benchmark (e.g., USO for oil)
        if benchmark_ticker is None or benchmark_ticker not in prices.columns:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        mask = (prices.index >= start_dt) & (prices.index <= end_dt)
        period_prices = prices.loc[mask, benchmark_ticker]
        
        if len(period_prices) < 2:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        portfolio_returns = period_prices.pct_change().dropna()
    
    else:
        # Unknown method, default to equal-weight
        mask = (prices.index >= start_dt) & (prices.index <= end_dt)
        period_prices = prices.loc[mask]
        
        if len(period_prices) < 2:
            return {'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0}
        
        daily_returns = period_prices.pct_change().dropna()
        portfolio_returns = daily_returns.mean(axis=1)
    
    # Calculate metrics
    cumulative = (1 + portfolio_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
    
    if len(portfolio_returns) > 0 and portfolio_returns.std() > 0:
        excess = portfolio_returns.mean() - (RISK_FREE_RATE / TRADING_DAYS_PER_YEAR)
        sharpe = (excess / portfolio_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0
    
    if len(cumulative) > 0:
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())
    else:
        max_dd = 0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GP EVOLUTION ARENA
# ═══════════════════════════════════════════════════════════════════════════════

class GPEvolutionArena:
    """Genetic Programming Arena - evolves expression trees that compute stock scores."""

    WARMUP_DAYS = 300

    def __init__(
        self,
        tickers: List[str] = None,
        start_date: str = "2020-01-01",
        end_date: str = None,
        population_size: int = 50,
        max_depth: int = 7,
        tournament_size: int = 3,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2,
        elite_count: int = 2,
        parsimony_coefficient: float = 0.001,
        initial_capital: float = None,
        timeframe: str = 'weekly',
        # Priority 1 & 2 parameters
        rebalance_threshold: float = 0.20,
        use_stops: bool = False,
        use_kelly: bool = False,
        use_calmar_fitness: bool = False,
        # Priority 3+ parameters
        enable_smc: bool = False,
        enable_sr: bool = False,
        enable_oil: bool = False,
        enable_regime: bool = False,
        enable_dilution: bool = False,
        # Performance optimization
        parallel_workers: int = 10,
        # RC-3: Fitness v2 parameters
        universe_type: str = None,  # Auto-detected if None
        recency_half_life: int = None,  # Uses config default if None
        use_fitness_v2: bool = False,
    ):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.initial_capital = initial_capital or DEFAULT_INITIAL_CAPITAL

        self.population_size = population_size
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_count = elite_count
        self.parsimony_coefficient = parsimony_coefficient
        
        # Multi-timeframe support
        self.timeframe = timeframe
        self._setup_timeframe_config()
        
        # Priority 1 & 2 features
        self.rebalance_threshold = rebalance_threshold
        self.use_stops = use_stops
        self.use_kelly = use_kelly
        self.use_calmar_fitness = use_calmar_fitness
        
        # Priority 3+ features
        self.enable_smc = enable_smc
        self.enable_sr = enable_sr
        self.enable_oil = enable_oil
        self.enable_regime = enable_regime
        self.enable_dilution = enable_dilution
        
        # RC-3: Fitness v2 parameters
        self.recency_half_life = recency_half_life or RECENCY_HALF_LIFE
        self.use_fitness_v2 = use_fitness_v2
        
        # Performance optimization
        # If parallel_workers is 0, use sequential (1 worker)
        # If parallel_workers > 0, use that many workers
        # If parallel_workers < 0, auto-detect (cpu_count - 1)
        if parallel_workers == 0:
            self.parallel_workers = 1  # Sequential evaluation
        elif parallel_workers > 0:
            self.parallel_workers = parallel_workers
        else:
            self.parallel_workers = max(1, cpu_count() - 1)  # Auto-detect
        
        # Initialize new components
        self._init_priority_components()

        # Handle special universe keywords
        # RC-4: Expanded oil universe with reference panel
        self.is_oil_universe = False
        self.tradeable_tickers = None  # None = all tickers are tradeable
        self.oil_reference_panel = []
        
        if tickers and len(tickers) == 1 and tickers[0] == 'oil':
            from data.universe import (
                get_oil_universe, get_oil_tradeable_tickers,
                get_oil_reference_panel, get_oil_benchmarks
            )
            self.is_oil_universe = True
            
            if OIL_USE_EXPANDED_UNIVERSE:
                # RC-4: Download full universe (tradeable + reference + benchmarks)
                self.tickers = get_oil_universe(expanded=True)
                # Only these tickers can be held in the portfolio
                self.tradeable_tickers = get_oil_tradeable_tickers()
                self.oil_reference_panel = get_oil_reference_panel()
                print_status(
                    f"Oil expanded universe: {len(self.tickers)} total "
                    f"({len(self.tradeable_tickers)} tradeable, "
                    f"{len(self.oil_reference_panel)} reference panel)",
                    "info"
                )
            else:
                # Legacy: 8-ticker universe
                self.tickers = get_oil_universe(expanded=False)
        else:
            self.tickers = tickers if tickers else get_universe_for_period(start_date)
        
        # RC-3: Auto-detect universe type if not specified
        if universe_type:
            self.universe_type = universe_type
        elif self.is_oil_universe:
            self.universe_type = 'oil_microcap'
        else:
            self.universe_type = UNIVERSE_TYPE
        
        self.data_fetcher = DataFetcher()

        self.feature_lib = FeatureLibrary(
            enable_smc=enable_smc,
            enable_sr=enable_sr,
            enable_oil=enable_oil
        )
        self.generator = TreeGenerator(self.feature_lib.feature_names)
        self.operators = GPOperators(self.feature_lib.feature_names, max_depth)

        self.db = GPDatabase()
        self.run_id = str(uuid.uuid4())[:12]

        self.population: List[GPStrategy] = []
        self.generation = 0
        self.best_fitness = -float('inf')
        self.best_strategy: Optional[GPStrategy] = None
        self.stagnant_generations = 0

        self.generation_history: List[Dict] = []
        self.prices: Optional[pd.DataFrame] = None
        self.periods: List[Tuple[str, str, str, str]] = []
        self.benchmark_results: List[Dict] = []
        self.oil_benchmark_results: List[Dict] = []  # USO/BNO benchmarks
        
        # Feature cache for performance
        self.feature_cache: Optional[Dict[str, pd.DataFrame]] = None
    
    def _init_priority_components(self):
        """Initialize Priority 1 & 2 components."""
        from backtest.rebalancing import PartialRebalancer
        from backtest.stops import TrailingVolatilityStop
        from backtest.position_sizing import CombinedPositionSizer
        
        # Partial rebalancing (Priority 1.1)
        self.rebalancer = PartialRebalancer(deviation_threshold=self.rebalance_threshold)
        
        # Trailing stops (Priority 2.1)
        self.stop_manager = TrailingVolatilityStop(atr_multiplier=2.0, lookback=14) if self.use_stops else None
        
        # Kelly + Volatility position sizing (Priority 2.2 & 2.3)
        self.position_sizer = CombinedPositionSizer(
            kelly_lookback=20,
            kelly_max=0.25,
            vol_lookback=20,
            target_vol=0.15
        ) if self.use_kelly else None
    
    def _setup_timeframe_config(self):
        """Setup timeframe-specific configuration."""
        from backtest.multi_timeframe import Timeframe, TIMEFRAME_CONFIGS
        
        timeframe_map = {
            'intraday': Timeframe.INTRADAY,
            'swing': Timeframe.SWING,
            'weekly': Timeframe.WEEKLY,
            'monthly': Timeframe.MONTHLY
        }
        
        self.timeframe_enum = timeframe_map.get(self.timeframe, Timeframe.WEEKLY)
        self.timeframe_config = TIMEFRAME_CONFIGS[self.timeframe_enum]
        
        # Initialize risk management components
        from backtest.risk_management import MicrocapSlippageModel, DilutionFilter, LiquidityConstraint
        
        self.slippage_model = MicrocapSlippageModel(
            base_slippage_bps=self.timeframe_config.base_slippage_bps,
            volume_impact_factor=0.5,
            commission_per_trade=self.timeframe_config.commission_per_trade
        )
        
        self.dilution_filter = DilutionFilter(
            volume_spike_threshold=5.0,
            price_drop_threshold=-0.10
        )
        
        self.liquidity_constraint = LiquidityConstraint(
            max_pct_adv=0.05,
            min_adv_dollars=50000
        )

    def _generate_walk_forward_periods(
        self,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3
    ) -> List[Tuple[str, str, str, str]]:
        """Generate walk-forward train/test periods with anchored recent period."""
        periods = []
        start = pd.Timestamp(self.start_date)
        end = pd.Timestamp(self.end_date)

        current = start
        while current + pd.DateOffset(months=train_months + test_months) <= end:
            train_start = current.strftime("%Y-%m-%d")
            train_end = (current + pd.DateOffset(months=train_months)).strftime("%Y-%m-%d")
            test_start = train_end
            test_end = (current + pd.DateOffset(months=train_months + test_months)).strftime("%Y-%m-%d")

            periods.append((train_start, train_end, test_start, test_end))
            current += pd.DateOffset(months=step_months)

        # ═══════════════════════════════════════════════════════════════
        # RC-1/RC-3: Anchored recent period — guarantees most recent data
        # is always in the evaluation window (e.g., March 2026)
        # ═══════════════════════════════════════════════════════════════
        recent_test_end = end
        recent_test_start = recent_test_end - pd.DateOffset(months=test_months)
        recent_train_end = recent_test_start
        recent_train_start = recent_train_end - pd.DateOffset(months=train_months)
        
        anchored_period = (
            recent_train_start.strftime("%Y-%m-%d"),
            recent_train_end.strftime("%Y-%m-%d"),
            recent_test_start.strftime("%Y-%m-%d"),
            recent_test_end.strftime("%Y-%m-%d"),
        )
        
        # Only add if it doesn't duplicate the last standard period
        if not periods or anchored_period != periods[-1]:
            periods.append(anchored_period)
            print_status(
                f"Anchored recent period: test {anchored_period[2]} to {anchored_period[3]}",
                "info"
            )

        return periods

    def _calculate_benchmarks(self):
        """Calculate benchmark returns for all test periods."""
        print_status("Calculating benchmarks...", "progress")
        
        self.benchmark_results = []
        self.oil_benchmark_results = []
        
        for train_start, train_end, test_start, test_end in self.periods:
            # RC-4: Use XLE as primary benchmark for oil universe if available
            if self.is_oil_universe and OIL_PRIMARY_BENCHMARK in self.prices.columns:
                bench = calculate_benchmark(
                    self.prices,
                    test_start,
                    test_end,
                    method="single_ticker",
                    benchmark_ticker=OIL_PRIMARY_BENCHMARK
                )
            else:
                # Standard equal-weight benchmark
                bench = calculate_benchmark(
                    self.prices,
                    test_start,
                    test_end,
                    method="equal_weight",
                    rebalance=False
                )
            bench['test_start'] = test_start
            bench['test_end'] = test_end
            self.benchmark_results.append(bench)
            
            bench_name = f'{OIL_PRIMARY_BENCHMARK}_benchmark' if self.is_oil_universe else 'equal_weight_buy_hold'
            self.db.save_benchmark(self.run_id, bench_name, test_start, test_end, bench)
            
            # Oil-specific benchmarks (USO, BNO, XLE, XOP)
            if self.is_oil_universe:
                for oil_ticker in ['USO', 'BNO', 'XLE', 'XOP']:
                    if oil_ticker in self.prices.columns:
                        oil_bench = calculate_benchmark(
                            self.prices,
                            test_start,
                            test_end,
                            method="single_ticker",
                            benchmark_ticker=oil_ticker
                        )
                        oil_bench['test_start'] = test_start
                        oil_bench['test_end'] = test_end
                        oil_bench['ticker'] = oil_ticker
                        self.oil_benchmark_results.append(oil_bench)
                        
                        self.db.save_benchmark(self.run_id, f'{oil_ticker}_benchmark', test_start, test_end, oil_bench)
        
        avg_sharpe = np.mean([b['sharpe_ratio'] for b in self.benchmark_results])
        avg_ret = np.mean([b['total_return'] for b in self.benchmark_results])
        bench_label = f"{OIL_PRIMARY_BENCHMARK}" if self.is_oil_universe else "Equal-Weight"
        print_status(f"{bench_label} Benchmark: avg Sharpe={avg_sharpe:.2f}, avg Return={avg_ret:.1%}", "success")
        
        # Print oil benchmarks if available
        if self.oil_benchmark_results:
            for ticker in ['USO', 'BNO', 'XLE', 'XOP']:
                ticker_results = [b for b in self.oil_benchmark_results if b.get('ticker') == ticker]
                if ticker_results:
                    avg_sharpe_oil = np.mean([b['sharpe_ratio'] for b in ticker_results])
                    avg_ret_oil = np.mean([b['total_return'] for b in ticker_results])
                    print_status(f"{ticker} Benchmark: avg Sharpe={avg_sharpe_oil:.2f}, avg Return={avg_ret_oil:.1%}", "success")

    def initialize_population(self):
        """Create initial population using ramped half-and-half."""
        print_status(f"Creating initial population of {self.population_size} strategies...", "progress")

        self.population = []
        depths = range(2, self.max_depth + 1)

        for i in range(self.population_size):
            depth = list(depths)[i % len(depths)]
            method = "grow" if i % 2 == 0 else "full"
            tree = self.generator.random_tree(max_depth=depth, method=method)

            strategy = GPStrategy(
                tree=tree,
                top_pct=random.choice([10, 15, 20, 25, 30]),
                generation=0,
                origin="random"
            )
            self.population.append(strategy)

        print_status(f"Created {len(self.population)} strategies", "success")

    def _precompute_features(self):
        """
        Pre-compute all features once for performance optimization (RC-5).
        
        Replaces the placeholder with a proper FeaturePrecomputeCache that
        computes features for all rebalance dates across all walk-forward periods.
        """
        if ENABLE_FEATURE_CACHE and hasattr(self.feature_lib, 'compute_all'):
            print_status("Pre-computing features for all rebalance dates (RC-5)...", "progress")
            try:
                from evolution.feature_cache import FeaturePrecomputeCache
                
                # Get volume data if available
                volume = getattr(self, 'volume', None)
                
                self.feature_cache = FeaturePrecomputeCache(
                    feature_lib=self.feature_lib,
                    prices=self.prices,
                    volume=volume,
                    periods=self.periods,
                    rebalance_frequency=21,
                    max_dates=FEATURE_CACHE_MAX_DATES,
                )
                print_status(
                    f"Feature cache ready: {self.feature_cache.stats['cached_dates']} dates, "
                    f"{self.feature_cache.stats['features_per_date']} features each",
                    "success"
                )
            except Exception as e:
                print_status(f"Feature cache failed ({e}), falling back to on-demand", "warning")
                self.feature_cache = {}
        else:
            print_status("Feature caching: on-demand mode", "info")
            self.feature_cache = {}

    def evaluate_population(self, use_parallel: bool = True, n_jobs: int = None) -> Dict[str, Tuple[float, List[Dict]]]:
        """
        Evaluate all strategies with walk-forward validation.
        
        Args:
            use_parallel: Enable parallel evaluation (default: True)
            n_jobs: Number of parallel workers (default: cpu_count() - 1)
        
        Returns:
            Dict mapping strategy_id to (fitness, period_results)
        """
        evaluator = WalkForwardEvaluator(
            prices=self.prices,
            periods=self.periods,
            benchmark_results=self.benchmark_results,
            transaction_cost=0.002,
            rebalance_frequency=21,
            # Pass Priority 1 & 2 components
            rebalancer=self.rebalancer,
            stop_manager=self.stop_manager,
            position_sizer=self.position_sizer,
            use_calmar_fitness=self.use_calmar_fitness,
            # RC-3: Fitness v2 parameters
            universe_type=self.universe_type,
            recency_half_life=self.recency_half_life,
            use_fitness_v2=self.use_fitness_v2,
            # RC-4: Oil reference panel — restrict portfolio to tradeable tickers
            tradeable_tickers=self.tradeable_tickers,
        )
        
        results = {}
        
        if use_parallel and len(self.population) > 1:
            # Parallel evaluation using multiprocessing
            if n_jobs is None:
                n_jobs = max(1, cpu_count() - 1)  # Leave one core free
            
            print_status(f"Evaluating {len(self.population)} strategies using {n_jobs} parallel workers...", "progress")
            
            # Serialize evaluator config for pickling
            evaluator_config = evaluator.to_config()
            
            # Prepare arguments for parallel evaluation
            eval_args = [(strategy, evaluator_config) for strategy in self.population]
            
            # Use multiprocessing Pool for parallel evaluation with progress tracking
            with Pool(processes=n_jobs) as pool:
                # Use imap_unordered with chunksize=1 for real-time progress updates
                parallel_results = []
                for i, result in enumerate(pool.imap_unordered(evaluate_strategy_parallel, eval_args, chunksize=1)):
                    parallel_results.append(result)
                    print_progress(i + 1, len(self.population), "Evaluating")
                
                print()  # New line after progress bar
            
            # Update strategies and collect results
            # Note: results are in completion order, not original order
            strategy_map = {s.strategy_id: s for s in self.population}
            for strategy_id, fitness, period_results in parallel_results:
                if strategy_id in strategy_map:
                    strategy_map[strategy_id].period_metrics = period_results
                    strategy_map[strategy_id].fitness = fitness
                    results[strategy_id] = (fitness, period_results)
            
            print_status(f"Completed parallel evaluation", "success")
        else:
            # Sequential evaluation (fallback or single strategy)
            print_status(f"Evaluating {len(self.population)} strategies sequentially...", "progress")
            
            for i, strategy in enumerate(self.population):
                print_progress(i + 1, len(self.population), "Evaluating")
                
                fitness_result = evaluator.evaluate_strategy(strategy)
                
                strategy.period_metrics = fitness_result.period_results
                strategy.fitness = fitness_result.total
                
                results[strategy.strategy_id] = (fitness_result.total, fitness_result.period_results)
            
            print()
        
        return results

    def adjusted_fitness(self, strategy: GPStrategy) -> float:
        """Fitness adjusted for complexity."""
        return strategy.fitness - self.parsimony_coefficient * strategy.complexity()

    def tournament_select(self) -> GPStrategy:
        """Select parent via tournament selection."""
        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(tournament, key=lambda s: self.adjusted_fitness(s))

    def create_next_generation(self):
        """Create next generation using GP operators with diversity enforcement."""
        sorted_pop = sorted(self.population, key=lambda s: self.adjusted_fitness(s), reverse=True)

        new_population = []
        seen_formulas: Set[str] = set()

        # Elitism - keep top performers
        for elite in sorted_pop[:self.elite_count]:
            elite_copy = elite.copy()
            elite_copy.generation = self.generation + 1
            elite_copy.origin = "elite"
            elite_copy.strategy_id = str(uuid.uuid4())[:8]
            
            formula = elite_copy.get_formula()
            if formula not in seen_formulas:
                new_population.append(elite_copy)
                seen_formulas.add(formula)

        # Check for convergence - if population is too similar, inject diversity
        unique_formulas = len(set(s.get_formula() for s in self.population))
        diversity_ratio = unique_formulas / len(self.population)
        
        # Adaptive mutation rate based on diversity
        adaptive_mutation_prob = self.mutation_prob
        if diversity_ratio < 0.5:  # Less than 50% unique - more aggressive
            adaptive_mutation_prob = min(0.7, self.mutation_prob * 4)  # Quadruple mutation rate
            print_status(f"Low diversity ({diversity_ratio:.1%}), increasing mutation to {adaptive_mutation_prob:.1%}", "warning")
        
        # Generate rest with diversity check
        attempts = 0
        max_attempts = self.population_size * 5  # Increased attempts
        
        while len(new_population) < self.population_size and attempts < max_attempts:
            attempts += 1
            r = random.random()

            # Adjust probabilities based on diversity - favor mutation and new random when low
            if diversity_ratio < 0.3:
                # Low diversity: prioritize mutation and random injection
                crossover_threshold = 0.2  # Reduce crossover
                mutation_threshold = crossover_threshold + 0.5  # Increase mutation
                # Remaining 30% will be random or reproduction
            else:
                crossover_threshold = self.crossover_prob
                mutation_threshold = crossover_threshold + adaptive_mutation_prob

            if r < crossover_threshold:
                parent1 = self.tournament_select()
                parent2 = self.tournament_select()
                child = self.operators.crossover(parent1, parent2)
            elif r < mutation_threshold:
                parent = self.tournament_select()
                child = self.operators.mutate(parent)
            else:
                # When diversity is low, prefer creating new random strategies
                if diversity_ratio < 0.4:  # Increased threshold
                    depth = random.randint(2, self.max_depth)
                    tree = self.generator.random_tree(max_depth=depth, method="grow")
                    child = GPStrategy(
                        tree=tree,
                        top_pct=random.choice([10, 15, 20, 25, 30]),
                        generation=self.generation + 1,
                        origin="random_injection"
                    )
                else:
                    parent = self.tournament_select()
                    child = parent.copy()
                    child.generation = self.generation + 1
                    child.origin = "reproduction"
                    child.strategy_id = str(uuid.uuid4())[:8]

            if child.tree.depth() > self.max_depth:
                continue
                
            formula = child.get_formula()
            # Only check duplicates within this generation, not globally
            if formula in seen_formulas:
                # If we're stuck with duplicates and diversity is low, force a mutation
                if diversity_ratio < 0.5 and attempts > max_attempts * 0.3:  # More aggressive
                    child = self.operators.mutate(child)
                    formula = child.get_formula()
                    if formula in seen_formulas:
                        # Try one more time with a different mutation
                        child = self.operators.mutate(child)
                        formula = child.get_formula()
                        if formula in seen_formulas:
                            continue
                else:
                    continue
                
            new_population.append(child)
            seen_formulas.add(formula)

        # Fill remaining with random if needed
        while len(new_population) < self.population_size:
            depth = random.randint(2, self.max_depth)
            tree = self.generator.random_tree(max_depth=depth, method="grow")
            strategy = GPStrategy(
                tree=tree,
                top_pct=random.choice([10, 15, 20, 25, 30]),
                generation=self.generation + 1,
                origin="random_fill"
            )
            formula = strategy.get_formula()
            if formula not in seen_formulas:
                new_population.append(strategy)
                seen_formulas.add(formula)

        self.generation += 1
        self.population = new_population[:self.population_size]

    def run_generation(self) -> Dict:
        """Run a single generation."""
        print_section(f"GENERATION {self.generation}", "🧬")
        print_status(f"Evaluating {len(self.population)} strategies...", "info")

        # Use parallel evaluation if parallel_workers > 1
        use_parallel = self.parallel_workers > 1
        results = self.evaluate_population(
            use_parallel=use_parallel,
            n_jobs=self.parallel_workers if use_parallel else None
        )

        fitnesses = [s.fitness for s in self.population]
        complexities = [s.complexity() for s in self.population]

        all_sharpes, all_returns, all_dds = [], [], []
        for _, period_metrics in results.values():
            if period_metrics:
                all_sharpes.extend([m['sharpe_ratio'] for m in period_metrics])
                all_returns.extend([m['total_return'] for m in period_metrics])
                all_dds.extend([m['max_drawdown'] for m in period_metrics])

        avg_fitness = np.mean(fitnesses) if fitnesses else 0
        max_fitness = np.max(fitnesses) if fitnesses else 0
        min_fitness = np.min(fitnesses) if fitnesses else 0

        # Track best
        if max_fitness > self.best_fitness:
            self.best_fitness = max_fitness
            self.best_strategy = max(self.population, key=lambda s: s.fitness).copy()
            self.stagnant_generations = 0
        else:
            self.stagnant_generations += 1

        # Print stats
        print(f"\n  Generation Stats:")
        print(f"    Fitness:      min={min_fitness:+.4f} avg={avg_fitness:+.4f} max={max_fitness:+.4f}")
        print(f"    Best Ever:    {self.best_fitness:+.4f}")
        print(f"    Avg Sharpe:   {np.mean(all_sharpes) if all_sharpes else 0:.3f}")
        print(f"    Avg Return:   {np.mean(all_returns) if all_returns else 0:.1%} (per period)")
        print(f"    Avg MaxDD:    {np.mean(all_dds) if all_dds else 0:.1%}")
        print(f"    Avg Complexity: {np.mean(complexities):.1f} nodes")
        print(f"    Stagnant:     {self.stagnant_generations} generations")

        # Top 5 strategies
        sorted_pop = sorted(self.population, key=lambda s: s.fitness, reverse=True)[:5]
        print(f"\n  Top 5 Formulas:")
        
        for i, s in enumerate(sorted_pop, 1):
            formula = s.get_formula()
            if len(formula) > 50:
                formula = formula[:47] + "..."

            metrics = getattr(s, 'period_metrics', [])
            avg_s = np.mean([m['sharpe_ratio'] for m in metrics]) if metrics else 0
            avg_r = np.mean([m['total_return'] for m in metrics]) if metrics else 0

            print(f"    {i}. fit={s.fitness:+.3f} sharpe={avg_s:.2f} ret={avg_r:.1%}")
            print(f"       {style(formula, Theme.MUTED)}")

        stats = {
            'generation': self.generation,
            'avg_fitness': avg_fitness,
            'max_fitness': max_fitness,
            'min_fitness': min_fitness,
            'best_ever': self.best_fitness,
            'avg_sharpe': np.mean(all_sharpes) if all_sharpes else 0,
            'avg_return': np.mean(all_returns) if all_returns else 0,
            'avg_max_dd': np.mean(all_dds) if all_dds else 0,
            'avg_complexity': np.mean(complexities),
            'stagnant': self.stagnant_generations,
        }

        self.generation_history.append(stats)
        return stats

    def should_early_stop(self) -> bool:
        """Check early stopping conditions."""
        # More lenient early stopping - allow more time for evolution
        if self.stagnant_generations >= 15:  # Increased from 10
            return True
        if len(self.generation_history) < 15:  # Need more history
            return False
        
        # Check if fitness is improving at all
        recent_best = [h['max_fitness'] for h in self.generation_history[-15:]]
        if max(recent_best[-5:]) <= max(recent_best[:10]) + 0.02:  # More lenient threshold
            return True
        
        return False

    def _save_to_database(self):
        """Save evolution results to database - only best strategy per run."""
        print_status("Saving results...", "progress")
        
        config = {
            'tickers': self.tickers[:10],
            'n_tickers': len(self.tickers),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'population_size': self.population_size,
            'max_depth': self.max_depth,
        }
        
        self.db.create_run(self.run_id, config)
        
        # Only save the best strategy from this run (no duplicates)
        if self.best_strategy:
            self.db.save_strategy(self.best_strategy, self.run_id)
            
            if hasattr(self.best_strategy, 'period_metrics') and self.best_strategy.period_metrics:
                self.db.save_period_results(
                    self.best_strategy.strategy_id,
                    self.run_id,
                    self.best_strategy.period_metrics
                )
        
        # Save generation statistics for evolution tracking
        for stats in self.generation_history:
            self.db.save_generation_stats(self.run_id, stats)
        
        best_id = self.best_strategy.strategy_id if self.best_strategy else ""
        self.db.complete_run(self.run_id, self.generation, self.best_fitness, best_id)
        
        print_status(f"Saved best strategy: {best_id[:8]}... (fitness={self.best_fitness:+.4f})", "success")

    def run(
        self,
        n_generations: int = 30,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3,
    ) -> List[Dict]:
        """Run the full GP evolution."""
        print_banner()

        print_section("CONFIGURATION", "⚙️")
        print(f"  Tickers:        {len(self.tickers)} stocks")
        print(f"  Date Range:     {self.start_date} to {self.end_date}")
        print(f"  Run ID:         {self.run_id}")
        print(f"  Population:     {self.population_size}")
        print(f"  Max Depth:      {self.max_depth}")
        print(f"  Generations:    {n_generations}")

        self.periods = self._generate_walk_forward_periods(train_months, test_months, step_months)
        print(f"  WF Periods:     {len(self.periods)}")

        if not self.periods:
            print_status("No valid periods. Check date range.", "error")
            return []

        print_section("DATA PREPARATION", "📊")
        print_status("Loading price data...", "progress")

        try:
            first_test_start = pd.Timestamp(self.periods[0][2])
            data_start = first_test_start - pd.Timedelta(days=int(self.WARMUP_DAYS * 1.5))
            data_start_str = data_start.strftime("%Y-%m-%d")

            self.prices = self.data_fetcher.fetch(
                start_date=data_start_str,
                end_date=self.end_date,
                tickers=self.tickers
            )

            self.prices = self.prices.dropna(axis=1, how='all').ffill().bfill()
            print_status(f"Loaded {len(self.prices)} days, {len(self.prices.columns)} tickers", "success")

        except Exception as e:
            print_status(f"Error loading data: {e}", "error")
            return []

        if len(self.prices) < 252:
            print_status(f"Insufficient data: {len(self.prices)} days", "error")
            return []

        # Pre-compute features for performance
        self._precompute_features()
        
        self._calculate_benchmarks()
        self.initialize_population()

        print_section("GP EVOLUTION", "🧬")
        start_time = datetime.now()

        for gen in range(n_generations):
            self.generation = gen
            self.run_generation()

            if self.should_early_stop():
                print_status("Early stopping triggered", "warning")
                break

            if gen < n_generations - 1:
                print_status("Evolving...", "progress")
                self.create_next_generation()
                print_status(f"Generation {self.generation} ready", "success")

        elapsed = datetime.now() - start_time
        self._save_to_database()
        self._print_final_report(elapsed)

        return self.generation_history

    def _print_final_report(self, elapsed: timedelta):
        """Print final evolution report."""
        print_section("EVOLUTION COMPLETE", "🏆")

        print(f"  Time:           {elapsed}")
        print(f"  Generations:    {self.generation + 1}")
        print(f"  Best Fitness:   {self.best_fitness:+.4f}")

        if self.best_strategy:
            print(f"\n  BEST FORMULA:")
            print(f"  Fitness:    {self.best_strategy.fitness:+.4f}")
            print(f"  Complexity: {self.best_strategy.complexity()} nodes")
            
            formula = self.best_strategy.get_formula()
            print(f"  Formula:    {formula[:70]}{'...' if len(formula) > 70 else ''}")

            metrics = getattr(self.best_strategy, 'period_metrics', [])
            if metrics:
                strat_ret = np.mean([m['total_return'] for m in metrics])
                strat_sharpe = np.mean([m['sharpe_ratio'] for m in metrics])
                bench_ret = np.mean([b['total_return'] for b in self.benchmark_results])
                bench_sharpe = np.mean([b['sharpe_ratio'] for b in self.benchmark_results])
                
                bench_label = f"{OIL_PRIMARY_BENCHMARK}" if self.is_oil_universe else "EQUAL-WEIGHT"
                print(f"\n  VS {bench_label} BENCHMARK:")
                print(f"    Strategy:  ret={strat_ret:+.1%}  sharpe={strat_sharpe:.2f}")
                print(f"    Benchmark: ret={bench_ret:+.1%}  sharpe={bench_sharpe:.2f}")
                print(f"    Alpha:     {strat_ret - bench_ret:+.1%}")
                
                # Oil-specific benchmarks
                if self.oil_benchmark_results:
                    print(f"\n  VS OIL BENCHMARKS:")
                    for ticker in ['USO', 'BNO', 'XLE', 'XOP']:
                        ticker_results = [b for b in self.oil_benchmark_results if b.get('ticker') == ticker]
                        if ticker_results:
                            oil_ret = np.mean([b['total_return'] for b in ticker_results])
                            oil_sharpe = np.mean([b['sharpe_ratio'] for b in ticker_results])
                            print(f"    {ticker}:      ret={oil_ret:+.1%}  sharpe={oil_sharpe:.2f}")
                            print(f"    Alpha vs {ticker}: {strat_ret - oil_ret:+.1%}")

        print()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="GP Arena - Evolve strategy formulas")

    parser.add_argument('--tickers', type=str, nargs='+', default=None)
    parser.add_argument('--start', type=str, default='2020-01-01')
    parser.add_argument('--end', type=str, default=None)
    parser.add_argument('--generations', type=int, default=30)
    parser.add_argument('--population', type=int, default=50)
    parser.add_argument('--max-depth', type=int, default=5)
    parser.add_argument('--tournament', type=int, default=3)
    parser.add_argument('--crossover', type=float, default=0.7)
    parser.add_argument('--mutation', type=float, default=0.2)
    parser.add_argument('--elite', type=int, default=2)
    parser.add_argument('--parsimony', type=float, default=0.001)
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--train-months', type=int, default=12)
    parser.add_argument('--test-months', type=int, default=3)
    parser.add_argument('--step-months', type=int, default=3)
    parser.add_argument('--timeframe', type=str, default='weekly',
                       choices=['intraday', 'swing', 'weekly', 'monthly'],
                       help='Trading timeframe (default: weekly, recommended for microcaps)')
    parser.add_argument('--universe', type=str, default=None,
                       help='Universe keyword (e.g., "oil" for oil stocks)')
    
    # Priority 1 & 2 parameters
    parser.add_argument('--rebalance-threshold', type=float, default=0.20,
                       help='Partial rebalancing deviation threshold (default: 0.20 = 20%%)')
    parser.add_argument('--use-stops', action='store_true',
                       help='Enable trailing volatility stops (ATR-based)')
    parser.add_argument('--use-kelly', action='store_true',
                       help='Enable Kelly Criterion + volatility position sizing')
    parser.add_argument('--use-calmar-fitness', action='store_true',
                       help='Use Calmar Ratio fitness instead of Sharpe-based')
    
    # Priority 3+ parameters (Advanced Features)
    parser.add_argument('--enable-smc', action='store_true',
                       help='Enable Smart Money Concepts features (order blocks, FVG, liquidity sweeps)')
    parser.add_argument('--enable-sr', action='store_true',
                       help='Enable Support/Resistance features (volume profile, pivot points)')
    parser.add_argument('--enable-oil', action='store_true',
                       help='Enable oil-specific features (WTI/Brent correlation, crack spreads, inventory)')
    parser.add_argument('--enable-regime', action='store_true',
                       help='Enable regime detection for adaptive position sizing')
    parser.add_argument('--enable-dilution', action='store_true',
                       help='Enable enhanced dilution detection (SEC filings, insider trading, news)')
    parser.add_argument('--use-nsga2', action='store_true',
                       help='Use NSGA-II multi-objective optimization')
    parser.add_argument('--objectives', type=str, default='sharpe,calmar',
                       help='Objectives for NSGA-II (comma-separated: sharpe,calmar,sortino,return)')
    parser.add_argument('--diversity-method', type=str, default=None,
                       choices=['fitness_sharing', 'novelty', 'island'],
                       help='Diversity preservation method')
    
    # Performance optimization
    parser.add_argument('--parallel-workers', type=int, default=-1,
                       help='Number of parallel workers for strategy evaluation (0=sequential, >0=specific count, -1=auto-detect)')
    
    # RC-3: Fitness v2 parameters
    parser.add_argument('--use-fitness-v2', action='store_true',
                       help='Use enhanced fitness v2 with recency weighting + universe-adaptive penalties (RC-3)')
    parser.add_argument('--universe-type', type=str, default=None,
                       choices=['general', 'oil_microcap', 'oil_largecap'],
                       help='Universe type for fitness penalty calibration (auto-detected if not set)')
    parser.add_argument('--recency-half-life', type=int, default=None,
                       help='Recency weighting half-life in periods (default: 4 = 1 year with quarterly steps)')

    args = parser.parse_args()
    
    # Print timeframe info
    if args.timeframe:
        print_section("Trading Timeframe Configuration", "⏱")
        timeframe_info = {
            'intraday': ('5-min candles, 6-hour holds', '75 bps', '1512%', 'NOT RECOMMENDED for microcaps'),
            'swing': ('Daily candles, 5-day holds', '35 bps', '35.3%', 'Challenging but possible'),
            'weekly': ('Daily candles, 10-day holds', '25 bps', '12.6%', 'OPTIMAL for microcaps'),
            'monthly': ('Daily candles, 30-day holds', '20 bps', '3.4%', 'Best risk-adjusted')
        }
        
        if args.timeframe in timeframe_info:
            desc, slippage, annual_slip, note = timeframe_info[args.timeframe]
            print_status(f"Timeframe: {args.timeframe.upper()}", "info")
            print_status(f"  Description: {desc}", "info")
            print_status(f"  Base Slippage: {slippage} per trade", "info")
            print_status(f"  Expected Annual Slippage: {annual_slip}", "warning" if float(annual_slip.rstrip('%')) > 50 else "info")
            print_status(f"  Note: {note}", "warning" if "NOT" in note else "success")

    # Handle universe keyword
    tickers = args.tickers
    if args.universe:
        if args.universe == 'oil':
            tickers = ['oil']  # Will be expanded in GPEvolutionArena.__init__
        else:
            print_status(f"Unknown universe keyword: {args.universe}", "warning")
    
    arena = GPEvolutionArena(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        population_size=args.population,
        max_depth=args.max_depth,
        tournament_size=args.tournament,
        crossover_prob=args.crossover,
        mutation_prob=args.mutation,
        elite_count=args.elite,
        parsimony_coefficient=args.parsimony,
        initial_capital=args.capital,
        timeframe=args.timeframe,
        rebalance_threshold=args.rebalance_threshold,
        use_stops=args.use_stops,
        use_kelly=args.use_kelly,
        use_calmar_fitness=args.use_calmar_fitness,
        enable_smc=args.enable_smc,
        enable_sr=args.enable_sr,
        enable_oil=args.enable_oil,
        enable_regime=args.enable_regime,
        enable_dilution=args.enable_dilution,
        parallel_workers=args.parallel_workers,
        # RC-3: Fitness v2 parameters
        universe_type=args.universe_type,
        recency_half_life=args.recency_half_life,
        use_fitness_v2=args.use_fitness_v2,
    )

    arena.run(
        n_generations=args.generations,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )


if __name__ == "__main__":
    main()