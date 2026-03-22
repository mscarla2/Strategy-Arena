# evolution/walkforward.py
"""
Walk-forward backtesting evaluator.

Contains: WalkForwardEvaluator
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from evolution.gp_fitness import (
    FitnessResult, calculate_fitness, calculate_fitness_v2
)


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class WalkForwardEvaluator:
    """Walk-forward backtesting evaluator."""
    
    def __init__(
        self,
        prices: pd.DataFrame,
        periods: List[Tuple[str, str, str, str]],
        benchmark_results: List[Dict] = None,
        transaction_cost: float = 0.002,
        rebalance_frequency: int = 21,
        # Priority 1 & 2 components
        rebalancer=None,
        stop_manager=None,
        position_sizer=None,
        use_calmar_fitness: bool = False,
        # RC-3: Fitness v2 parameters
        universe_type: str = 'general',
        recency_half_life: int = 4,
        use_fitness_v2: bool = False,
        # RC-4: Oil reference panel — tickers to exclude from portfolio selection
        tradeable_tickers: Optional[List[str]] = None,
        # Expanding window option
        expanding_window: bool = False,
        volume: Optional[pd.DataFrame] = None,
    ):
        self.prices = prices
        self.periods = periods
        self.benchmark_results = benchmark_results or []
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency
        
        # Priority 1 & 2 components
        self.rebalancer = rebalancer
        self.stop_manager = stop_manager
        self.position_sizer = position_sizer
        self.use_calmar_fitness = use_calmar_fitness
        
        # RC-3: Fitness v2 parameters
        self.universe_type = universe_type
        self.recency_half_life = recency_half_life
        self.use_fitness_v2 = use_fitness_v2
        
        # RC-4: If tradeable_tickers is set, only these tickers can be held
        # (reference panel tickers are used for features but not traded)
        self.tradeable_tickers = tradeable_tickers
        
        # Expanding window: when True, training data grows over time
        # (all history up to test_start instead of fixed train window)
        self.expanding_window = expanding_window
        
        self.volume = volume
    
    def evaluate_strategy(self, strategy) -> FitnessResult:
        """Evaluate strategy across all periods."""
        period_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(self.periods):
            result = self._evaluate_single_period(strategy, train_start, test_start, test_end)
            
            if result is not None:
                period_results.append(result)
        
        # Use pre-computed benchmarks, sliced to match period count
        benchmarks = self.benchmark_results[:len(period_results)]
        
        # RC-3: Use fitness v2 with recency weighting + universe-adaptive penalties
        if self.use_fitness_v2:
            return calculate_fitness_v2(
                period_results,
                benchmarks,
                transaction_cost=self.transaction_cost,
                recency_half_life=self.recency_half_life,
                universe_type=self.universe_type,
            )
        # Use Calmar fitness if enabled (Priority 2.4)
        elif self.use_calmar_fitness:
            from evolution.fitness_calmar import calculate_calmar_fitness
            return calculate_calmar_fitness(
                period_results,
                benchmarks,
                target_turnover=0.5
            )
        else:
            return calculate_fitness(
                period_results,
                benchmarks,
                self.transaction_cost
            )
    
    def _evaluate_single_period(
        self,
        strategy,
        train_start: str,
        test_start: str,
        test_end: str,
    ) -> Optional[Dict]:
        """Evaluate on a single test period."""

        test_start_dt = pd.Timestamp(test_start)
        test_end_dt = pd.Timestamp(test_end)

        # Always use full price history up to test_end for feature computation.
        # Features (momentum, volatility, etc.) require up to 504 days of lookback.
        # Restricting to the training window (e.g. 6 months ≈ 126 days) causes
        # score_stocks() to return constant 0.5 for every stock, making all
        # strategies score identically. The GP tree is already evolved externally,
        # so using full history here introduces no look-ahead bias.
        mask = self.prices.index <= test_end_dt
        available_prices = self.prices.loc[mask]
        available_volume = self.volume.loc[mask] if self.volume is not None else None

        test_mask = (available_prices.index >= test_start_dt)
        test_indices = available_prices.index[test_mask]

        if len(test_indices) < 5:
            return None

        portfolio_returns = []
        turnovers = []
        current_positions = []

        for i, date in enumerate(test_indices):
            rebalance_today = (i == 0 or i % self.rebalance_frequency == 0)

            if rebalance_today:
                date_idx = available_prices.index.get_loc(date)
                prices_to_date = available_prices.iloc[:date_idx + 1]
                volume_to_date = available_volume.iloc[:date_idx + 1] if available_volume is not None else None

                if self.tradeable_tickers:
                    scores = strategy.score_stocks(prices_to_date, volume=volume_to_date)
                    tradeable_in_prices = [t for t in self.tradeable_tickers if t in scores.index]

                    if tradeable_in_prices:
                        tradeable_scores = scores[tradeable_in_prices]

                        if tradeable_scores.nunique() <= 1:
                            return {
                                'period_start': test_start,
                                'period_end': test_end,
                                'total_return': -0.5,
                                'sharpe_ratio': -2.0,
                                'max_drawdown': 0.5,
                                'turnover': 1.0,
                                'n_days': 0,
                            }

                        tradeable_scores = tradeable_scores + np.random.uniform(0, 1e-6, len(tradeable_scores))
                        n_select = max(1, int(len(tradeable_scores) * strategy.top_pct / 100))
                        new_positions = tradeable_scores.nlargest(n_select).index.tolist()

                        if current_positions:
                            current_set = set(current_positions)
                            new_set = set(new_positions)
                            exited = len(current_set - new_set)
                            entered = len(new_set - current_set)
                            turnover = (exited + entered) / (2 * max(len(current_set), len(new_set), 1))
                        else:
                            turnover = 1.0
                    else:
                        new_positions = []
                        turnover = 0.0
                else:
                    new_positions, turnover = strategy.select_stocks(
                        prices_to_date,
                        volume=volume_to_date,
                        current_positions=current_positions,
                    )

                turnovers.append(turnover)
                current_positions = new_positions

            # Calculate return for this day
            if current_positions and i > 0:
                prev_date = test_indices[i - 1]
                day_return = (
                    available_prices.loc[date, current_positions] /
                    available_prices.loc[prev_date, current_positions] - 1
                ).mean()

                if rebalance_today and turnovers:
                    cost = turnovers[-1] * self.transaction_cost * 2
                    day_return -= cost

                portfolio_returns.append(day_return)

        if not portfolio_returns:
            return None

        returns_series = pd.Series(portfolio_returns)
        cumulative = (1 + returns_series).cumprod()
        total_return = cumulative.iloc[-1] - 1

        if returns_series.std() > 0:
            sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
        else:
            sharpe = 0

        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())
        avg_turnover = np.mean(turnovers) if turnovers else 0

        return {
            'period_start': test_start,
            'period_end': test_end,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'turnover': avg_turnover,
            'n_days': len(portfolio_returns),
        }
    
    def to_config(self) -> Dict[str, Any]:
        """Serialize evaluator to picklable config dict."""
        return {
            'prices': self.prices,
            'periods': self.periods,
            'benchmark_results': self.benchmark_results,
            'transaction_cost': self.transaction_cost,
            'rebalance_frequency': self.rebalance_frequency,
            'rebalancer': self.rebalancer,
            'stop_manager': self.stop_manager,
            'position_sizer': self.position_sizer,
            'use_calmar_fitness': self.use_calmar_fitness,
            # RC-3
            'universe_type': self.universe_type,
            'recency_half_life': self.recency_half_life,
            'use_fitness_v2': self.use_fitness_v2,
            # RC-4
            'tradeable_tickers': self.tradeable_tickers,
            # Expanding window
            'expanding_window': self.expanding_window,
            'volume': self.volume,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WalkForwardEvaluator':
        """Deserialize evaluator from config dict."""
        return cls(**config)
