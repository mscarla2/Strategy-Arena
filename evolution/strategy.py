# evolution/strategy.py
"""
GP Strategy — expression-tree-based stock scoring strategy.

Contains: GPStrategy
"""

import uuid
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from evolution.nodes import Node


# ═══════════════════════════════════════════════════════════════════════════════
# GP STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPStrategy:
    """GP Strategy with expression tree."""
    
    tree: Node
    top_pct: float = 20.0
    holding_period: int = 21
    execution_lag: int = 1
    
    generation: int = 0
    origin: str = "random"
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    fitness: float = 0.0
    period_metrics: Optional[List[Dict]] = field(default=None, repr=False)
    # Feature library configuration
    enable_oil: bool = False
    
    def __post_init__(self):
        from evolution.features import FeatureLibrary
        self.feature_lib = FeatureLibrary(enable_oil=self.enable_oil)
        self._last_positions: Optional[List[str]] = None
        self._position_ages: Dict[str, int] = {}

    def copy(self) -> 'GPStrategy':
        """Deep copy."""
        new_strategy = GPStrategy(
            tree=self.tree.copy(),
            top_pct=self.top_pct,
            holding_period=self.holding_period,
            execution_lag=self.execution_lag,
            generation=self.generation,
            origin=self.origin,
        )
        new_strategy.fitness = self.fitness
        if self.period_metrics:
            new_strategy.period_metrics = list(self.period_metrics)
        return new_strategy

    def score_stocks(self, prices: pd.DataFrame, current_date_idx: int = -1, volume: pd.DataFrame = None) -> pd.Series:
        """Score stocks with proper execution lag."""
        if prices.empty:
            return pd.Series(dtype=float)
        
        if self.execution_lag > 0:
            if current_date_idx == -1:
                effective_idx = len(prices) - 1 - self.execution_lag
            else:
                effective_idx = current_date_idx - self.execution_lag
            
            if effective_idx < self.feature_lib.max_lookback:
                return pd.Series(0.5, index=prices.columns)
            
            lagged_prices = prices.iloc[:effective_idx + 1]
            lagged_volume = volume.iloc[:effective_idx + 1] if volume is not None else None  # ← add
        else:
            lagged_prices = prices
            lagged_volume = volume  # ← add
        
        features = self.feature_lib.compute_all(
            lagged_prices,
            volume=lagged_volume,
            lag=0
        )
        
        try:
            scores = self.tree.evaluate(features)
            
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                scores = (scores - min_s) / (max_s - min_s)
            else:
                scores = pd.Series(0.5, index=scores.index)
            
            return scores.fillna(0.5)
        except Exception:
            return pd.Series(0.5, index=prices.columns)
    
    def select_stocks(
        self, 
        prices: pd.DataFrame,
        volume: pd.DataFrame = None,
        current_positions: List[str] = None
    ) -> Tuple[List[str], float]:
        """Select stocks with turnover tracking."""
        scores = self.score_stocks(prices, volume=volume)
        if scores.empty:
            return [], 0.0
        
        n_select = max(1, int(len(scores) * self.top_pct / 100))
        top_stocks = scores.nlargest(n_select).index.tolist()
        
        if current_positions:
            current_set = set(current_positions)
            new_set = set(top_stocks)
            
            exited = len(current_set - new_set)
            entered = len(new_set - current_set)
            
            turnover = (exited + entered) / (2 * max(len(current_set), len(new_set), 1))
        else:
            turnover = 1.0
        
        return top_stocks, turnover
    
    def get_formula(self) -> str:
        """Get human-readable formula."""
        return self.tree.to_string()
    
    def complexity(self) -> int:
        """Tree complexity."""
        return self.tree.size()
