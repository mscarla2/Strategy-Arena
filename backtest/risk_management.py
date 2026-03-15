"""
Risk Management Module for Microcap Trading

Implements Priority 1 risk management components:
1. Slippage Model: Realistic transaction costs for low-float stocks
2. Dilution Filter: Detect ATM offerings and dilution events
3. Liquidity Constraints: Enforce maximum position sizes based on ADV

These are CRITICAL for microcap trading to bridge the gap between
backtest returns and live trading reality.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SlippageResult:
    """Result of slippage calculation."""
    ticker: str
    order_size_dollars: float
    adv_dollars: float
    order_pct_adv: float
    base_slippage: float
    volume_impact: float
    commission_fee: float
    total_slippage: float
    total_cost_dollars: float


class MicrocapSlippageModel:
    """
    Model realistic slippage for low-float microcap stocks.
    
    Key Insight: Backtest returns of 10.5% can become 2-3% in live trading
    due to slippage. On low-float stocks (PRSO, BATL), your own buying
    pressure moves the market.
    
    Realistic slippage: 20-30 bps per trade (0.2-0.3%)
    Pink sheet commission: $6.95 per trade (fixed fee)
    """
    
    def __init__(self, base_slippage_bps: float = 25, volume_impact_factor: float = 0.5,
                 commission_per_trade: float = 6.95):
        """
        Args:
            base_slippage_bps: Base slippage in basis points (25 bps = 0.25%)
            volume_impact_factor: Additional slippage per % of ADV traded
            commission_per_trade: Fixed commission per trade ($6.95 for pink sheets)
        """
        self.base_slippage_bps = base_slippage_bps
        self.volume_impact_factor = volume_impact_factor
        self.commission_per_trade = commission_per_trade
    
    def calculate_slippage(
        self,
        ticker: str,
        order_size_dollars: float,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        lookback: int = 20
    ) -> SlippageResult:
        """
        Calculate expected slippage for an order including commission fees.
        
        Args:
            ticker: Stock ticker
            order_size_dollars: Order size in dollars
            prices: Price data (DataFrame with tickers as columns)
            volume: Volume data (DataFrame with tickers as columns)
            lookback: Days for ADV calculation
        
        Returns:
            SlippageResult with detailed breakdown
        """
        if ticker not in prices.columns or ticker not in volume.columns:
            # Return zero slippage if data not available
            return SlippageResult(
                ticker=ticker,
                order_size_dollars=order_size_dollars,
                adv_dollars=0,
                order_pct_adv=0,
                base_slippage=0,
                volume_impact=0,
                commission_fee=0,
                total_slippage=0,
                total_cost_dollars=0
            )
        
        # Calculate Average Daily Volume (ADV)
        adv_shares = volume[ticker].rolling(lookback).mean().iloc[-1]
        current_price = prices[ticker].iloc[-1]
        adv_dollars = adv_shares * current_price
        
        # Order size as % of ADV
        order_pct_adv = order_size_dollars / adv_dollars if adv_dollars > 0 else 0
        
        # Base slippage (percentage)
        base_slippage = self.base_slippage_bps / 10000
        
        # Volume impact (square root model, percentage)
        volume_impact = self.volume_impact_factor * np.sqrt(order_pct_adv)
        
        # Commission fee (percentage of order size)
        commission_fee_pct = self.commission_per_trade / order_size_dollars if order_size_dollars > 0 else 0
        
        # Total slippage (percentage)
        total_slippage = base_slippage + volume_impact + commission_fee_pct
        
        # Cap at 10% (extreme illiquidity + small orders)
        total_slippage = min(total_slippage, 0.10)
        
        # Total cost in dollars
        total_cost_dollars = order_size_dollars * total_slippage
        
        return SlippageResult(
            ticker=ticker,
            order_size_dollars=order_size_dollars,
            adv_dollars=adv_dollars,
            order_pct_adv=order_pct_adv,
            base_slippage=base_slippage,
            volume_impact=volume_impact,
            commission_fee=commission_fee_pct,
            total_slippage=total_slippage,
            total_cost_dollars=total_cost_dollars
        )
    
    def apply_slippage_to_returns(
        self,
        returns: float,
        turnover: float,
        avg_slippage: float,
        num_trades: int = None,
        portfolio_value: float = None
    ) -> float:
        """
        Adjust returns for slippage including commission fees.
        
        Args:
            returns: Strategy returns
            turnover: Portfolio turnover rate
            avg_slippage: Average slippage per trade (percentage)
            num_trades: Number of trades (for commission calculation)
            portfolio_value: Portfolio value (for commission calculation)
        
        Returns:
            Adjusted returns
        """
        # Slippage cost = turnover * slippage * 2 (buy and sell)
        slippage_cost = turnover * avg_slippage * 2
        
        # Add commission costs if provided
        if num_trades is not None and portfolio_value is not None and portfolio_value > 0:
            # Commission cost as percentage of portfolio
            commission_cost = (num_trades * self.commission_per_trade) / portfolio_value
            slippage_cost += commission_cost
        
        return returns - slippage_cost
    
    def calculate_commission_impact(self, order_size_dollars: float) -> Dict[str, float]:
        """
        Calculate the impact of fixed commission on different order sizes.
        
        For small orders, the $6.95 commission can be a significant percentage.
        For example:
        - $100 order: 6.95% commission
        - $500 order: 1.39% commission
        - $1000 order: 0.695% commission
        - $5000 order: 0.139% commission
        
        Args:
            order_size_dollars: Order size in dollars
        
        Returns:
            Dict with commission metrics
        """
        commission_pct = (self.commission_per_trade / order_size_dollars * 100) if order_size_dollars > 0 else 0
        
        return {
            'commission_dollars': self.commission_per_trade,
            'commission_pct': commission_pct,
            'order_size': order_size_dollars,
            'breakeven_move_pct': commission_pct * 2  # Need to overcome buy + sell commission
        }
    
    def calculate_portfolio_slippage(
        self,
        positions: Dict[str, float],
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        lookback: int = 20
    ) -> Tuple[float, List[SlippageResult]]:
        """
        Calculate average slippage across portfolio including commission fees.
        
        Args:
            positions: Dict of {ticker: position_size_dollars}
            prices: Price data
            volume: Volume data
            lookback: Days for ADV calculation
        
        Returns:
            (avg_slippage, list of SlippageResults)
        """
        results = []
        total_slippage = 0
        total_size = 0
        
        for ticker, size in positions.items():
            result = self.calculate_slippage(ticker, size, prices, volume, lookback)
            results.append(result)
            total_slippage += result.total_slippage * size
            total_size += size
        
        avg_slippage = total_slippage / total_size if total_size > 0 else 0
        
        return avg_slippage, results


class DilutionFilter:
    """
    Detect and respond to At-The-Market (ATM) offerings.
    
    Key Insight: TPET, BATL often have ATM offerings. When volume spikes
    (300M+ on Mar 3-5), the company may be selling shares directly into
    your buy orders. This causes -20% to -50% overnight losses.
    
    This is a CRITICAL filter for microcaps.
    """
    
    def __init__(
        self,
        volume_spike_threshold: float = 3.0,
        price_drop_threshold: float = -0.10,
        lookback: int = 20
    ):
        """
        Args:
            volume_spike_threshold: Volume / avg_volume ratio (3x, lowered from 5x for microcaps)
            price_drop_threshold: Price drop threshold (-10%)
            lookback: Days for average volume calculation
        """
        self.volume_spike_threshold = volume_spike_threshold
        self.price_drop_threshold = price_drop_threshold
        self.lookback = lookback
    
    def detect_dilution_event(
        self,
        prices: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Detect potential ATM offering or dilution event.
        
        Args:
            prices: Price series for a single ticker
            volume: Volume series for a single ticker
        
        Returns:
            Boolean series indicating dilution events
        """
        # Calculate average volume
        avg_volume = volume.rolling(self.lookback).mean()
        
        # Volume spike
        volume_ratio = volume / avg_volume
        volume_spike = volume_ratio > self.volume_spike_threshold
        
        # Price drop
        returns = prices.pct_change()
        price_drop = returns < self.price_drop_threshold
        
        # Dilution event = volume spike + price drop
        dilution_event = volume_spike & price_drop
        
        return dilution_event
    
    def should_exit(
        self,
        ticker: str,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        current_positions: List[str],
        check_days: int = 3
    ) -> bool:
        """
        Check if we should exit due to dilution.
        
        Args:
            ticker: Stock ticker
            prices: Price data (DataFrame with tickers as columns)
            volume: Volume data (DataFrame with tickers as columns)
            current_positions: List of currently held tickers
            check_days: Number of recent days to check
        
        Returns:
            True if should exit
        """
        if ticker not in current_positions:
            return False
        
        if ticker not in prices.columns or ticker not in volume.columns:
            return False
        
        dilution = self.detect_dilution_event(
            prices[ticker], volume[ticker]
        )
        
        # Exit if dilution detected in last N days
        recent_dilution = dilution.iloc[-check_days:].any()
        
        return recent_dilution
    
    def check_all_positions(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        current_positions: List[str],
        check_days: int = 3
    ) -> List[str]:
        """
        Check all positions for dilution events.
        
        Args:
            prices: Price data
            volume: Volume data
            current_positions: List of currently held tickers
            check_days: Number of recent days to check
        
        Returns:
            List of tickers to exit
        """
        exits = []
        
        for ticker in current_positions:
            if self.should_exit(ticker, prices, volume, current_positions, check_days):
                exits.append(ticker)
        
        return exits


class LiquidityConstraint:
    """
    Enforce maximum position size based on liquidity.
    
    Key Insight: Can't trade more than X% of ADV without severe slippage.
    Industry standard: 5-10% of ADV for microcaps.
    
    Also filters out stocks with insufficient liquidity.
    """
    
    def __init__(
        self,
        max_pct_adv: float = 0.05,
        min_adv_dollars: float = 50000,
        lookback: int = 20
    ):
        """
        Args:
            max_pct_adv: Maximum % of ADV to trade (5%)
            min_adv_dollars: Minimum ADV in dollars ($50k)
            lookback: Days for ADV calculation
        """
        self.max_pct_adv = max_pct_adv
        self.min_adv_dollars = min_adv_dollars
        self.lookback = lookback
    
    def calculate_adv(
        self,
        ticker: str,
        prices: pd.DataFrame,
        volume: pd.DataFrame
    ) -> float:
        """
        Calculate Average Daily Volume in dollars.
        
        Args:
            ticker: Stock ticker
            prices: Price data
            volume: Volume data
        
        Returns:
            ADV in dollars
        """
        if ticker not in prices.columns or ticker not in volume.columns:
            return 0
        
        adv_shares = volume[ticker].rolling(self.lookback).mean().iloc[-1]
        current_price = prices[ticker].iloc[-1]
        adv_dollars = adv_shares * current_price
        
        return adv_dollars
    
    def filter_by_liquidity(
        self,
        tickers: List[str],
        prices: pd.DataFrame,
        volume: pd.DataFrame
    ) -> List[str]:
        """
        Filter tickers by liquidity requirements.
        
        Args:
            tickers: List of tickers to filter
            prices: Price data
            volume: Volume data
        
        Returns:
            List of tickers that meet liquidity requirements
        """
        liquid_tickers = []
        
        for ticker in tickers:
            adv_dollars = self.calculate_adv(ticker, prices, volume)
            
            # Check minimum ADV
            if adv_dollars >= self.min_adv_dollars:
                liquid_tickers.append(ticker)
        
        return liquid_tickers
    
    def calculate_max_position_size(
        self,
        ticker: str,
        prices: pd.DataFrame,
        volume: pd.DataFrame
    ) -> float:
        """
        Calculate maximum position size based on liquidity.
        
        Args:
            ticker: Stock ticker
            prices: Price data
            volume: Volume data
        
        Returns:
            Maximum position size in dollars
        """
        adv_dollars = self.calculate_adv(ticker, prices, volume)
        
        # Max position = max_pct_adv * ADV
        max_position = self.max_pct_adv * adv_dollars
        
        return max_position
    
    def enforce_constraints(
        self,
        positions: Dict[str, float],
        prices: pd.DataFrame,
        volume: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Enforce liquidity constraints on positions.
        
        Args:
            positions: Dict of {ticker: position_size_dollars}
            prices: Price data
            volume: Volume data
        
        Returns:
            Adjusted positions
        """
        adjusted_positions = {}
        
        for ticker, position_size in positions.items():
            max_size = self.calculate_max_position_size(ticker, prices, volume)
            adjusted_size = min(position_size, max_size)
            
            # Only include if size > 0
            if adjusted_size > 0:
                adjusted_positions[ticker] = adjusted_size
        
        return adjusted_positions
    
    def get_liquidity_report(
        self,
        tickers: List[str],
        prices: pd.DataFrame,
        volume: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate liquidity report for tickers.
        
        Args:
            tickers: List of tickers
            prices: Price data
            volume: Volume data
        
        Returns:
            DataFrame with liquidity metrics
        """
        report_data = []
        
        for ticker in tickers:
            adv_dollars = self.calculate_adv(ticker, prices, volume)
            max_position = self.calculate_max_position_size(ticker, prices, volume)
            meets_min = adv_dollars >= self.min_adv_dollars
            
            report_data.append({
                'ticker': ticker,
                'adv_dollars': adv_dollars,
                'max_position': max_position,
                'meets_minimum': meets_min
            })
        
        return pd.DataFrame(report_data)


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for a strategy."""
    # Returns
    total_return: float
    adjusted_return: float  # After slippage
    slippage_cost: float
    
    # Slippage details
    avg_slippage_bps: float
    max_slippage_bps: float
    total_slippage_cost: float
    
    # Dilution events
    dilution_events: int
    dilution_exits: List[str]
    
    # Liquidity
    liquid_tickers: List[str]
    illiquid_tickers: List[str]
    avg_adv_dollars: float
    
    # Position sizing
    avg_position_pct_adv: float
    max_position_pct_adv: float


class RiskManager:
    """
    Integrated risk management system combining all Priority 1 components.
    """
    
    def __init__(
        self,
        slippage_model: Optional[MicrocapSlippageModel] = None,
        dilution_filter: Optional[DilutionFilter] = None,
        liquidity_constraint: Optional[LiquidityConstraint] = None
    ):
        """
        Args:
            slippage_model: Slippage model instance
            dilution_filter: Dilution filter instance
            liquidity_constraint: Liquidity constraint instance
        """
        self.slippage_model = slippage_model or MicrocapSlippageModel()
        self.dilution_filter = dilution_filter or DilutionFilter()
        self.liquidity_constraint = liquidity_constraint or LiquidityConstraint()
    
    def filter_universe(
        self,
        tickers: List[str],
        prices: pd.DataFrame,
        volume: pd.DataFrame
    ) -> List[str]:
        """
        Filter universe by liquidity constraints.
        
        Args:
            tickers: List of tickers
            prices: Price data
            volume: Volume data
        
        Returns:
            Filtered list of liquid tickers
        """
        return self.liquidity_constraint.filter_by_liquidity(tickers, prices, volume)
    
    def check_dilution_exits(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        current_positions: List[str]
    ) -> List[str]:
        """
        Check for dilution-based exits.
        
        Args:
            prices: Price data
            volume: Volume data
            current_positions: Currently held positions
        
        Returns:
            List of tickers to exit
        """
        return self.dilution_filter.check_all_positions(
            prices, volume, current_positions
        )
    
    def calculate_adjusted_returns(
        self,
        returns: float,
        turnover: float,
        positions: Dict[str, float],
        prices: pd.DataFrame,
        volume: pd.DataFrame
    ) -> Tuple[float, float, List[SlippageResult]]:
        """
        Calculate returns adjusted for slippage.
        
        Args:
            returns: Unadjusted returns
            turnover: Portfolio turnover
            positions: Position sizes
            prices: Price data
            volume: Volume data
        
        Returns:
            (adjusted_returns, slippage_cost, slippage_details)
        """
        # Calculate portfolio slippage
        avg_slippage, slippage_results = self.slippage_model.calculate_portfolio_slippage(
            positions, prices, volume
        )
        
        # Adjust returns
        adjusted_returns = self.slippage_model.apply_slippage_to_returns(
            returns, turnover, avg_slippage
        )
        
        slippage_cost = returns - adjusted_returns
        
        return adjusted_returns, slippage_cost, slippage_results
    
    def enforce_position_limits(
        self,
        positions: Dict[str, float],
        prices: pd.DataFrame,
        volume: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Enforce liquidity-based position limits.
        
        Args:
            positions: Desired positions
            prices: Price data
            volume: Volume data
        
        Returns:
            Adjusted positions
        """
        return self.liquidity_constraint.enforce_constraints(
            positions, prices, volume
        )
    
    def generate_risk_report(
        self,
        returns: float,
        turnover: float,
        positions: Dict[str, float],
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        current_positions: List[str],
        all_tickers: List[str]
    ) -> RiskMetrics:
        """
        Generate comprehensive risk metrics.
        
        Args:
            returns: Unadjusted returns
            turnover: Portfolio turnover
            positions: Position sizes
            prices: Price data
            volume: Volume data
            current_positions: Currently held positions
            all_tickers: All tickers in universe
        
        Returns:
            RiskMetrics object
        """
        # Calculate slippage
        adjusted_returns, slippage_cost, slippage_results = self.calculate_adjusted_returns(
            returns, turnover, positions, prices, volume
        )
        
        avg_slippage_bps = np.mean([r.total_slippage for r in slippage_results]) * 10000
        max_slippage_bps = max([r.total_slippage for r in slippage_results]) * 10000 if slippage_results else 0
        
        # Check dilution
        dilution_exits = self.check_dilution_exits(prices, volume, current_positions)
        
        # Check liquidity
        liquid_tickers = self.filter_universe(all_tickers, prices, volume)
        illiquid_tickers = [t for t in all_tickers if t not in liquid_tickers]
        
        # Calculate ADV metrics
        advs = [self.liquidity_constraint.calculate_adv(t, prices, volume) for t in all_tickers]
        avg_adv = np.mean([a for a in advs if a > 0]) if advs else 0
        
        # Calculate position % of ADV
        pct_advs = []
        for ticker, size in positions.items():
            adv = self.liquidity_constraint.calculate_adv(ticker, prices, volume)
            if adv > 0:
                pct_advs.append(size / adv)
        
        avg_pct_adv = np.mean(pct_advs) if pct_advs else 0
        max_pct_adv = max(pct_advs) if pct_advs else 0
        
        return RiskMetrics(
            total_return=returns,
            adjusted_return=adjusted_returns,
            slippage_cost=slippage_cost,
            avg_slippage_bps=avg_slippage_bps,
            max_slippage_bps=max_slippage_bps,
            total_slippage_cost=slippage_cost,
            dilution_events=len(dilution_exits),
            dilution_exits=dilution_exits,
            liquid_tickers=liquid_tickers,
            illiquid_tickers=illiquid_tickers,
            avg_adv_dollars=avg_adv,
            avg_position_pct_adv=avg_pct_adv,
            max_position_pct_adv=max_pct_adv
        )
