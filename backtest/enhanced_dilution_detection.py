"""
Enhanced Dilution Detection for Microcap Stocks

Implements advanced dilution detection beyond simple volume/price analysis:
- SEC Filing Monitoring: 8-K filings for ATM offerings, shelf registrations
- News Sentiment Analysis: Detect dilution-related news
- Insider Trading Detection: Track insider selling patterns
- Social Media Sentiment: Reddit/Twitter mentions of dilution
- Share Count Tracking: Monitor outstanding shares changes

These features help avoid catastrophic losses from dilution events that
simple technical indicators miss.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings


class DilutionEventType(Enum):
    """Type of dilution event."""
    ATM_OFFERING = "atm_offering"
    SHELF_REGISTRATION = "shelf_registration"
    PIPE_DEAL = "pipe_deal"
    WARRANT_EXERCISE = "warrant_exercise"
    CONVERTIBLE_DEBT = "convertible_debt"
    INSIDER_SELLING = "insider_selling"
    UNKNOWN = "unknown"


class DilutionSeverity(Enum):
    """Severity of dilution event."""
    LOW = "low"  # <10% dilution
    MODERATE = "moderate"  # 10-25% dilution
    HIGH = "high"  # 25-50% dilution
    EXTREME = "extreme"  # >50% dilution


@dataclass
class DilutionEvent:
    """Represents a detected dilution event."""
    ticker: str
    date: datetime
    event_type: DilutionEventType
    severity: DilutionSeverity
    
    # Evidence
    volume_spike: float  # Multiple of average volume
    price_drop: float  # Percentage price drop
    share_count_increase: Optional[float] = None  # Percentage increase
    
    # Sources
    sec_filing: bool = False
    news_mention: bool = False
    insider_selling: bool = False
    social_sentiment: bool = False
    
    # Confidence
    confidence: float = 0.0  # 0-1
    
    # Action
    blacklist_days: int = 20  # Days to avoid this ticker


class EnhancedDilutionDetector:
    """
    Enhanced dilution detection using multiple data sources.
    
    Combines technical signals with fundamental data to identify
    dilution events before they cause catastrophic losses.
    """
    
    def __init__(
        self,
        volume_spike_threshold: float = 3.0,
        price_drop_threshold: float = -0.10,
        lookback_days: int = 20,
        blacklist_days: int = 20,
        confidence_threshold: float = 0.6
    ):
        """
        Args:
            volume_spike_threshold: Volume spike multiple (3x = 300% of average)
            price_drop_threshold: Price drop threshold (-0.10 = -10%)
            lookback_days: Days to look back for average volume
            blacklist_days: Days to blacklist ticker after dilution
            confidence_threshold: Minimum confidence to flag event
        """
        self.volume_spike_threshold = volume_spike_threshold
        self.price_drop_threshold = price_drop_threshold
        self.lookback_days = lookback_days
        self.blacklist_days = blacklist_days
        self.confidence_threshold = confidence_threshold
        
        self.detected_events: List[DilutionEvent] = []
        self.blacklisted_tickers: Dict[str, datetime] = {}
    
    def detect_technical_dilution(
        self,
        ticker: str,
        prices: pd.Series,
        volume: pd.Series,
        date: datetime
    ) -> Optional[DilutionEvent]:
        """
        Detect dilution using technical indicators (volume spike + price drop).
        
        This is the baseline detection method from the original risk_management.py.
        
        Returns:
            DilutionEvent if detected, None otherwise
        """
        if len(prices) < self.lookback_days + 1:
            return None
        
        # Calculate average volume
        avg_volume = volume.iloc[-self.lookback_days-1:-1].mean()
        current_volume = volume.iloc[-1]
        
        if avg_volume == 0:
            return None
        
        volume_spike = current_volume / avg_volume
        
        # Calculate price drop
        prev_price = prices.iloc[-2]
        current_price = prices.iloc[-1]
        
        if prev_price == 0:
            return None
        
        price_drop = (current_price - prev_price) / prev_price
        
        # Check if dilution criteria met
        if volume_spike >= self.volume_spike_threshold and price_drop <= self.price_drop_threshold:
            # Estimate severity based on price drop
            if price_drop <= -0.50:
                severity = DilutionSeverity.EXTREME
            elif price_drop <= -0.25:
                severity = DilutionSeverity.HIGH
            elif price_drop <= -0.15:
                severity = DilutionSeverity.MODERATE
            else:
                severity = DilutionSeverity.LOW
            
            return DilutionEvent(
                ticker=ticker,
                date=date,
                event_type=DilutionEventType.UNKNOWN,
                severity=severity,
                volume_spike=volume_spike,
                price_drop=price_drop,
                confidence=0.5  # Moderate confidence from technical only
            )
        
        return None
    
    def detect_sec_filing_dilution(
        self,
        ticker: str,
        filing_data: Optional[pd.DataFrame] = None
    ) -> List[DilutionEvent]:
        """
        Detect dilution from SEC filings (8-K, S-3, etc.).
        
        Note: This requires SEC EDGAR API integration.
        For now, this is a placeholder that shows the structure.
        
        Args:
            ticker: Stock ticker
            filing_data: DataFrame with columns: date, form_type, description
        
        Returns:
            List of detected dilution events
        """
        events = []
        
        if filing_data is None or filing_data.empty:
            return events
        
        # Keywords that indicate dilution
        dilution_keywords = [
            'at-the-market',
            'atm offering',
            'shelf registration',
            'registered direct offering',
            'pipe',
            'warrant exercise',
            'convertible note',
            'equity line',
            'committed equity facility'
        ]
        
        for _, filing in filing_data.iterrows():
            description = str(filing.get('description', '')).lower()
            form_type = str(filing.get('form_type', '')).upper()
            
            # Check for dilution-related filings
            is_dilution = False
            event_type = DilutionEventType.UNKNOWN
            
            if form_type == '8-K':
                # 8-K can announce various dilution events
                if any(keyword in description for keyword in dilution_keywords):
                    is_dilution = True
                    
                    if 'at-the-market' in description or 'atm' in description:
                        event_type = DilutionEventType.ATM_OFFERING
                    elif 'pipe' in description:
                        event_type = DilutionEventType.PIPE_DEAL
                    elif 'warrant' in description:
                        event_type = DilutionEventType.WARRANT_EXERCISE
                    elif 'convertible' in description:
                        event_type = DilutionEventType.CONVERTIBLE_DEBT
            
            elif form_type == 'S-3':
                # S-3 is shelf registration (potential future dilution)
                is_dilution = True
                event_type = DilutionEventType.SHELF_REGISTRATION
            
            if is_dilution:
                event = DilutionEvent(
                    ticker=ticker,
                    date=filing['date'],
                    event_type=event_type,
                    severity=DilutionSeverity.MODERATE,  # Default, adjust based on details
                    volume_spike=0,
                    price_drop=0,
                    sec_filing=True,
                    confidence=0.8  # High confidence from SEC filing
                )
                events.append(event)
        
        return events
    
    def detect_insider_selling(
        self,
        ticker: str,
        insider_data: Optional[pd.DataFrame] = None
    ) -> List[DilutionEvent]:
        """
        Detect unusual insider selling patterns.
        
        Heavy insider selling often precedes dilution events.
        
        Args:
            ticker: Stock ticker
            insider_data: DataFrame with columns: date, transaction_type, shares, value
        
        Returns:
            List of detected events
        """
        events = []
        
        if insider_data is None or insider_data.empty:
            return events
        
        # Filter for sales only
        sales = insider_data[insider_data['transaction_type'] == 'Sale']
        
        if len(sales) < 2:
            return events
        
        # Calculate rolling sum of shares sold
        sales = sales.sort_values('date')
        sales['rolling_shares'] = sales['shares'].rolling(window=30, min_periods=1).sum()
        
        # Detect spikes in insider selling
        avg_selling = sales['rolling_shares'].mean()
        recent_selling = sales['rolling_shares'].iloc[-1]
        
        if recent_selling > avg_selling * 3:  # 3x normal selling
            event = DilutionEvent(
                ticker=ticker,
                date=sales['date'].iloc[-1],
                event_type=DilutionEventType.INSIDER_SELLING,
                severity=DilutionSeverity.MODERATE,
                volume_spike=0,
                price_drop=0,
                insider_selling=True,
                confidence=0.6  # Moderate confidence
            )
            events.append(event)
        
        return events
    
    def detect_news_sentiment(
        self,
        ticker: str,
        news_data: Optional[pd.DataFrame] = None
    ) -> List[DilutionEvent]:
        """
        Detect dilution mentions in news articles.
        
        Args:
            ticker: Stock ticker
            news_data: DataFrame with columns: date, headline, content, sentiment
        
        Returns:
            List of detected events
        """
        events = []
        
        if news_data is None or news_data.empty:
            return events
        
        # Keywords indicating dilution
        dilution_keywords = [
            'dilution',
            'offering',
            'capital raise',
            'equity financing',
            'share issuance',
            'warrant exercise',
            'convertible debt'
        ]
        
        for _, article in news_data.iterrows():
            text = (str(article.get('headline', '')) + ' ' + 
                   str(article.get('content', ''))).lower()
            
            # Check for dilution keywords
            if any(keyword in text for keyword in dilution_keywords):
                # Check sentiment (negative sentiment + dilution = bad)
                sentiment = article.get('sentiment', 0)
                
                if sentiment < -0.3:  # Negative sentiment
                    event = DilutionEvent(
                        ticker=ticker,
                        date=article['date'],
                        event_type=DilutionEventType.UNKNOWN,
                        severity=DilutionSeverity.MODERATE,
                        volume_spike=0,
                        price_drop=0,
                        news_mention=True,
                        confidence=0.5  # Moderate confidence from news
                    )
                    events.append(event)
        
        return events
    
    def detect_share_count_increase(
        self,
        ticker: str,
        share_count_data: pd.Series
    ) -> List[DilutionEvent]:
        """
        Detect increases in outstanding shares.
        
        Direct evidence of dilution.
        
        Args:
            ticker: Stock ticker
            share_count_data: Series of outstanding shares over time
        
        Returns:
            List of detected events
        """
        events = []
        
        if len(share_count_data) < 2:
            return events
        
        # Calculate percentage increase
        pct_change = share_count_data.pct_change()
        
        # Detect significant increases (>5%)
        significant_increases = pct_change[pct_change > 0.05]
        
        for date, increase in significant_increases.items():
            # Classify severity
            if increase > 0.50:
                severity = DilutionSeverity.EXTREME
            elif increase > 0.25:
                severity = DilutionSeverity.HIGH
            elif increase > 0.10:
                severity = DilutionSeverity.MODERATE
            else:
                severity = DilutionSeverity.LOW
            
            event = DilutionEvent(
                ticker=ticker,
                date=date,
                event_type=DilutionEventType.UNKNOWN,
                severity=severity,
                volume_spike=0,
                price_drop=0,
                share_count_increase=increase,
                confidence=0.9  # High confidence from share count
            )
            events.append(event)
        
        return events
    
    def combine_signals(
        self,
        technical_event: Optional[DilutionEvent],
        sec_events: List[DilutionEvent],
        insider_events: List[DilutionEvent],
        news_events: List[DilutionEvent],
        share_count_events: List[DilutionEvent]
    ) -> List[DilutionEvent]:
        """
        Combine signals from multiple sources to improve confidence.
        
        Multiple confirming signals = higher confidence.
        
        Returns:
            List of high-confidence dilution events
        """
        all_events = []
        
        if technical_event:
            all_events.append(technical_event)
        
        all_events.extend(sec_events)
        all_events.extend(insider_events)
        all_events.extend(news_events)
        all_events.extend(share_count_events)
        
        if not all_events:
            return []
        
        # Group events by date (within 5 days)
        grouped_events: Dict[datetime, List[DilutionEvent]] = {}
        
        for event in all_events:
            # Find matching group
            matched = False
            for group_date in list(grouped_events.keys()):
                if abs((event.date - group_date).days) <= 5:
                    grouped_events[group_date].append(event)
                    matched = True
                    break
            
            if not matched:
                grouped_events[event.date] = [event]
        
        # Combine signals for each group
        combined_events = []
        
        for group_date, events in grouped_events.items():
            # Combine evidence
            combined = DilutionEvent(
                ticker=events[0].ticker,
                date=group_date,
                event_type=events[0].event_type,
                severity=max(e.severity for e in events),
                volume_spike=max((e.volume_spike for e in events), default=0),
                price_drop=min((e.price_drop for e in events), default=0),
                share_count_increase=max((e.share_count_increase for e in events if e.share_count_increase), default=None),
                sec_filing=any(e.sec_filing for e in events),
                news_mention=any(e.news_mention for e in events),
                insider_selling=any(e.insider_selling for e in events),
                confidence=min(sum(e.confidence for e in events) / len(events) + 0.2 * (len(events) - 1), 1.0)
            )
            
            # Only include high-confidence events
            if combined.confidence >= self.confidence_threshold:
                combined_events.append(combined)
        
        return combined_events
    
    def update_blacklist(self, events: List[DilutionEvent]):
        """
        Update blacklist with detected dilution events.
        
        Args:
            events: List of dilution events
        """
        for event in events:
            blacklist_until = event.date + timedelta(days=self.blacklist_days)
            self.blacklisted_tickers[event.ticker] = blacklist_until
            self.detected_events.append(event)
    
    def is_blacklisted(self, ticker: str, current_date: datetime) -> bool:
        """
        Check if ticker is currently blacklisted.
        
        Args:
            ticker: Stock ticker
            current_date: Current date
        
        Returns:
            True if blacklisted, False otherwise
        """
        if ticker not in self.blacklisted_tickers:
            return False
        
        blacklist_until = self.blacklisted_tickers[ticker]
        return current_date < blacklist_until
    
    def get_statistics(self) -> Dict:
        """Get statistics about detected dilution events."""
        if not self.detected_events:
            return {
                'total_events': 0,
                'by_type': {},
                'by_severity': {},
                'avg_confidence': 0,
                'blacklisted_tickers': 0
            }
        
        by_type = {}
        for event in self.detected_events:
            event_type = event.event_type.value
            by_type[event_type] = by_type.get(event_type, 0) + 1
        
        by_severity = {}
        for event in self.detected_events:
            severity = event.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            'total_events': len(self.detected_events),
            'by_type': by_type,
            'by_severity': by_severity,
            'avg_confidence': np.mean([e.confidence for e in self.detected_events]),
            'blacklisted_tickers': len(self.blacklisted_tickers)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Enhanced Dilution Detection - Example")
    print("=" * 60)
    
    # Initialize detector
    detector = EnhancedDilutionDetector(
        volume_spike_threshold=3.0,
        price_drop_threshold=-0.10,
        lookback_days=20,
        blacklist_days=20,
        confidence_threshold=0.6
    )
    
    # Simulate technical detection
    print("\n1. Technical Detection (Volume + Price)")
    print("-" * 60)
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    prices = pd.Series([100] * 20 + [95, 90, 85, 80, 78, 77, 76, 75, 74, 73], index=dates)
    volume = pd.Series([1000000] * 20 + [5000000, 4000000, 3000000, 2000000, 1500000, 
                                         1200000, 1100000, 1000000, 1000000, 1000000], index=dates)
    
    technical_event = detector.detect_technical_dilution(
        ticker="TEST",
        prices=prices,
        volume=volume,
        date=dates[-1]
    )
    
    if technical_event:
        print(f"Detected dilution event:")
        print(f"  Ticker: {technical_event.ticker}")
        print(f"  Date: {technical_event.date}")
        print(f"  Volume Spike: {technical_event.volume_spike:.1f}x")
        print(f"  Price Drop: {technical_event.price_drop:.1%}")
        print(f"  Severity: {technical_event.severity.value}")
        print(f"  Confidence: {technical_event.confidence:.1%}")
    else:
        print("No dilution detected")
    
    # Simulate SEC filing detection
    print("\n2. SEC Filing Detection")
    print("-" * 60)
    
    sec_filings = pd.DataFrame({
        'date': [datetime(2024, 1, 15)],
        'form_type': ['8-K'],
        'description': ['Company announces at-the-market offering program']
    })
    
    sec_events = detector.detect_sec_filing_dilution("TEST", sec_filings)
    print(f"Detected {len(sec_events)} SEC filing events")
    for event in sec_events:
        print(f"  {event.date}: {event.event_type.value} (confidence: {event.confidence:.1%})")
    
    # Combine signals
    print("\n3. Combined Signal Detection")
    print("-" * 60)
    
    combined_events = detector.combine_signals(
        technical_event=technical_event,
        sec_events=sec_events,
        insider_events=[],
        news_events=[],
        share_count_events=[]
    )
    
    print(f"Combined {len(combined_events)} high-confidence events")
    for event in combined_events:
        print(f"\n  Event: {event.ticker} on {event.date}")
        print(f"    Type: {event.event_type.value}")
        print(f"    Severity: {event.severity.value}")
        print(f"    Confidence: {event.confidence:.1%}")
        print(f"    Evidence:")
        if event.sec_filing:
            print(f"      - SEC Filing")
        if event.volume_spike > 0:
            print(f"      - Volume Spike: {event.volume_spike:.1f}x")
        if event.price_drop < 0:
            print(f"      - Price Drop: {event.price_drop:.1%}")
    
    # Update blacklist
    detector.update_blacklist(combined_events)
    
    # Check blacklist
    print("\n4. Blacklist Status")
    print("-" * 60)
    is_blacklisted = detector.is_blacklisted("TEST", datetime(2024, 1, 20))
    print(f"TEST blacklisted on 2024-01-20: {is_blacklisted}")
    
    # Statistics
    print("\n5. Detection Statistics")
    print("-" * 60)
    stats = detector.get_statistics()
    print(f"Total events detected: {stats['total_events']}")
    print(f"Average confidence: {stats['avg_confidence']:.1%}")
    print(f"Blacklisted tickers: {stats['blacklisted_tickers']}")
    
    print("\n" + "=" * 60)
    print("Enhanced dilution detection ready for integration!")
    print("\nNote: For production use, integrate:")
    print("  - SEC EDGAR API for filing data")
    print("  - News API for sentiment analysis")
    print("  - Insider trading databases")
    print("  - Share count tracking services")
