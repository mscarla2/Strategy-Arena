"""
Phase 1 — Semantic Parser
Extracts ticker symbols, price levels, and session type from raw watchlist text.
Uses regex for tickers and a rule-based price-level extractor to avoid LLM dependency.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import RawWatchlist, SessionType, WatchlistEntry

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Tickers: 1–5 uppercase letters preceded by $ sign (most reliable signal),
# OR bare 2-5 uppercase letters followed by a colon (e.g. "AAPL: watch 10")
_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?=:)")

# Price levels: matches $1.08, 1.08, $.85, .7350 style prices
_PRICE_RE = re.compile(
    r"\$?(\d{0,5}\.?\d{1,4})\b"  # captures the numeric part
)


def _parse_price(m: re.Match) -> Optional[float]:
    """Extract and validate a price from a _PRICE_RE match. Returns None if invalid."""
    try:
        val = float(m.group(1))
        return val if 0.001 < val < 100000 else None
    except (ValueError, TypeError):
        return None

# Support keywords
_SUPPORT_KW = re.compile(
    r"support|sppt|sup|floor|hold[s]?|bounce|base|key level|demand",
    re.IGNORECASE,
)

# Resistance keywords
_RESIST_KW = re.compile(
    r"resist[ance]*|res|target|tgt|ceiling|r\/r|supply|break[out]*",
    re.IGNORECASE,
)

# Stop-loss keywords
_STOP_KW = re.compile(
    r"stop|sl|cut|invalidat[e]*|below",
    re.IGNORECASE,
)

# Session detection
_PREMARKET_KW = re.compile(r"pre[\s\-]?market|pm\b|premarket|before open", re.IGNORECASE)
_AH_KW = re.compile(r"after[\s\-]?hours?|ah\b|extended hours?|post[\s\-]?market", re.IGNORECASE)
_OPEN_KW = re.compile(r"market open|open[ing]?|rth|regular hours?|9:30", re.IGNORECASE)

# Words that look like tickers but aren't — common English words, trading slang,
# disclaimers, and other non-symbol tokens that appear in watchlist posts.
_TICKER_BLACKLIST = {
    # Articles / prepositions / conjunctions
    "THE", "AND", "FOR", "BUT", "NOT", "ARE", "WAS", "HAS", "HAD",
    "ITS", "ALL", "NEW", "GET", "SET", "WITH", "FROM", "INTO", "OVER",
    "SOME", "THIS", "THAT", "THEN", "ALSO", "WHEN", "WILL", "BOTH",
    "NEXT", "LAST", "THEM", "THEY", "THEIR", "THAN", "ONLY", "THESE",
    "THOSE", "JUST", "BEEN", "HAVE", "WERE", "DOES", "WHAT", "WHICH",
    "SUCH", "EACH", "EVEN", "VERY", "MUCH", "MOST", "MANY", "MORE",
    "LESS", "BACK", "TOOK", "MAKE", "MADE", "TAKE", "TOOK", "GIVE",
    "GAVE", "GOES", "GOING", "COME", "CAME", "KEEP", "KEPT", "TELL",
    "TOLD", "KNOW", "KNEW", "SHOW", "SHOWN", "SAID", "SAYS", "SAYS",
    "NEED", "NEEDS", "WANT", "WANTS", "FEEL", "FELT",
    # Pronouns
    "YOUR", "THEIR", "OUR", "ITS", "HIS", "HER", "OWN", "MINE",
    "YOU", "HIM", "SHE", "WE", "MY", "ME",
    # Common verb forms
    "DO", "DID", "DOES", "DONE", "IS", "IT", "IN", "OF", "TO", "ON",
    "AT", "BY", "UP", "AS", "IF", "OR", "AN", "A",
    # Common trading/post words
    "BUY", "SELL", "STOP", "HOLD", "WATCH", "LONG", "SHORT", "LOOK",
    "LIKE", "PLAY", "FLAT", "WAIT", "SIDE", "BODY", "HIGH", "LOW",
    "OPEN", "CLOSE", "BREAK", "ABOVE", "BELOW", "LEVEL", "PRICE",
    "ENTRY", "EXIT", "RISK", "RISKS", "LOSS", "GAIN", "MONEY", "LOSE",
    "HUGE", "SMALL", "LARGE", "GOOD", "GREAT", "BEST", "BULL", "BEAR",
    "PUSH", "PULL", "PUMP", "DUMP", "MOVE", "MOVERS", "PRINT", "PRINTS",
    "AFTER", "BEFORE", "HOURS", "HOUR", "DAYS", "DAY", "WEEK", "MONTH",
    "YEAR", "TIME", "DATE", "LIST", "LISTS", "IDEA", "IDEAS", "PLAN",
    "PLANS", "NOTE", "NOTES", "ALERT", "ALERTS", "CALL", "CALLS", "PUT",
    "PUTS", "STOCK", "STOCKS", "MARKET", "CHART", "CHARTS", "TRADE",
    "TRADES", "TRADING", "SETUP", "SETUPS", "COULD", "WOULD", "SHOULD",
    "MIGHT", "MUST", "MAY", "CAN", "CANNOT", "NEVER", "ALWAYS", "OFTEN",
    "STILL", "AGAIN", "MAYBE", "THERE", "WHERE", "WHEN", "WHILE",
    "THANK", "THANKS", "PLEASE", "SORRY", "HELLO", "SURE", "OKAY",
    "YES", "NO", "NOT", "NONE", "NULL",
    # Disclaimer / NFA boilerplate
    "NFA", "DYOR", "DD", "IMO", "IMHO", "FWIW", "TBH", "IIRC",
    "RN", "AH", "PM", "EOD", "EOW", "YTD", "ATH", "ATM", "OTC",
    "HALT", "HALTED", "RESUME",
    # Technical / acronym false positives
    "EMA", "SMA", "RSI", "ETF", "IPO", "SEC", "OTC", "ADR",
    "WTF", "LOL", "OMG", "FYI", "BTW", "TBD", "TBA",
    # Common single-post context words
    "PLAYS", "THEME", "BELOW", "IS", "IF",
    "OTHER", "OTHERS", "NIGHT", "CHINA", "ILL", "FEE", "FEES", "TAX", "RATE", "RATES",
    "TOO", "ROOM", "ROOM", "LESS", "MORE", "MUCH", "VERY",
    "LOSE", "LOSE", "LOSS", "LOST", "MAKE", "MADE",
    "TOP", "BOT", "BOTTOM", "FLOOR", "CEILING", "ZONE", "AREA",
    "RED", "GREEN", "WHITE", "BLACK", "BLUE", "GOLD",
    "BIG", "OUT", "OFF", "OWN", "ODD", "OLD", "ANY", "ALL",
    "MID", "LOW", "HIT", "RUN", "WIN", "POP", "DIP", "RIP",
    "UP", "DOWN", "LEFT", "RIGHT", "END", "START", "BEGIN",
    "CASH", "COST", "LOSS", "SELL", "BUY", "CAP", "FLOW",
    "GAP", "FILL", "FADE", "TRAP", "SQUEEZE", "TREND",
    "SWING", "SCALP", "FLIP", "HOLD", "ADDING", "ADD",
    "POSITION", "POS", "AVG", "AVERAGE", "SIZE",
    "WATCH", "WATCHLIST", "DAILY", "WEEKLY", "MONTHLY",
    "CLOSE", "OPEN", "PRINT", "REPORT", "EARN", "EARNINGS",
    "NEWS", "EVENT", "CATALYST", "SECTOR", "INDUSTRY",
    "SMALL", "MID", "LARGE", "CAP", "MICRO",
    "ABOVE", "UNDER", "OVER", "PAST", "NEXT", "NEAR",
    "STRONG", "WEAK", "NICE", "CLEAN", "TIGHT", "WIDE",
    "DEEP", "SHALLOW", "FAST", "SLOW", "HARD", "SOFT",
    "HOT", "COLD", "FRESH", "STALE", "NEW", "OLD",
    "REAL", "FAKE", "TRUE", "FALSE", "FREE", "PAID",
    "ONCE", "TWICE", "THIRD", "HALF", "FULL",
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "TEN",
    "FIRST", "SECOND", "THIRD", "LAST", "NEXT",
    "SAME", "DIFF", "DIFFERENT", "SIMILAR",
    "HERE", "THERE", "NOW", "THEN", "SOON", "LATE", "EARLY",
    "DAILY", "INTRA", "INTER", "PRE", "POST", "MID",
    "LONG", "SHORT", "BOTH", "EITHER", "NEITHER",
    "ENTIRE", "TOTAL", "PARTIAL", "FULL", "HALF",
    "MAJOR", "MINOR", "KEY", "MAIN", "CORE", "BASE",
    "ALPHA", "BETA", "GAMMA", "DELTA", "SIGMA",
    "BULL", "BEAR", "FLAT", "CHOP", "RANGE",
    "BREAKOUT", "BREAKDOWN", "REVERSAL", "BOUNCE", "RALLY",
    "CANDLE", "BAR", "LINE", "CHART", "GRAPH",
    "VOLUME", "VOL", "VOLATILITY", "MOMENTUM", "MOM",
    "RELATIVE", "REL", "STRENGTH", "STR", "WEAK",
    "SUPPORT", "RESISTANCE", "SUPPLY", "DEMAND",
    "WATCH", "ALERT", "FLAG", "SIGNAL", "SETUP",
    "IDEAL", "PERFECT", "GREAT", "GOOD", "BAD",
    "SAFE", "RISKY", "VOLATILE", "STABLE",
    "POTENTIAL", "POSSIBLE", "LIKELY", "UNLIKELY",
    "CONFIRMED", "PENDING", "ACTIVE", "CLOSED",
    "ENTRY", "EXIT", "TARGET", "STOP", "LIMIT",
    "MARKET", "LIMIT", "ORDER", "FILL", "PARTIAL",
    "PROFIT", "LOSS", "GAIN", "RETURN", "YIELD",
    "PERCENT", "PCT", "RATIO", "MULTIPLE",
    "SECTOR", "SPACE", "GROUP", "GROUP",
    "LINK", "POST", "SHARE", "COMMENT", "REPLY",
    "UPDATE", "EDIT", "DELETE", "REPORT",
    "FOLLOW", "LIKE", "RETWEET", "TWEET", "RT",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_session(text: str) -> SessionType:
    if _PREMARKET_KW.search(text):
        return SessionType.PRE_MARKET
    if _AH_KW.search(text):
        return SessionType.AFTER_HOURS
    if _OPEN_KW.search(text):
        return SessionType.MARKET_OPEN
    return SessionType.UNKNOWN


def _extract_tickers(text: str) -> List[str]:
    """Return deduplicated list of candidate tickers, preserving order."""
    found: list[str] = []
    seen: set[str] = set()
    for m in _TICKER_RE.finditer(text):
        sym = (m.group(1) or m.group(2) or "").upper()
        if sym and sym not in _TICKER_BLACKLIST and sym not in seen:
            found.append(sym)
            seen.add(sym)
    return found


def _prices_near_keyword(
    text: str,
    kw_pattern: re.Pattern,
    window: int = 80,
) -> List[float]:
    """Return price values that appear within `window` characters of a keyword match."""
    prices: list[float] = []
    for kw_match in kw_pattern.finditer(text):
        start = max(0, kw_match.start() - window)
        end = min(len(text), kw_match.end() + window)
        snippet = text[start:end]
        for p_match in _PRICE_RE.finditer(snippet):
            val = _parse_price(p_match)
            if val is not None:
                prices.append(val)
    return prices


def _best_price(candidates: List[float]) -> Optional[float]:
    return candidates[0] if candidates else None


def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    if not ts_str:
        return None
    try:
        # ISO format from Reddit API
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Per-ticker block splitter
# ---------------------------------------------------------------------------

def _split_into_ticker_blocks(content: str, tickers: List[str]) -> dict[str, str]:
    """
    Attempt to split a multi-ticker watchlist body into per-ticker text blocks.
    Falls back to giving every ticker the full text if splitting isn't possible.
    """
    if not tickers:
        return {}

    # Build a pattern that matches any of the tickers
    pattern = re.compile(
        r"(?:^|\n)\s*\$?(" + "|".join(re.escape(t) for t in tickers) + r")\b",
        re.IGNORECASE,
    )

    splits = list(pattern.finditer(content))
    if len(splits) < 2:
        return {t: content for t in tickers}

    blocks: dict[str, str] = {}
    for i, match in enumerate(splits):
        sym = match.group(1).upper()
        block_start = match.start()
        block_end = splits[i + 1].start() if i + 1 < len(splits) else len(content)
        blocks[sym] = content[block_start:block_end]

    # Any ticker without a dedicated block gets the full text
    for t in tickers:
        if t not in blocks:
            blocks[t] = content

    return blocks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_watchlist_post(raw: RawWatchlist) -> List[WatchlistEntry]:
    """
    Convert one raw scraped post into a list of WatchlistEntry objects
    (one per detected ticker).
    """
    text = f"{raw.title}\n{raw.content}"
    tickers = _extract_tickers(text)

    if not tickers:
        return []

    blocks = _split_into_ticker_blocks(raw.content, tickers)
    session = _detect_session(text)
    post_ts = _parse_timestamp(raw.timestamp)

    entries: list[WatchlistEntry] = []
    for ticker in tickers:
        block = blocks.get(ticker, text)

        support_candidates = _prices_near_keyword(block, _SUPPORT_KW)
        resist_candidates = _prices_near_keyword(block, _RESIST_KW)
        stop_candidates = _prices_near_keyword(block, _STOP_KW)

        # Heuristic: support < current price; resistance > support
        all_prices = [v for m in _PRICE_RE.finditer(block) if (v := _parse_price(m)) is not None]
        median_price = sorted(all_prices)[len(all_prices) // 2] if all_prices else 0.0

        support = _best_price(
            [p for p in support_candidates if p < median_price] or support_candidates
        )
        resistance = _best_price(
            [p for p in resist_candidates if support is None or p > support]
            or resist_candidates
        )
        stop = _best_price(stop_candidates)

        entries.append(
            WatchlistEntry(
                post_title=raw.title,
                post_timestamp=post_ts,
                raw_text=block.strip(),
                ticker=ticker,
                session_type=session,
                support_level=support,
                resistance_level=resistance,
                stop_level=stop,
                sentiment_notes=block[:200].strip(),
            )
        )

    return entries


def parse_scraped_file(filepath: str | Path) -> List[WatchlistEntry]:
    """Load scraped_watchlists.json and return all parsed entries."""
    path = Path(filepath)
    with path.open() as fh:
        raw_posts = json.load(fh)

    all_entries: list[WatchlistEntry] = []
    for post_dict in raw_posts:
        raw = RawWatchlist(**post_dict)
        entries = parse_watchlist_post(raw)
        all_entries.extend(entries)

    print(f"[parser] {len(raw_posts)} posts → {len(all_entries)} ticker entries")
    return all_entries
