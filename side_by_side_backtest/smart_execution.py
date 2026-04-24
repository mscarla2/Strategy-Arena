import time
import logging
import math
from datetime import datetime, timezone
from typing import Optional, Callable

logger = logging.getLogger(__name__)

import numpy as np

class VolumeLiquidityGate:
    """
    Phase 2: Volume Constraints (Liquidity Gate)
    Caps trade size based on historical median volume for the specific Time-of-Day bucket.
    """
    def __init__(self, vol_matrix: dict, participation_rate: float = 0.02):
        """
        vol_matrix: {ticker: {'HH:MM': [volumes]}}
        participation_rate: max fraction of median volume to take (default 2%)
        """
        self.vol_matrix = vol_matrix
        self.rate = participation_rate

    def get_max_trade_size(self, ticker: str, current_ts: datetime) -> int:
        """Return the maximum allowed shares based on TOD liquidity."""
        profile = self.vol_matrix.get(ticker)
        if not profile:
            return 0

        bucket = current_ts.strftime("%H:%M")
        vols = profile.get(bucket)

        if not vols:
            # Fallback: use global median for this ticker if specific bucket is empty
            all_vols = [v for b in profile.values() for v in b]
            if not all_vols:
                return 0
            median_vol = np.median(all_vols)
        else:
            median_vol = np.median(vols)

        return int(median_vol * self.rate)

    def constrain_size(self, requested_qty: int, ticker: str, current_ts: datetime) -> tuple[int, bool]:
        """
        Caps requested_qty at the TOD liquidity limit.
        Returns (final_qty, is_constrained).
        """
        max_shares = self.get_max_trade_size(ticker, current_ts)
        if max_shares <= 0:
            # If no volume data at all, default to a conservative 100 shares to avoid lockup
            # or return 0 to block. We return 0 to be safe (Liquidity Vacuum protection).
            return 0, True

        if requested_qty > max_shares:
            return max_shares, True
        return requested_qty, False


class MasterBracketController:
    """
    Phase 4: OCO Synchronization Logic
    Dynamically updates the Quantity of the OCO (Take-Profit/Stop-Loss) orders
    in real-time as the Slicing Engine reports partial fills.
    """
    def __init__(self, broker, ticker: str, entry_price: float, pt_price: float, sl_price: float):
        self.broker = broker
        self.ticker = ticker
        self.entry_price = entry_price
        self.pt_price = pt_price
        self.sl_price = sl_price
        
        self.total_filled_qty = 0
        self.active_oco_id = None

    def sync_bracket(self, new_fill_qty: int):
        """
        Called by MonitorFills() whenever a new slice is filled.
        Updates the OCO bracket size to match the cumulative filled quantity.
        """
        if new_fill_qty <= 0:
            return
            
        self.total_filled_qty += new_fill_qty
        logger.info(f"[MasterBracket] Syncing bracket for {self.ticker}. New cumulative qty: {self.total_filled_qty}")
        
        if self.active_oco_id:
            # Cancel existing bracket to replace with updated quantity
            try:
                self.broker.cancel_order(self.active_oco_id)
                logger.info(f"[MasterBracket] Cancelled stale OCO bracket {self.active_oco_id}")
            except Exception as e:
                logger.error(f"[MasterBracket] Failed to cancel OCO {self.active_oco_id}: {e}")
                
        # Deploy new expanded OCO bracket
        try:
            # Place OCO order via Broker API (assuming place_oco exists or is stubbed)
            oco_order = self.broker.place_oco(
                ticker=self.ticker,
                quantity=self.total_filled_qty,
                tp_price=round(self.pt_price, 4),
                sl_price=round(self.sl_price, 4)
            )
            self.active_oco_id = oco_order.order_id
            logger.info(f"[MasterBracket] Deployed fresh OCO bracket {self.active_oco_id} for {self.total_filled_qty} shares")
        except Exception as e:
            logger.error(f"[MasterBracket] CRITICAL: Failed to deploy OCO bracket for {self.ticker}: {e}")
            # Fallback: market sell immediately if bracket fails to prevent unprotected exposure
            # self.emergency_liquidate()


class SlicingEngine:
    """
    Phase 4: Order Slicing (Iceberg/TWAP)
    Decomposes a large Final_Size into smaller chunks to minimize market impact.
    """
    def __init__(self, broker, ticker: str, total_qty: int, limit_price: float, 
                 max_chunk_shares: int = 1000, interval_secs: int = 30):
        self.broker = broker
        self.ticker = ticker
        self.total_qty = total_qty
        self.limit_price = limit_price
        self.max_chunk = max_chunk_shares
        self.interval = interval_secs
        
        self.filled_qty = 0

    def execute_slicing(self, on_fill_callback: Callable[[int], None]):
        """
        Runs the slicing loop. Sleeps between slices to let liquidity recover.
        Calls on_fill_callback(fill_qty) immediately upon slice completion.
        """
        remaining_qty = self.total_qty
        logger.info(f"[SlicingEngine] Starting execution for {self.ticker}. Total Qty: {self.total_qty}")
        
        slice_count = 0
        consecutive_timeouts = 0
        max_timeouts = 3

        while remaining_qty > 0:
            slice_count += 1
            current_chunk = min(remaining_qty, self.max_chunk)
            logger.info(f"[SlicingEngine] Working Slice #{slice_count} for {current_chunk} shares...")
            
            try:
                # Place limit buy order for the chunk
                order = self.broker.place_order(
                    ticker=self.ticker,
                    side="buy",
                    quantity=current_chunk,
                    limit_price=round(self.limit_price, 4)
                )
                
                if order.status == "error":
                    logger.error(f"[SlicingEngine] Slice #{slice_count} rejected: {order.message}")
                    break

                # Monitor fills for this chunk (timeout after 120 seconds)
                fill_qty = self._monitor_chunk_fills(order.order_id, current_chunk, timeout=120)
                
                if fill_qty > 0:
                    self.filled_qty += fill_qty
                    remaining_qty -= fill_qty
                    consecutive_timeouts = 0 # Reset on success
                    logger.info(f"[SlicingEngine] Slice #{slice_count} filled {fill_qty} shares. Remaining: {remaining_qty}")
                    
                    # Trigger Master Bracket Controller sync
                    on_fill_callback(fill_qty)
                else:
                    consecutive_timeouts += 1
                    logger.warning(f"[SlicingEngine] Slice #{slice_count} timed out unfilled ({consecutive_timeouts}/{max_timeouts}). Cancelling.")
                    self.broker.cancel_order(order.order_id)
                    
                    if consecutive_timeouts >= max_timeouts:
                        logger.error(f"[SlicingEngine] Slicing ABORTED after {max_timeouts} consecutive timeouts.")
                        break
                    
            except Exception as e:
                logger.error(f"[SlicingEngine] Error executing slice #{slice_count}: {e}")
                break
                
            if remaining_qty > 0:
                logger.info(f"[SlicingEngine] Sleeping {self.interval}s for liquidity recovery...")
                time.sleep(self.interval)
                
        logger.info(f"[SlicingEngine] Slicing complete for {self.ticker}. Final Filled Qty: {self.filled_qty}/{self.total_qty}")
        return self.filled_qty

    def _monitor_chunk_fills(self, order_id: str, expected_qty: int, timeout: int) -> int:
        """
        Polls the broker for the specific chunk order until filled or timeout.
        Returns the number of filled shares (expected_qty if filled, 0 if failed/cancelled).
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(2)
            try:
                status = self.broker.get_order_status(order_id)
                if status.status in ("filled", "paper"):
                    return expected_qty
                elif status.status in ("cancelled", "error"):
                    return 0
            except Exception as e:
                logger.warning(f"[SlicingEngine] Error polling order {order_id}: {e}")

        return 0
