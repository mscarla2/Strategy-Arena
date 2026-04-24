import unittest
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

from side_by_side_backtest.smart_execution import VolumeLiquidityGate, SlicingEngine, MasterBracketController
from side_by_side_backtest.schwab_broker import OrderResult

class MockBroker:
    """Mock broker that simulates fills and OCO placement for testing."""
    def __init__(self):
        self.order_counter = 0
        self.placed_orders = {}
        self.placed_ocos = []
        self.cancelled_orders = []
        
    def place_order(self, ticker: str, side: str, quantity: int, limit_price: float) -> OrderResult:
        self.order_counter += 1
        order_id = f"ord-{self.order_counter}"
        # Simulate instant fill for testing
        self.placed_orders[order_id] = {
            "ticker": ticker, "side": side, "qty": quantity, "price": limit_price, "status": "filled"
        }
        return OrderResult(order_id=order_id, status="filled", fill_price=limit_price)
        
    def get_order_status(self, order_id: str) -> OrderResult:
        order = self.placed_orders.get(order_id)
        if order:
            return OrderResult(order_id=order_id, status=order["status"], fill_price=order["price"])
        return OrderResult(order_id=order_id, status="error", message="Not found")
        
    def cancel_order(self, order_id: str) -> bool:
        self.cancelled_orders.append(order_id)
        if order_id in self.placed_orders:
            self.placed_orders[order_id]["status"] = "cancelled"
        return True
        
    def place_oco(self, ticker: str, quantity: int, tp_price: float, sl_price: float) -> OrderResult:
        self.placed_ocos.append({
            "ticker": ticker, "qty": quantity, "tp": tp_price, "sl": sl_price
        })
        return OrderResult(order_id=f"oco-{len(self.placed_ocos)}", status="filled")


class TestVolumeLiquidityGate(unittest.TestCase):
    def setUp(self):
        # Build a mock volume matrix: 10:00 has volumes [1000, 2000, 3000] -> median 2000
        # 10:05 has volumes [5000] -> median 5000
        self.vol_matrix = {
            "AAPL": {
                "10:00": [1000, 2000, 3000],
                "10:05": [5000]
            }
        }
        self.gate = VolumeLiquidityGate(self.vol_matrix, participation_rate=0.02) # 2%

    def test_get_max_trade_size_exact_match(self):
        ts = datetime(2026, 4, 24, 10, 0, 0)
        # Median of [1000, 2000, 3000] is 2000. 2% of 2000 is 40.
        max_shares = self.gate.get_max_trade_size("AAPL", ts)
        self.assertEqual(max_shares, 40)
        
        ts2 = datetime(2026, 4, 24, 10, 5, 0)
        # Median of [5000] is 5000. 2% of 5000 is 100.
        max_shares2 = self.gate.get_max_trade_size("AAPL", ts2)
        self.assertEqual(max_shares2, 100)

    def test_get_max_trade_size_fallback(self):
        ts = datetime(2026, 4, 24, 10, 10, 0) # 10:10 is missing
        # Should fallback to global median of all volumes: [1000, 2000, 3000, 5000] -> median 2500
        # 2% of 2500 is 50.
        max_shares = self.gate.get_max_trade_size("AAPL", ts)
        self.assertEqual(max_shares, 50)

    def test_constrain_size_unconstrained(self):
        ts = datetime(2026, 4, 24, 10, 0, 0) # Cap is 40
        qty, constrained = self.gate.constrain_size(30, "AAPL", ts)
        self.assertEqual(qty, 30)
        self.assertFalse(constrained)

    def test_constrain_size_constrained(self):
        ts = datetime(2026, 4, 24, 10, 0, 0) # Cap is 40
        qty, constrained = self.gate.constrain_size(100, "AAPL", ts)
        self.assertEqual(qty, 40)
        self.assertTrue(constrained)

    def test_constrain_size_no_data(self):
        ts = datetime(2026, 4, 24, 10, 0, 0)
        qty, constrained = self.gate.constrain_size(100, "TSLA", ts) # TSLA missing
        self.assertEqual(qty, 0)
        self.assertTrue(constrained)


class TestMasterBracketController(unittest.TestCase):
    def setUp(self):
        self.broker = MockBroker()
        self.controller = MasterBracketController(
            broker=self.broker,
            ticker="AAPL",
            entry_price=100.0,
            pt_price=103.0,
            sl_price=98.5
        )

    def test_sync_bracket_first_fill(self):
        # First fill: place initial OCO
        self.controller.sync_bracket(100)
        self.assertEqual(self.controller.total_filled_qty, 100)
        self.assertEqual(len(self.broker.placed_ocos), 1)
        self.assertEqual(self.broker.placed_ocos[0]["qty"], 100)
        self.assertEqual(self.controller.active_oco_id, "oco-1")

    def test_sync_bracket_multiple_fills(self):
        # First fill
        self.controller.sync_bracket(100)
        
        # Second fill: cancel oco-1, place oco-2 with 250 shares
        self.controller.sync_bracket(150)
        self.assertEqual(self.controller.total_filled_qty, 250)
        self.assertEqual(len(self.broker.placed_ocos), 2)
        self.assertEqual(self.broker.placed_ocos[1]["qty"], 250)
        self.assertEqual(self.controller.active_oco_id, "oco-2")
        
        # Verify cancellation
        self.assertIn("oco-1", self.broker.cancelled_orders)


class TestSlicingEngine(unittest.TestCase):
    def setUp(self):
        self.broker = MockBroker()
        self.engine = SlicingEngine(
            broker=self.broker,
            ticker="AAPL",
            total_qty=500,
            limit_price=100.0,
            max_chunk_shares=200,
            interval_secs=0 # no sleep in tests
        )

    def test_execute_slicing_full_fill(self):
        callback_args = []
        def mock_callback(qty):
            callback_args.append(qty)
            
        # 500 total, max_chunk 200 -> 3 slices (200, 200, 100)
        total_filled = self.engine.execute_slicing(on_fill_callback=mock_callback)
        
        self.assertEqual(total_filled, 500)
        self.assertEqual(self.engine.filled_qty, 500)
        self.assertEqual(len(self.broker.placed_orders), 3)
        self.assertEqual(callback_args, [200, 200, 100])

    def test_execute_slicing_broker_error(self):
        original_place = self.broker.place_order
        call_count = 0
        def faulty_place(ticker, side, quantity, limit_price):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return original_place(ticker, side, quantity, limit_price)
            return OrderResult(order_id=None, status="error", message="Rejected")
            
        self.broker.place_order = faulty_place
        
        callback_args = []
        total_filled = self.engine.execute_slicing(on_fill_callback=lambda q: callback_args.append(q))
        
        # 1st slice fills 200. 2nd fails. Loop breaks.
        self.assertEqual(total_filled, 200)
        self.assertEqual(callback_args, [200])

    def test_execute_slicing_timeout(self):
        # Mock _monitor_chunk_fills to return 0 (timeout)
        self.engine._monitor_chunk_fills = MagicMock(return_value=0)

        # Should abort after 3 consecutive timeouts
        total_filled = self.engine.execute_slicing(on_fill_callback=lambda x: None)

        self.assertEqual(total_filled, 0)
        self.assertEqual(self.engine.filled_qty, 0)
        # Should place exactly 3 orders (max_timeouts=3) before aborting
        self.assertEqual(len(self.broker.placed_orders), 3)
        # Should cancel all 3 orders
        self.assertEqual(len(self.broker.cancelled_orders), 3)

if __name__ == "__main__":
    unittest.main()
