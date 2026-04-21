"""
Schwab Broker Client
====================
Wraps the Schwab Trader API (api.schwabapi.com/trader/v1/) with OAuth 2.0.

Paper mode (CONFIG.paper_mode = True):
  All methods return mock responses and log intended actions.
  No network calls are made. Safe to run without credentials.

Live mode (CONFIG.paper_mode = False):
  Requires SCHWAB_CLIENT_ID and SCHWAB_CLIENT_SECRET in .env
  First run requires browser-based OAuth consent flow (run schwab_broker.py
  directly to complete the initial auth and save the refresh token).

Schwab Developer Portal: https://developer.schwab.com
OAuth callback URL to register: https://127.0.0.1

Setup
-----
1. Register at developer.schwab.com → create app → note client_id + client_secret
2. Add to .env:
       SCHWAB_CLIENT_ID=your_client_id
       SCHWAB_CLIENT_SECRET=your_client_secret
       SCHWAB_ACCOUNT_HASH=your_account_hash  # from GET /accounts after first auth
3. Run first auth:
       python -m side_by_side_backtest.schwab_broker --auth
4. Flip CONFIG.paper_mode = False in autonomous_config.py
"""
from __future__ import annotations

import json
import logging
import os
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode, urlparse, parse_qs

logger = logging.getLogger(__name__)

# Token cache file — stored alongside the DB
_TOKEN_PATH = Path(__file__).parent / ".schwab_tokens.json"

# Schwab OAuth + API endpoints
_AUTH_URL    = "https://api.schwabapi.com/v1/oauth/authorize"
_TOKEN_URL   = "https://api.schwabapi.com/v1/oauth/token"
_API_BASE    = "https://api.schwabapi.com/trader/v1"
_REDIRECT    = "https://127.0.0.1"


@dataclass
class OrderResult:
    """Result of a place_order call."""
    order_id:    Optional[str]
    status:      str   # 'filled' | 'pending' | 'cancelled' | 'paper' | 'error'
    fill_price:  float = 0.0
    message:     str   = ""


class SchwabBroker:
    """
    Thin wrapper around the Schwab Trader REST API.
    In paper mode, all methods are no-ops that log their intended action.
    """

    def __init__(self, config=None) -> None:
        if config is None:
            from .autonomous_config import CONFIG
            config = CONFIG
        self._cfg          = config
        self._access_token: Optional[str] = None
        self._token_expiry: float         = 0.0
        self._account_hash: Optional[str] = None

        if not self._cfg.paper_mode:
            self._load_env()

    # ------------------------------------------------------------------
    # Public trading API
    # ------------------------------------------------------------------

    def place_order(self, ticker: str, side: str, quantity: int,
                    limit_price: Optional[float] = None) -> OrderResult:
        """
        Place a buy or sell order.

        side        : 'buy' | 'sell'
        limit_price : float for limit order, None for market order
        """
        if self._cfg.paper_mode:
            msg = (f"[PAPER] {side.upper()} {quantity}×{ticker} "
                   f"@ {'market' if limit_price is None else f'${limit_price:.4f}'}")
            logger.info(msg)
            print(msg)
            return OrderResult(order_id=None, status="paper",
                               fill_price=limit_price or 0.0, message=msg)

        # Live path
        try:
            token = self._get_access_token()
            payload = self._build_order_payload(ticker, side, quantity, limit_price)
            resp = self._post(
                f"/accounts/{self._account_hash}/orders",
                json=payload,
                token=token,
            )
            order_id = resp.headers.get("Location", "").split("/")[-1]
            logger.info(f"[schwab] Order placed: {side} {quantity}×{ticker} → id={order_id}")
            return OrderResult(order_id=order_id, status="pending",
                               fill_price=limit_price or 0.0)
        except Exception as exc:
            logger.error(f"[schwab] place_order failed: {exc}")
            return OrderResult(order_id=None, status="error", message=str(exc))

    def get_order_status(self, order_id: str) -> OrderResult:
        """Poll order status. Returns 'filled' | 'pending' | 'cancelled'."""
        if self._cfg.paper_mode:
            return OrderResult(order_id=order_id, status="filled")

        try:
            token = self._get_access_token()
            data  = self._get(
                f"/accounts/{self._account_hash}/orders/{order_id}",
                token=token,
            )
            status     = data.get("status", "UNKNOWN").lower()
            fill_price = data.get("price", 0.0)
            return OrderResult(order_id=order_id, status=status, fill_price=fill_price)
        except Exception as exc:
            logger.error(f"[schwab] get_order_status failed: {exc}")
            return OrderResult(order_id=order_id, status="error", message=str(exc))

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""
        if self._cfg.paper_mode:
            logger.info(f"[PAPER] cancel order {order_id}")
            return True

        try:
            token = self._get_access_token()
            self._delete(
                f"/accounts/{self._account_hash}/orders/{order_id}",
                token=token,
            )
            return True
        except Exception as exc:
            logger.error(f"[schwab] cancel_order failed: {exc}")
            return False

    def get_positions(self) -> list:
        """Return current open positions as a list of dicts."""
        if self._cfg.paper_mode:
            return []

        try:
            token = self._get_access_token()
            data  = self._get(
                f"/accounts/{self._account_hash}?fields=positions",
                token=token,
            )
            positions = data.get("securitiesAccount", {}).get("positions", [])
            return [
                {
                    "ticker":        p.get("instrument", {}).get("symbol", ""),
                    "quantity":      p.get("longQuantity", 0),
                    "avg_price":     p.get("averagePrice", 0.0),
                    "market_value":  p.get("marketValue", 0.0),
                }
                for p in positions
            ]
        except Exception as exc:
            logger.error(f"[schwab] get_positions failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # OAuth helpers
    # ------------------------------------------------------------------

    def run_first_auth(self) -> None:
        """
        Interactive browser-based first auth flow.
        Run once: python -m side_by_side_backtest.schwab_broker --auth
        Saves access + refresh tokens to .schwab_tokens.json.
        """
        client_id = os.environ.get("SCHWAB_CLIENT_ID", "")
        if not client_id:
            raise ValueError("SCHWAB_CLIENT_ID not set in .env")

        params = {
            "response_type": "code",
            "client_id":     client_id,
            "redirect_uri":  _REDIRECT,
            "scope":         "readonly,trading",
        }
        url = f"{_AUTH_URL}?{urlencode(params)}"
        print(f"\nOpening browser for Schwab OAuth consent:\n{url}\n")
        webbrowser.open(url)

        redirect_url = input("Paste the full redirect URL here (https://127.0.0.1?code=...): ").strip()
        code = parse_qs(urlparse(redirect_url).query).get("code", [""])[0]
        if not code:
            raise ValueError("No auth code found in redirect URL")

        self._exchange_code_for_tokens(code)
        print("✅ Auth complete — tokens saved to", _TOKEN_PATH)

    def _exchange_code_for_tokens(self, code: str) -> None:
        import base64, requests
        client_id     = os.environ["SCHWAB_CLIENT_ID"]
        client_secret = os.environ["SCHWAB_CLIENT_SECRET"]
        credentials   = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

        resp = requests.post(
            _TOKEN_URL,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            data={
                "grant_type":   "authorization_code",
                "code":          code,
                "redirect_uri":  _REDIRECT,
            },
            timeout=15,
        )
        resp.raise_for_status()
        tokens = resp.json()
        self._save_tokens(tokens)

    def _get_access_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        now = time.time()
        if self._access_token and now < self._token_expiry - 60:
            return self._access_token

        # Load saved tokens
        tokens = self._load_tokens()
        if not tokens.get("refresh_token"):
            raise RuntimeError("No refresh token — run --auth first")

        # Refresh
        import base64, requests
        client_id     = os.environ["SCHWAB_CLIENT_ID"]
        client_secret = os.environ["SCHWAB_CLIENT_SECRET"]
        credentials   = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

        resp = requests.post(
            _TOKEN_URL,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type":  "application/x-www-form-urlencoded",
            },
            data={
                "grant_type":    "refresh_token",
                "refresh_token":  tokens["refresh_token"],
            },
            timeout=15,
        )
        resp.raise_for_status()
        new_tokens = resp.json()
        self._save_tokens(new_tokens)
        self._access_token = new_tokens["access_token"]
        self._token_expiry = now + new_tokens.get("expires_in", 1800)
        return self._access_token

    @staticmethod
    def _build_order_payload(ticker: str, side: str, quantity: int,
                              limit_price: Optional[float]) -> dict:
        instruction = "BUY" if side == "buy" else "SELL"
        order_type  = "LIMIT" if limit_price else "MARKET"
        payload = {
            "orderType":  order_type,
            "session":    "NORMAL",
            "duration":   "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity":    quantity,
                    "instrument":  {"symbol": ticker, "assetType": "EQUITY"},
                }
            ],
        }
        if limit_price:
            payload["price"] = str(round(limit_price, 2))
        return payload

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, token: str) -> dict:
        import requests
        resp = requests.get(
            f"{_API_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, json: dict, token: str):
        import requests
        resp = requests.post(
            f"{_API_BASE}{path}",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            },
            json=json,
            timeout=10,
        )
        resp.raise_for_status()
        return resp

    def _delete(self, path: str, token: str) -> None:
        import requests
        resp = requests.delete(
            f"{_API_BASE}{path}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        resp.raise_for_status()

    # ------------------------------------------------------------------
    # Token persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _save_tokens(tokens: dict) -> None:
        _TOKEN_PATH.write_text(json.dumps(tokens, indent=2))

    @staticmethod
    def _load_tokens() -> dict:
        if _TOKEN_PATH.exists():
            return json.loads(_TOKEN_PATH.read_text())
        return {}

    @staticmethod
    def _load_env() -> None:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# CLI — run first auth
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth", action="store_true",
                        help="Run the browser-based OAuth first-auth flow")
    args = parser.parse_args()

    if args.auth:
        SchwabBroker._load_env()
        broker = SchwabBroker.__new__(SchwabBroker)
        broker._cfg = None
        broker._access_token = None
        broker._token_expiry = 0.0
        broker._account_hash = None
        broker._load_env()
        broker.run_first_auth()
    else:
        parser.print_help()
