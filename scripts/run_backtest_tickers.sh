#!/usr/bin/env bash
# =============================================================================
# run_backtest_tickers.sh
# Run the Side-by-Side Body backtest + sanity checks for specific tickers.
#
# Usage:
#   ./scripts/run_backtest_tickers.sh UGRO ANNA TURB
#   ./scripts/run_backtest_tickers.sh AAPL TSLA --no-sanity
#   ./scripts/run_backtest_tickers.sh UGRO --same-window   # leaky support (control)
#
# Options (appended after tickers):
#   --no-sanity      Skip the sanity check suite
#   --same-window    Derive support from same window (leaky — for comparison)
#   --no-autotune    Use quick single-pass only (no Bayesian optimisation)
#   --trials N       Number of auto-tune trials (default: 100)
# =============================================================================

set -euo pipefail

# ── Parse args ────────────────────────────────────────────────────────────────
TICKERS=()
SKIP_SANITY=false
SAME_WINDOW=""
NO_AUTOTUNE=false
N_TRIALS=100

for arg in "$@"; do
  case "$arg" in
    --no-sanity)    SKIP_SANITY=true ;;
    --same-window)  SAME_WINDOW="--same-window-support" ;;
    --no-autotune)  NO_AUTOTUNE=true ;;
    --trials)       shift; N_TRIALS="$1" ;;
    -*)             echo "Unknown option: $arg" >&2; exit 1 ;;
    *)              TICKERS+=("$arg") ;;
  esac
done

if [ ${#TICKERS[@]} -eq 0 ]; then
  echo "Usage: $0 TICKER [TICKER ...] [--no-sanity] [--same-window] [--no-autotune]"
  exit 1
fi

TICKER_STR="${TICKERS[*]}"
echo "════════════════════════════════════════════════════════════"
echo "  Side-by-Side Body Backtest"
echo "  Tickers : $TICKER_STR"
echo "  Sanity  : $([ $SKIP_SANITY = true ] && echo 'disabled' || echo 'enabled')"
echo "════════════════════════════════════════════════════════════"

# ── Build base command ────────────────────────────────────────────────────────
CMD="python -m side_by_side_backtest.main"
CMD="$CMD --tickers $TICKER_STR"
CMD="$CMD --skip-fetch"
CMD="$CMD --export --verbose"
[ -n "$SAME_WINDOW" ] && CMD="$CMD $SAME_WINDOW"

if [ $NO_AUTOTUNE = false ]; then
  CMD="$CMD --auto-tune --n-trials $N_TRIALS"
else
  CMD="$CMD --no-sweep"
fi

if [ $SKIP_SANITY = false ]; then
  CMD="$CMD --sanity"
fi

echo "Running: $CMD"
echo ""
eval "$CMD"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Done. Reports saved to side_by_side_backtest/reports/"
echo "════════════════════════════════════════════════════════════"
