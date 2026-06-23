#!/usr/bin/env bash
# Punk Records Inference — benchmark suite runner
#
# Usage:
#   ./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
#   ./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000
#   ./bench/run_suite.sh --tier sweep --base-url http://127.0.0.1:8000
#   ./bench/run_suite.sh --tier geometry --base-url http://127.0.0.1:8000 --sweep-json bench/results/turn_sweep_cp20_80.json

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIER=""
BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
OUT_DIR="${ROOT}/bench/results"
SEED=42
SWEEP_JSON="${ROOT}/bench/results/turn_sweep_cp20_80.json"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --sweep-json) SWEEP_JSON="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --tier 1|opencode|sweep|geometry|mode-compare [--base-url URL] [--seed N] [--noise-turns N]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$TIER" ]]; then
  echo "ERROR: --tier required (1, opencode, sweep, geometry, mode-compare)"
  exit 1
fi

mkdir -p "$OUT_DIR"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
API="${BASE_URL%/}/v1/chat/completions"
export PRI_API="$API"
export PRI_BASE_URL="$BASE_URL"
export NLS_API="$API"

echo "[run_suite] tier=$TIER base=$BASE_URL out=$OUT_DIR"

case "$TIER" in
  1)
    python3 -u "${ROOT}/bench/tier1/smoke_health.py" --base-url "$BASE_URL" \
      | tee "${OUT_DIR}/tier1_health_$(date +%Y%m%d_%H%M%S).log"
    python3 -u "${ROOT}/bench/tier1/marco_facts.py" \
      --base-url "$BASE_URL" --seed "$SEED" \
      --out "${OUT_DIR}/tier1_marco_facts_${SEED}.json"
    ;;
  opencode)
    python3 -u "${ROOT}/bench/opencode/opencode_long_session_harness.py" \
      --base-url "$BASE_URL" \
      --seed "$SEED" \
      --out "${OUT_DIR}/opencode_long_session.json" \
      2>&1 | tee "${OUT_DIR}/opencode_$(date +%Y%m%d_%H%M%S).log"
    ;;
  sweep)
    python3 -u "${ROOT}/bench/tier1/turn_sweep.py" \
      --base-url "$BASE_URL" \
      --checkpoints 20,40,60,80 \
      --garbled-retries 2 \
      --out "${OUT_DIR}/turn_sweep_cp20_80_clean.json"
    ;;
  mode-compare)
    python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
      --base-url "$BASE_URL" \
      --seed "$SEED" \
      --noise-turns "${NOISE_TURNS:-0}" \
      --out "${OUT_DIR}/inject_mode_compare_${SEED}.json"
    ;;
  geometry)
    python3 -u "${ROOT}/bench/tier1/geometry_audit.py" \
      --base-url "$BASE_URL" \
      --from-sweep "$SWEEP_JSON" \
      --out "${OUT_DIR}/geometry_audit_turn_sweep.json"
    ;;
  diagnose)
    python3 -u "${ROOT}/bench/tier1/sweep_diagnose.py" \
      "${SWEEP_JSON}" \
      --base-url "$BASE_URL" \
      --out "${OUT_DIR}/turn_sweep_cp20_80_diagnose.json"
    ;;
  *)
    echo "Unknown tier: $TIER"
    exit 1
    ;;
esac

echo "[run_suite] done"
