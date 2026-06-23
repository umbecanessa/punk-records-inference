#!/usr/bin/env bash
# Punk Records Inference — benchmark suite runner
#
# Usage:
#   ./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
#   ./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIER=""
BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
OUT_DIR="${ROOT}/bench/results"
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --tier 1|opencode [--base-url URL] [--seed N]"
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$TIER" ]]; then
  echo "ERROR: --tier required (1 or opencode)"
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
  *)
    echo "Unknown tier: $TIER"
    exit 1
    ;;
esac

echo "[run_suite] done"
