#!/usr/bin/env bash
# Parity assumption test on a frozen turn-sweep chain (attn vs SSM + garble A/B).
#
# Usage:
#   ./bench/run_parity_assumption_test.sh \\
#       bench/results/overnight_.../turn_sweep_cp60_80_garble_inv.json 80
#
# Env: PRI_BASE_URL, PRI_CONTAINER (default pri-inference), OUT_DIR

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/bench/load_bench_env.sh"

SWEEP_JSON="${1:-${ROOT}/bench/results/overnight_20260624_003614/turn_sweep_cp60_80_garble_inv.json}"
CHECKPOINT="${2:-80}"
BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
CONTAINER="${PRI_CONTAINER:-pri-inference}"
OUT_DIR="${OUT_DIR:-$(dirname "$SWEEP_JSON")}"
TURNS_JSON="${TURNS_JSON:-}"
RUN_TAG="parity_cp${CHECKPOINT}_$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/parity_assumption_${RUN_TAG}.log"
OUT="${OUT_DIR}/$(basename "${SWEEP_JSON%.json}")_parity_assumption_cp${CHECKPOINT}.json"

mkdir -p "$OUT_DIR"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PRI_BASE_URL="$BASE_URL"
export PRI_API="${BASE_URL%/}/v1/chat/completions"
export NLS_API="$PRI_API"
export NLS_MICROSCOPE_DIR="${NLS_MICROSCOPE_DIR:-/tmp/nls_microscope}"
export PYTHONUNBUFFERED=1

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

if [[ ! -f "$SWEEP_JSON" ]]; then
  echo "FATAL: sweep JSON not found: $SWEEP_JSON" >&2
  exit 1
fi

log "parity assumption test sweep=$SWEEP_JSON checkpoint=$CHECKPOINT"

EXTRA_ARGS=()
if [[ -n "$TURNS_JSON" && -f "$TURNS_JSON" ]]; then
  EXTRA_ARGS+=(--turns-json "$TURNS_JSON")
fi

# Stage bench into container for in-container compare (torch + capture dir).
if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker cp "$SWEEP_JSON" "${CONTAINER}:/tmp/parity_sweep.json" 2>/dev/null || true
  docker cp "${ROOT}/bench" "${CONTAINER}:/tmp/pri-parity/bench" 2>/dev/null || true
  docker cp "${ROOT}/pri" "${CONTAINER}:/tmp/pri-parity/pri" 2>/dev/null || true
  if [[ -n "$TURNS_JSON" && -f "$TURNS_JSON" ]]; then
    docker cp "$TURNS_JSON" "${CONTAINER}:/tmp/parity_turns.json" 2>/dev/null || true
    EXTRA_ARGS=(--turns-json /tmp/parity_turns.json)
  fi

  docker exec \
    -e PYTHONPATH="/opt/pri-repo:/tmp/pri-parity" \
    -e PRI_BASE_URL="$BASE_URL" \
    -e NLS_API="$PRI_API" \
    -e NLS_MICROSCOPE_DIR="$NLS_MICROSCOPE_DIR" \
    -e NLS_MEMORY_DIR="${NLS_MEMORY_DIR:-/data/pri}" \
    "$CONTAINER" \
    python3 -u /tmp/pri-parity/bench/tier1/resume_parity_assumption_test.py \
    --from-sweep /tmp/parity_sweep.json \
    --checkpoint "$CHECKPOINT" \
    --base-url "$BASE_URL" \
    --out "/tmp/parity_assumption_cp${CHECKPOINT}.json" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG" || true

  docker cp "${CONTAINER}:/tmp/parity_assumption_cp${CHECKPOINT}.json" "$OUT" 2>/dev/null || true
else
  log "no container $CONTAINER — running on host (requires torch for compare)"
  python3 -u "${ROOT}/bench/tier1/resume_parity_assumption_test.py" \
    --from-sweep "$SWEEP_JSON" \
    --checkpoint "$CHECKPOINT" \
    --base-url "$BASE_URL" \
    --out "$OUT" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$LOG" || true
fi

if [[ -f "$OUT" ]]; then
  log "wrote $OUT"
  python3 - <<PY
import json, sys
p = "${OUT}"
d = json.load(open(p, encoding="utf-8"))
pq = d.get("parity", {}).get("primary_question", {}).get("minimal_vs_resume", {})
st = pq.get("stages") or {}
attn = (st.get("attn_input_hs") or {}).get("query_cosine_avg")
ssm = (st.get("ssm_state") or {}).get("query_cosine_avg")
print(f"minimal_vs_resume: attn={attn} ssm={ssm} gap={pq.get('attn_vs_ssm_gap')}")
for v in d.get("assumption_verdicts") or []:
    print(f"  [{v['verdict']}] {v['id']}: {v['evidence']}")
PY
else
  log "WARN: output missing — see $LOG"
  exit 1
fi

log "done log=$LOG"
