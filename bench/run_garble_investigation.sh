#!/usr/bin/env bash
# cp60-80 sweep (no abort on plant garble) + garble root-cause diagnostics at cp60 and cp80.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/bench/load_bench_env.sh"

BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
OUT_DIR="${OUT_DIR:-${ROOT}/bench/results/overnight_latest}"
CONTAINER="${PRI_CONTAINER:-pri-inference}"
RUN_TAG="garble_inv_$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/garble_investigation_${RUN_TAG}.log"
SWEEP="${OUT_DIR}/turn_sweep_cp60_80_garble_inv.json"

mkdir -p "$OUT_DIR"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PRI_BASE_URL="$BASE_URL"
export PRI_API="${BASE_URL%/}/v1/chat/completions"
export NLS_API="$PRI_API"
export PYTHONUNBUFFERED=1

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

log "garble investigation run_tag=${RUN_TAG}"

"${ROOT}/bench/wipe_memory.sh" --restart
for i in $(seq 1 120); do
  curl -sf "${BASE_URL%/}/health" >/dev/null 2>&1 && break
  sleep 10
done

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker cp "${ROOT}/pri/resume.py" "${CONTAINER}:/opt/pri-repo/pri/resume.py" 2>/dev/null || true
  docker cp "${ROOT}/pri/connector.py" "${CONTAINER}:/opt/pri-repo/pri/connector.py" 2>/dev/null || true
  docker cp "${ROOT}/pri/inject_geometry_audit.py" "${CONTAINER}:/opt/pri-repo/pri/inject_geometry_audit.py" 2>/dev/null || true
fi

log "turn_sweep cp60,80 --no-stop-on-noise-garble"
python3 -u "${ROOT}/bench/tier1/turn_sweep.py" \
  --base-url "$BASE_URL" \
  --checkpoints 60,80 \
  --garbled-retries 2 \
  --no-stop-on-noise-garble \
  --out "$SWEEP" || true

if [[ -f "$SWEEP" ]]; then
  for CP in 60 80; do
    if grep -q "\"checkpoint_noise\": ${CP}" "$SWEEP"; then
      log "garble_root_cause cp${CP}"
      python3 -u "${ROOT}/bench/tier1/garble_root_cause.py" \
        --from-sweep "$SWEEP" \
        --base-url "$BASE_URL" \
        --checkpoint "$CP" \
        --out "${OUT_DIR}/turn_sweep_cp60_80_garble_inv_garble_cause_cp${CP}.json" || true
    fi
  done

  docker cp "$SWEEP" "${CONTAINER}:/tmp/sweep_garble_inv.json" 2>/dev/null || true
  docker cp "${ROOT}/pri" "${CONTAINER}:/tmp/pri-garble" 2>/dev/null || true
  docker cp "${ROOT}/bench" "${CONTAINER}:/tmp/pri-garble/bench" 2>/dev/null || true
  docker exec -e PYTHONPATH="/opt/pri-repo:/tmp/pri-garble" -e PRI_BASE_URL="$BASE_URL" -e NLS_MEMORY_DIR=/data/pri \
    "$CONTAINER" python3 -u /tmp/pri-garble/bench/tier1/geometry_audit.py \
    --base-url "$BASE_URL" --from-sweep /tmp/sweep_garble_inv.json --out /tmp/geometry_garble_inv.json 2>/dev/null || true
  docker cp "${CONTAINER}:/tmp/geometry_garble_inv.json" "${OUT_DIR}/geometry_audit_garble_inv.json" 2>/dev/null || true

  python3 -u "${ROOT}/bench/tier1/rope_delta_microscope.py" \
    --geometry "${OUT_DIR}/geometry_audit_garble_inv.json" \
    --sweep "$SWEEP" \
    --out "${OUT_DIR}/rope_delta_microscope_garble_inv.json" 2>/dev/null || true
fi

log "garble investigation complete log=${LOG}"
