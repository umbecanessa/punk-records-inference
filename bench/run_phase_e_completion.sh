#!/usr/bin/env bash
# Complete Phase E: optional resume4096 rerun, pytest in container, build summary.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/bench/load_bench_env.sh"

BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
OUT_DIR="${OUT_DIR:-${ROOT}/bench/results/overnight_20260624_003614}"
CONTAINER="${PRI_CONTAINER:-pri-inference}"
RUN_TAG="phase_e_$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/phase_e_completion_${RUN_TAG}.log"

mkdir -p "$OUT_DIR"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PRI_BASE_URL="$BASE_URL"
export PRI_API="${BASE_URL%/}/v1/chat/completions"
export NLS_API="$PRI_API"
export PYTHONUNBUFFERED=1

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

FAILURES=0
run_step() {
  local name="$1"
  shift
  log "=== STEP: ${name} ==="
  if "$@"; then
    log "=== STEP OK: ${name} ==="
    return 0
  fi
  FAILURES=$((FAILURES + 1))
  log "=== STEP FAIL: ${name} ==="
  return 1
}

log "phase E completion run_tag=${RUN_TAG}"

run_step "wipe_memory_restart" "${ROOT}/bench/wipe_memory.sh" --restart
for i in $(seq 1 120); do
  curl -sf "${BASE_URL%/}/health" >/dev/null 2>&1 && break
  sleep 10
done

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker cp "${ROOT}/pri/resume.py" "${CONTAINER}:/opt/pri-repo/pri/resume.py" 2>/dev/null || true
  docker cp "${ROOT}/pri/connector.py" "${CONTAINER}:/opt/pri-repo/pri/connector.py" 2>/dev/null || true
fi

run_step "marco_openrouter_reasoning_none" python3 -u "${ROOT}/bench/tier1/marco_facts.py" \
  --base-url "$BASE_URL" \
  --text-backend openrouter \
  --run-id "${RUN_TAG}_or_marco" \
  --out "${OUT_DIR}/tier1_marco_facts_openrouter_reasoning_none.json"

run_step "inject_long12_resume4096" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
  --base-url "$BASE_URL" \
  --text-backend local \
  --run-id "${RUN_TAG}_long4096" \
  --noise-turns 12 \
  --garbled-retries 2 \
  --resume-max-tokens 4096 \
  --out "${OUT_DIR}/inject_mode_compare_long12_resume4096_phase_e.json"

run_step "pytest_container" docker exec "$CONTAINER" \
  python3 -m pip install -q pytest 2>/dev/null; \
  docker exec "$CONTAINER" python3 -m pytest \
    /opt/pri-repo/tests/test_rope_phantom.py \
    /opt/pri-repo/tests/test_capture_smoke.py \
    /opt/pri-repo/tests/test_sweep_lib.py -q \
  2>&1 | tee "${OUT_DIR}/pytest_container_phase_e.log"

run_step "collect_store_stats" bash -c "
  docker cp '${ROOT}/bench/collect_store_stats.py' '${CONTAINER}:/tmp/collect_store_stats.py'
  docker exec '${CONTAINER}' python3 -u /tmp/collect_store_stats.py \
    --base-url '${BASE_URL}' --data-dir /data/pri \
    --out /tmp/store_stats_phase_e.json --capture-sizes-csv /tmp/capture_sizes_phase_e.csv
  docker cp '${CONTAINER}:/tmp/store_stats_phase_e.json' '${OUT_DIR}/store_stats_phase_e.json'
  docker cp '${CONTAINER}:/tmp/capture_sizes_phase_e.csv' '${OUT_DIR}/capture_sizes_phase_e.csv'
"

# Prefer fresh store stats for Phase E if collect succeeded
if [[ -f "${OUT_DIR}/store_stats_phase_e.json" ]]; then
  cp "${OUT_DIR}/store_stats_phase_e.json" "${OUT_DIR}/store_stats.json"
fi

run_step "build_phase_e_summary" python3 -u "${ROOT}/bench/build_phase_e_summary.py" \
  --run-dir "$OUT_DIR"

run_step "build_research_reports" python3 -u "${ROOT}/bench/build_research_reports.py" \
  --run-dir "$OUT_DIR"

log "phase E completion failures=${FAILURES} log=${LOG}"
exit $(( FAILURES > 0 ? 1 : 0 ))
