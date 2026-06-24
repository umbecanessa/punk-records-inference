#!/usr/bin/env bash
# Remaining BENCH_DATA_PLAN proof steps after harness + RoPE fixes.
#
# Usage (GX10):
#   OUT_DIR=bench/results/overnight_20260624_003614 nohup ./bench/run_remainder_proof.sh \
#     >> bench/results/remainder_proof.log 2>&1 &

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/bench/load_bench_env.sh"

BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
OUT_DIR="${OUT_DIR:-${ROOT}/bench/results/overnight_latest}"
CONTAINER="${PRI_CONTAINER:-pri-inference}"
RUN_TAG="remainder_$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/remainder_${RUN_TAG}.log"
SWEEP_JSON="${OUT_DIR}/turn_sweep_cp60_80_ropefix.json"

mkdir -p "$OUT_DIR"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PRI_BASE_URL="$BASE_URL"
export PRI_API="${BASE_URL%/}/v1/chat/completions"
export NLS_API="$PRI_API"
export PYTHONUNBUFFERED=1

FAILURES=0

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "$LOG"
}

wait_vllm() {
  local tries="${1:-120}"
  local i
  for ((i = 1; i <= tries; i++)); do
    if curl -sf "${BASE_URL%/}/health" >/dev/null 2>&1; then
      log "vLLM healthy after ${i} attempts"
      return 0
    fi
    sleep 10
  done
  log "FATAL: vLLM not healthy"
  return 1
}

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

docker_bench() {
  local script_rel="$1"
  shift
  docker cp "${ROOT}/pri" "${CONTAINER}:/tmp/pri-remainder"
  docker cp "${ROOT}/bench" "${CONTAINER}:/tmp/pri-remainder/bench"
  docker exec \
    -e PYTHONPATH="/opt/pri-repo:/tmp/pri-remainder" \
    -e PRI_BASE_URL="$BASE_URL" \
    -e NLS_MEMORY_DIR=/data/pri \
    "$CONTAINER" \
    python3 -u "/tmp/pri-remainder/bench/${script_rel}" "$@"
}

run_geometry_audit() {
  docker cp "$SWEEP_JSON" "${CONTAINER}:/tmp/sweep_ropefix.json"
  docker_bench tier1/geometry_audit.py \
    --base-url "$BASE_URL" \
    --from-sweep /tmp/sweep_ropefix.json \
    --out /tmp/geometry_ropefix.json
  docker cp "${CONTAINER}:/tmp/geometry_ropefix.json" "${OUT_DIR}/geometry_audit_turn_sweep_ropefix.json"
}

log "remainder proof run_tag=${RUN_TAG} out=${OUT_DIR}"

run_step "wipe_memory_restart" "${ROOT}/bench/wipe_memory.sh" --restart
run_step "wait_vllm" wait_vllm 120

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker cp "${ROOT}/pri/resume.py" "${CONTAINER}:/opt/pri-repo/pri/resume.py" 2>/dev/null || true
  docker cp "${ROOT}/pri/connector.py" "${CONTAINER}:/opt/pri-repo/pri/connector.py" 2>/dev/null || true
  docker cp "${ROOT}/pri/inject_geometry_audit.py" "${CONTAINER}:/opt/pri-repo/pri/inject_geometry_audit.py" 2>/dev/null || true
fi

run_step "turn_sweep_cp60_80" python3 -u "${ROOT}/bench/tier1/turn_sweep.py" \
  --base-url "$BASE_URL" \
  --checkpoints 60,80 \
  --garbled-retries 2 \
  --out "$SWEEP_JSON"

if [[ -f "$SWEEP_JSON" ]]; then
  run_step "geometry_audit_ropefix" run_geometry_audit
  run_step "rope_delta_microscope" python3 -u "${ROOT}/bench/tier1/rope_delta_microscope.py" \
    --geometry "${OUT_DIR}/geometry_audit_turn_sweep_ropefix.json" \
    --sweep "$SWEEP_JSON" \
    --out "${OUT_DIR}/rope_delta_microscope_ropefix.json"
  run_step "sweep_diagnose" python3 -u "${ROOT}/bench/tier1/sweep_diagnose.py" \
    "$SWEEP_JSON" \
    --base-url "$BASE_URL" \
    --out "${OUT_DIR}/turn_sweep_cp60_80_ropefix_diagnose.json"
fi

run_step "inject_long12_resume4096" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
  --base-url "$BASE_URL" \
  --text-backend local \
  --run-id "${RUN_TAG}_long4096" \
  --noise-turns 12 \
  --garbled-retries 2 \
  --resume-max-tokens 4096 \
  --out "${OUT_DIR}/inject_mode_compare_long12_resume4096_ropefix.json"

run_step "opencode_baseline" python3 -u "${ROOT}/bench/opencode/opencode_long_session_harness.py" \
  --base-url "$BASE_URL" \
  --seed 42 \
  --baseline \
  --out "${OUT_DIR}/opencode_long_session_baseline_seed42.json"

run_step "manifest_proof" docker_bench opencode/manifest_proof.py \
  --base-url "$BASE_URL" \
  --out /tmp/manifest_opencode_t2_ropefix.json
docker cp "${CONTAINER}:/tmp/manifest_opencode_t2_ropefix.json" "${OUT_DIR}/manifest_opencode_t2_ropefix.json" 2>/dev/null || true

run_step "collect_store_stats" bash -c "
  docker cp '${ROOT}/bench/collect_store_stats.py' '${CONTAINER}:/tmp/collect_store_stats.py'
  docker exec '${CONTAINER}' python3 -u /tmp/collect_store_stats.py \
    --base-url '${BASE_URL}' \
    --data-dir /data/pri \
    --out /tmp/store_stats_ropefix.json \
    --capture-sizes-csv /tmp/capture_sizes_ropefix.csv
  docker cp '${CONTAINER}:/tmp/store_stats_ropefix.json' '${OUT_DIR}/store_stats_ropefix.json'
  docker cp '${CONTAINER}:/tmp/capture_sizes_ropefix.csv' '${OUT_DIR}/capture_sizes_ropefix.csv'
"

run_step "pytest_container" docker exec "$CONTAINER" python3 -m pytest /opt/pri-repo/tests/test_rope_phantom.py /opt/pri-repo/tests/test_capture_smoke.py /opt/pri-repo/tests/test_sweep_lib.py -q \
  2>&1 | tee "${OUT_DIR}/pytest_container.log"

python3 - <<PY
import json
from pathlib import Path

out = Path(${OUT_DIR@Q})
summary = {
    "run_tag": ${RUN_TAG@Q},
    "log": ${LOG@Q},
    "failures": ${FAILURES},
    "artifacts": sorted(
        str(p.relative_to(out)).replace("\\\\", "/")
        for p in out.rglob("*")
        if p.is_file() and ("ropefix" in p.name or "remainder" in p.name or "baseline_seed42" in p.name)
    ),
}
(out / "remainder_proof_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("summary written", out / "remainder_proof_summary.json")
PY

log "remainder proof complete failures=${FAILURES} log=${LOG}"
exit $(( FAILURES > 0 ? 1 : 0 ))
