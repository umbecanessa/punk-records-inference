#!/usr/bin/env bash
# Rerun failed overnight proof steps; keep green artifacts in OUT_DIR untouched.
#
# Preserved (not rerun):
#   tier1_marco_facts_*_marco.json (local 5/5)
#   inject_mode_compare_*_short.json (local 5/5)
#   opencode_long_session_*.json (6/6)
#
# Usage (GX10):
#   OUT_DIR=bench/results/overnight_20260624_003614 nohup ./bench/run_rerun_failed.sh \
#     >> bench/results/rerun_failed.log 2>&1 &

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/bench/load_bench_env.sh"

BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
OUT_DIR="${OUT_DIR:-${ROOT}/bench/results/overnight_latest}"
CONTAINER="${PRI_CONTAINER:-pri-inference}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
SWEEP_JSON="${OUT_DIR}/turn_sweep_cp20_80_v5.json"
LOG="${OUT_DIR}/rerun_${RUN_TAG}.log"

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

openrouter_ready() {
  [[ -n "${OPENROUTER_API_KEY:-}" ]]
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

# Run a bench Python script inside pri-inference (torch/numpy available).
docker_bench() {
  local script_rel="$1"
  shift
  docker cp "${ROOT}/pri" "${CONTAINER}:/tmp/pri-rerun"
  docker cp "${ROOT}/bench" "${CONTAINER}:/tmp/pri-rerun/bench"
  docker exec \
    -e PYTHONPATH="/opt/pri-repo:/tmp/pri-rerun" \
    -e PRI_BASE_URL="$BASE_URL" \
    -e OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" \
    -e OPENROUTER_MODEL="${OPENROUTER_MODEL:-}" \
    "$CONTAINER" \
    python3 -u "/tmp/pri-rerun/bench/${script_rel}" "$@"
}

run_geometry_audit() {
  docker cp "$SWEEP_JSON" "${CONTAINER}:/tmp/sweep_rerun.json"
  docker_bench tier1/geometry_audit.py \
    --base-url "$BASE_URL" \
    --from-sweep /tmp/sweep_rerun.json \
    --out /tmp/geometry_audit.json
  docker cp "${CONTAINER}:/tmp/geometry_audit.json" "${OUT_DIR}/geometry_audit_turn_sweep_v5.json"
}

run_manifest_proof() {
  docker_bench opencode/manifest_proof.py \
    --base-url "$BASE_URL" \
    --out /tmp/manifest_opencode_t2.json
  docker cp "${CONTAINER}:/tmp/manifest_opencode_t2.json" "${OUT_DIR}/manifest_opencode_t2.json"
}

collect_store_stats() {
  local data_dir="${NLS_MEMORY_DIR:-/data/pri}"
  docker cp "${ROOT}/bench/collect_store_stats.py" "${CONTAINER}:/tmp/collect_store_stats.py"
  docker exec "$CONTAINER" python3 -u /tmp/collect_store_stats.py \
    --base-url "$BASE_URL" \
    --data-dir "$data_dir" \
    --out "/tmp/store_stats.json" \
    --capture-sizes-csv "/tmp/capture_sizes.csv"
  docker cp "${CONTAINER}:/tmp/store_stats.json" "${OUT_DIR}/store_stats.json"
  docker cp "${CONTAINER}:/tmp/capture_sizes.csv" "${OUT_DIR}/capture_sizes.csv"
}

log "rerun failed steps run_tag=${RUN_TAG} out=${OUT_DIR}"

run_step "wipe_memory_restart" "${ROOT}/bench/wipe_memory.sh" --restart
run_step "wait_vllm" wait_vllm 120

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker cp "${ROOT}/pri/resume.py" "${CONTAINER}:/opt/pri-repo/pri/resume.py" 2>/dev/null || true
  docker cp "${ROOT}/bench/tier1" "${CONTAINER}:/opt/pri-repo/bench/tier1" 2>/dev/null || true
fi

run_step "turn_sweep_v5" python3 -u "${ROOT}/bench/tier1/turn_sweep.py" \
  --base-url "$BASE_URL" \
  --checkpoints 20,40,60,80 \
  --garbled-retries 2 \
  --out "$SWEEP_JSON"

if [[ -f "$SWEEP_JSON" ]]; then
  run_step "geometry_audit" run_geometry_audit

  run_step "sweep_diagnose" python3 -u "${ROOT}/bench/tier1/sweep_diagnose.py" \
    "$SWEEP_JSON" \
    --base-url "$BASE_URL" \
    --out "${OUT_DIR}/turn_sweep_v5_diagnose.json"
fi

run_step "inject_mode_compare_long12_local" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
  --base-url "$BASE_URL" \
  --text-backend local \
  --run-id "${RUN_TAG}_long12" \
  --noise-turns 12 \
  --out "${OUT_DIR}/inject_mode_compare_20260624_003614_long12.json"

if openrouter_ready; then
  run_step "tier1_marco_facts_openrouter" python3 -u "${ROOT}/bench/tier1/marco_facts.py" \
    --base-url "$BASE_URL" \
    --text-backend openrouter \
    --run-id "${RUN_TAG}_or_marco" \
    --out "${OUT_DIR}/tier1_marco_facts_openrouter.json"

  run_step "inject_mode_compare_short_openrouter" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
    --base-url "$BASE_URL" \
    --text-backend openrouter \
    --run-id "${RUN_TAG}_or_short" \
    --noise-turns 0 \
    --out "${OUT_DIR}/inject_mode_compare_short_openrouter.json"

  run_step "inject_mode_compare_long12_openrouter" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
    --base-url "$BASE_URL" \
    --text-backend openrouter \
    --run-id "${RUN_TAG}_or_long12" \
    --noise-turns 12 \
    --out "${OUT_DIR}/inject_mode_compare_long12_openrouter.json"
else
  log "SKIP OpenRouter arms — no OPENROUTER_API_KEY"
fi

run_step "inject_mode_compare_long12_resume4096" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
  --base-url "$BASE_URL" \
  --text-backend local \
  --run-id "${RUN_TAG}_long4096" \
  --noise-turns 12 \
  --resume-max-tokens 4096 \
  --out "${OUT_DIR}/inject_mode_compare_long12_resume4096.json"

run_step "manifest_proof" run_manifest_proof

run_step "collect_store_stats" collect_store_stats

run_step "pytest" docker exec "$CONTAINER" python3 -m pytest /opt/pri-repo/tests/ -q \
  2>&1 | tee "${OUT_DIR}/pytest.log"

python3 - <<PY
import json, os, subprocess
from pathlib import Path

out = Path(${OUT_DIR@Q})
root = Path(${ROOT@Q})
manifest_path = out / "manifest.json"
manifest = {}
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["rerun_tag"] = ${RUN_TAG@Q}
manifest["rerun_failures"] = ${FAILURES}
manifest["rerun_log"] = ${LOG@Q}
manifest["artifacts"] = sorted(
    str(p.relative_to(out)).replace("\\\\", "/")
    for p in out.rglob("*") if p.is_file()
)
manifest["openrouter_configured"] = bool(os.environ.get("OPENROUTER_API_KEY"))
try:
    manifest["git_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
except Exception:
    pass
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("manifest updated", manifest_path)
PY

log "rerun complete failures=${FAILURES} log=${LOG}"
exit $(( FAILURES > 0 ? 1 : 0 ))
