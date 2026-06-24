#!/usr/bin/env bash
# Full overnight proof pipeline — BENCH_DATA_PLAN Phases A–C (+ pytest sanity).
#
# Usage:
#   cd /home/wasnaga/punk-records-inference
#   cp bench/env.example bench/.env   # add OPENROUTER_API_KEY for isolated TEXT baseline
#   nohup ./bench/run_overnight.sh >> bench/results/overnight_latest.log 2>&1 &
#   tail -f bench/results/overnight_latest.log
#
# Env:
#   PRI_BASE_URL      default http://127.0.0.1:8000
#   SKIP_WIPE=1       skip memory wipe + vLLM restart (already clean)
#   SKIP_OPENCODE=1   skip long opencode harness
#   SKIP_OPENROUTER=1 force local TEXT only (even if OPENROUTER_API_KEY set)
#   OPENROUTER_*      see bench/env.example (loaded from bench/.env)

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/bench/load_bench_env.sh"

BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/bench/results/overnight_${RUN_TAG}"
SWEEP_JSON="${OUT_DIR}/turn_sweep_cp20_80_v5.json"
MANIFEST="${OUT_DIR}/manifest.json"

mkdir -p "$OUT_DIR"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PRI_BASE_URL="$BASE_URL"
export PRI_API="${BASE_URL%/}/v1/chat/completions"
export NLS_API="$PRI_API"
export PYTHONUNBUFFERED=1

declare -a STEP_NAMES=()
declare -a STEP_STATUS=()
FAILURES=0

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${OUT_DIR}/overnight.log"
}

openrouter_ready() {
  [[ "${SKIP_OPENROUTER:-0}" != "1" ]] && [[ -n "${OPENROUTER_API_KEY:-}" ]]
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
  log "FATAL: vLLM not healthy after ${tries} attempts"
  return 1
}

run_step() {
  local name="$1"
  shift
  log "=== STEP: ${name} ==="
  if "$@"; then
    STEP_NAMES+=("$name")
    STEP_STATUS+=("ok")
    log "=== STEP OK: ${name} ==="
    return 0
  fi
  STEP_NAMES+=("$name")
  STEP_STATUS+=("fail")
  FAILURES=$((FAILURES + 1))
  log "=== STEP FAIL: ${name} (continuing) ==="
  return 1
}

run_step_or_skip() {
  local name="$1"
  shift
  if "$@"; then
    run_step "$name" true
  else
    log "SKIP: ${name}"
    STEP_NAMES+=("$name")
    STEP_STATUS+=("skip")
  fi
}

collect_store_stats() {
  local data_dir="${NLS_MEMORY_DIR:-/data/pri}"
  if docker inspect pri-inference >/dev/null 2>&1; then
    docker cp "${ROOT}/bench/collect_store_stats.py" pri-inference:/tmp/collect_store_stats.py
    docker exec pri-inference python3 -u /tmp/collect_store_stats.py \
      --base-url "$BASE_URL" \
      --data-dir "$data_dir" \
      --out "/tmp/store_stats.json" \
      --capture-sizes-csv "/tmp/capture_sizes.csv"
    docker cp pri-inference:/tmp/store_stats.json "${OUT_DIR}/store_stats.json"
    docker cp pri-inference:/tmp/capture_sizes.csv "${OUT_DIR}/capture_sizes.csv"
  else
    python3 -u "${ROOT}/bench/collect_store_stats.py" \
      --base-url "$BASE_URL" \
      --data-dir "$data_dir" \
      --out "${OUT_DIR}/store_stats.json" \
      --capture-sizes-csv "${OUT_DIR}/capture_sizes.csv"
  fi
}

write_manifest() {
  local steps_json=""
  local i
  for ((i = 0; i < ${#STEP_NAMES[@]}; i++)); do
    [[ -n "$steps_json" ]] && steps_json+=","
    steps_json+=$(python3 -c "import json; print(json.dumps({'name': ${STEP_NAMES[i]@Q}, 'status': ${STEP_STATUS[i]@Q}}))")
  done
  python3 - <<PY
import json, os, subprocess
from pathlib import Path

out = Path(${OUT_DIR@Q})
artifacts = sorted(
    str(p.relative_to(out)).replace("\\\\", "/")
    for p in out.rglob("*") if p.is_file() and p.name != "manifest.json"
)
try:
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=${ROOT@Q}, text=True
    ).strip()
except Exception:
    git_sha = None
manifest = {
    "run_tag": ${RUN_TAG@Q},
    "base_url": ${BASE_URL@Q},
    "out_dir": str(out),
    "bench_data_plan": "Phases A-C overnight proof",
    "openrouter_configured": bool(os.environ.get("OPENROUTER_API_KEY")),
    "failures": ${FAILURES},
    "steps": [${steps_json}],
    "artifacts": artifacts,
    "git_sha": git_sha,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("manifest written", out / "manifest.json")
PY
}

trap write_manifest EXIT

log "overnight proof run ${RUN_TAG} root=${ROOT} base=${BASE_URL}"
log "output ${OUT_DIR}"
if openrouter_ready; then
  log "OpenRouter TEXT baseline: enabled (${OPENROUTER_MODEL:-default model})"
else
  log "OpenRouter TEXT baseline: skipped (set OPENROUTER_API_KEY in bench/.env)"
fi

if [[ "${SKIP_WIPE:-0}" != "1" ]]; then
  run_step "wipe_memory" "${ROOT}/bench/wipe_memory.sh" --restart || true
  run_step "wait_vllm_post_wipe" wait_vllm 120
else
  run_step "wait_vllm" wait_vllm 30
fi

if docker inspect pri-inference >/dev/null 2>&1; then
  docker cp "${ROOT}/pri/resume.py" pri-inference:/opt/pri-repo/pri/resume.py 2>/dev/null \
    && log "patched pri/resume.py into container" || log "resume.py patch skipped"
fi

# --- smoke + manifest proof (KL #648) ---
run_step "smoke_health" python3 -u "${ROOT}/bench/tier1/smoke_health.py" --base-url "$BASE_URL"

run_step "manifest_proof" python3 -u "${ROOT}/bench/opencode/manifest_proof.py" \
  --base-url "$BASE_URL" \
  --out "${OUT_DIR}/manifest_opencode_t2.json"

# --- Phase B1: Marco facts (local + optional OpenRouter TEXT) ---
MARCO_ID="${RUN_TAG}_marco"
run_step "tier1_marco_facts_local" python3 -u "${ROOT}/bench/tier1/marco_facts.py" \
  --base-url "$BASE_URL" \
  --text-backend local \
  --run-id "$MARCO_ID" \
  --out "${OUT_DIR}/tier1_marco_facts_${MARCO_ID}.json"

if openrouter_ready; then
  run_step "tier1_marco_facts_openrouter" python3 -u "${ROOT}/bench/tier1/marco_facts.py" \
    --base-url "$BASE_URL" \
    --text-backend openrouter \
    --run-id "${MARCO_ID}_or" \
    --out "${OUT_DIR}/tier1_marco_facts_openrouter.json"
fi

# --- Phase B3: turn sweep cp20-80 (v5 neutral fallback) ---
run_step "turn_sweep_v5" python3 -u "${ROOT}/bench/tier1/turn_sweep.py" \
  --base-url "$BASE_URL" \
  --checkpoints 20,40,60,80 \
  --garbled-retries 2 \
  --out "$SWEEP_JSON"

if [[ -f "$SWEEP_JSON" ]]; then
  run_step "geometry_audit" python3 -u "${ROOT}/bench/tier1/geometry_audit.py" \
    --base-url "$BASE_URL" \
    --from-sweep "$SWEEP_JSON" \
    --out "${OUT_DIR}/geometry_audit_turn_sweep_v5.json"

  run_step "sweep_diagnose" python3 -u "${ROOT}/bench/tier1/sweep_diagnose.py" \
    "$SWEEP_JSON" \
    --base-url "$BASE_URL" \
    --out "${OUT_DIR}/turn_sweep_v5_diagnose.json"
else
  log "skip geometry/diagnose — sweep JSON missing"
  FAILURES=$((FAILURES + 1))
fi

# --- Phase A: inject mode compare (local TEXT + optional OpenRouter) ---
MODE_ID="${RUN_TAG}_short"
run_step "inject_mode_compare_short_local" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
  --base-url "$BASE_URL" \
  --text-backend local \
  --run-id "$MODE_ID" \
  --noise-turns 0 \
  --out "${OUT_DIR}/inject_mode_compare_${MODE_ID}.json"

if openrouter_ready; then
  run_step "inject_mode_compare_short_openrouter" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
    --base-url "$BASE_URL" \
    --text-backend openrouter \
    --run-id "${MODE_ID}_or" \
    --noise-turns 0 \
    --out "${OUT_DIR}/inject_mode_compare_short_openrouter.json"
fi

MODE_LONG_ID="${RUN_TAG}_long12"
run_step "inject_mode_compare_long12_local" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
  --base-url "$BASE_URL" \
  --text-backend local \
  --run-id "$MODE_LONG_ID" \
  --noise-turns 12 \
  --out "${OUT_DIR}/inject_mode_compare_${MODE_LONG_ID}.json"

if openrouter_ready; then
  run_step "inject_mode_compare_long12_openrouter" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
    --base-url "$BASE_URL" \
    --text-backend openrouter \
    --run-id "${MODE_LONG_ID}_or" \
    --noise-turns 12 \
    --out "${OUT_DIR}/inject_mode_compare_long12_openrouter.json"
fi

run_step "inject_mode_compare_long12_resume4096" python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
  --base-url "$BASE_URL" \
  --text-backend local \
  --run-id "${MODE_LONG_ID}_r4096" \
  --noise-turns 12 \
  --resume-max-tokens 4096 \
  --out "${OUT_DIR}/inject_mode_compare_long12_resume4096.json"

# --- Phase B2: OpenCode long session (PRI) ---
if [[ "${SKIP_OPENCODE:-0}" != "1" ]]; then
  run_step "opencode_long_session" python3 -u "${ROOT}/bench/opencode/opencode_long_session_harness.py" \
    --base-url "$BASE_URL" \
    --seed 42 \
    --out "${OUT_DIR}/opencode_long_session_${RUN_TAG}.json"
fi

# --- Phase C: storage + profile ---
run_step "admin_stats" curl -sf "${BASE_URL%/}/admin/memory/stats" \
  | tee "${OUT_DIR}/admin_memory_stats.json"

run_step "collect_store_stats" collect_store_stats

# --- Unit tests (no GPU) ---
run_step "pytest" python3 -m pytest "${ROOT}/tests/" -q

log "overnight complete failures=${FAILURES} artifacts=${OUT_DIR}"
ln -sfn "$OUT_DIR" "${ROOT}/bench/results/overnight_latest"

exit $(( FAILURES > 0 ? 1 : 0 ))
