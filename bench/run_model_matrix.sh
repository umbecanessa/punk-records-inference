#!/usr/bin/env bash
# Model matrix — same Tier-1 battery on each swapped checkpoint (sequential, not parallel).
#
# Validates plug-and-play: startup_profile probes config.json → profile.env → vLLM boot.
# Compare outputs under bench/results/model_matrix_<tag>/ vs frozen Qwen baseline.
#
# Usage (GX10, container already running with MODEL_PATH set):
#   ./bench/run_model_matrix.sh --model-tag gemma27b --pass 1
#   ./bench/run_model_matrix.sh --model-tag gemma27b --pass 2
#
# Optional env:
#   PRI_BASE_URL=http://127.0.0.1:8000
#   MODEL_MATRIX_SEED=42
#   SKIP_WIPE=1          skip memory wipe (same model, pass 2)
#   SKIP_PARITY=1        skip parity-assumption (no microscope)

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/bench/load_bench_env.sh" 2>/dev/null || true

MODEL_TAG=""
PASS=""
BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
SEED="${MODEL_MATRIX_SEED:-42}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-tag) MODEL_TAG="$2"; shift 2 ;;
    --pass) PASS="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: $0 --model-tag NAME [--pass N] [--base-url URL]

Runs the canonical comparison battery (subset of overnight proof):
  1. smoke_health
  2. startup profile snapshot (from container /data/pri/model_profile.json)
  3. tier1 marco_facts (local TEXT vs RESUME)
  4. turn_sweep cp20-80 (TEXT / RESUME / ARM-D)
  5. geometry_audit + sweep_diagnose
  6. parity_assumption @ cp80 (optional)

Output: bench/results/model_matrix_<tag>/pass<N>/
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODEL_TAG" ]]; then
  echo "ERROR: --model-tag required (e.g. qwen35_hybrid, gemma27b, llama70b)"
  exit 1
fi
PASS="${PASS:-1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_${SEED}_p${PASS}"
OUT_DIR="${ROOT}/bench/results/model_matrix_${MODEL_TAG}/pass${PASS}"
SWEEP_JSON="${OUT_DIR}/turn_sweep_cp20_80.json"
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
  echo "[$(date -Iseconds)] $*" | tee -a "${OUT_DIR}/model_matrix.log"
}

wait_vllm() {
  local tries="${1:-90}"
  local i
  for ((i = 1; i <= tries; i++)); do
    if curl -sf "${BASE_URL%/}/health" >/dev/null 2>&1; then
      log "vLLM healthy after ${i} attempts"
      return 0
    fi
    sleep 5
  done
  log "FATAL: vLLM not healthy"
  return 1
}

run_step() {
  local name="$1"
  shift
  log "=== STEP: ${name} ==="
  if "$@"; then
    STEP_NAMES+=("$name")
    STEP_STATUS+=("ok")
    log "=== OK: ${name} ==="
    return 0
  fi
  STEP_NAMES+=("$name")
  STEP_STATUS+=("fail")
  FAILURES=$((FAILURES + 1))
  log "=== FAIL: ${name} ==="
  return 1
}

write_manifest() {
  python3 - <<PY
import json, subprocess
from pathlib import Path

out = Path(${OUT_DIR@Q})
steps = list(zip(${STEP_NAMES@Q}, ${STEP_STATUS@Q}))
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
    "model_tag": ${MODEL_TAG@Q},
    "pass": ${PASS@Q},
    "run_id": ${RUN_ID@Q},
    "base_url": ${BASE_URL@Q},
    "seed": ${SEED@Q},
    "failures": ${FAILURES},
    "steps": [{"name": n, "status": s} for n, s in steps],
    "artifacts": artifacts,
    "git_sha": git_sha,
    "compare_against": "bench/results/overnight_20260624_003614 (frozen Qwen baseline)",
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("manifest", out / "manifest.json")
PY
}

trap write_manifest EXIT

log "model_matrix tag=${MODEL_TAG} pass=${PASS} base=${BASE_URL}"
log "output ${OUT_DIR}"

run_step "wait_vllm" wait_vllm 90

# Snapshot startup profile written by container boot
if docker inspect pri-inference >/dev/null 2>&1; then
  run_step "copy_model_profile" bash -c "
    docker cp pri-inference:/data/pri/model_profile.json '${OUT_DIR}/model_profile.json' 2>/dev/null || \
    docker cp pri-inference:${NLS_MEMORY_DIR:-/data/pri}/model_profile.json '${OUT_DIR}/model_profile.json'
  " || true
  docker cp pri-inference:/data/pri/profile.env "${OUT_DIR}/profile.env" 2>/dev/null || true
fi

run_step "smoke_health" python3 -u "${ROOT}/bench/tier1/smoke_health.py" --base-url "$BASE_URL"

run_step "marco_facts" python3 -u "${ROOT}/bench/tier1/marco_facts.py" \
  --base-url "$BASE_URL" \
  --text-backend local \
  --run-id "${RUN_ID}" \
  --out "${OUT_DIR}/tier1_marco_facts_${RUN_ID}.json"

if [[ "${SKIP_WIPE:-0}" != "1" ]]; then
  run_step "wipe_memory" "${ROOT}/bench/wipe_memory.sh" --restart || true
  run_step "wait_vllm_post_wipe" wait_vllm 120
fi

run_step "turn_sweep" python3 -u "${ROOT}/bench/tier1/turn_sweep.py" \
  --base-url "$BASE_URL" \
  --checkpoints 20,40,60,80 \
  --garbled-retries 2 \
  --stop-on-noise-garble \
  --out "$SWEEP_JSON"

if [[ -f "$SWEEP_JSON" ]]; then
  run_step "geometry_audit" python3 -u "${ROOT}/bench/tier1/geometry_audit.py" \
    --base-url "$BASE_URL" \
    --from-sweep "$SWEEP_JSON" \
    --out "${OUT_DIR}/geometry_audit.json"

  run_step "sweep_diagnose" python3 -u "${ROOT}/bench/tier1/sweep_diagnose.py" \
    "$SWEEP_JSON" \
    --base-url "$BASE_URL" \
    --out "${OUT_DIR}/sweep_diagnose.json"

  if [[ "${SKIP_PARITY:-0}" != "1" ]]; then
    run_step "parity_assumption" python3 -u "${ROOT}/bench/tier1/resume_parity_assumption_test.py" \
      --from-sweep "$SWEEP_JSON" \
      --checkpoint 80 \
      --base-url "$BASE_URL" \
      --out "${OUT_DIR}/parity_assumption_cp80.json"
  fi
else
  log "skip geometry/parity — sweep missing"
  FAILURES=$((FAILURES + 1))
fi

log "model_matrix done failures=${FAILURES} out=${OUT_DIR}"
exit $((FAILURES > 0 ? 1 : 0))
