#!/usr/bin/env bash
# Append BENCH_DATA_PLAN proof steps to an in-progress or finished overnight run.
#
# Usage (GX10):
#   OUT_DIR=bench/results/overnight_20260624_003614 nohup ./bench/run_proof_completion.sh \
#     >> bench/results/proof_completion.log 2>&1 &
#
# Or wait for run_overnight.sh then run missing phases:
#   WAIT_PID=3220683 OUT_DIR=... ./bench/run_proof_completion.sh

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/bench/load_bench_env.sh"

BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
OUT_DIR="${OUT_DIR:-${ROOT}/bench/results/overnight_latest}"
WAIT_PID="${WAIT_PID:-}"

mkdir -p "$OUT_DIR"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PRI_BASE_URL="$BASE_URL"
export PRI_API="${BASE_URL%/}/v1/chat/completions"
export NLS_API="$PRI_API"
export PYTHONUNBUFFERED=1

log() {
  echo "[$(date -Iseconds)] $*" | tee -a "${OUT_DIR}/proof_completion.log"
}

openrouter_ready() {
  [[ -n "${OPENROUTER_API_KEY:-}" ]]
}

run_or_skip_openrouter() {
  local label="$1"
  shift
  if openrouter_ready; then
    log "=== ${label} (OpenRouter TEXT) ==="
    "$@" --text-backend openrouter
  else
    log "SKIP ${label}: OPENROUTER_API_KEY not set (copy bench/.env from env.example)"
  fi
}

if [[ -n "$WAIT_PID" ]]; then
  log "waiting for pid ${WAIT_PID} (run_overnight)..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 30
  done
  log "pid ${WAIT_PID} finished"
fi

log "proof completion out_dir=${OUT_DIR} base=${BASE_URL}"

# --- Phase B: manifest proof (KL #648) ---
if [[ ! -f "${OUT_DIR}/manifest_opencode_t2.json" ]]; then
  log "=== manifest_proof ==="
  python3 -u "${ROOT}/bench/opencode/manifest_proof.py" \
    --base-url "$BASE_URL" \
    --out "${OUT_DIR}/manifest_opencode_t2.json" \
    || log "manifest_proof failed (continuing)"
else
  log "skip manifest_proof — artifact exists"
fi

# --- Phase B1: Marco facts OpenRouter baseline ---
if openrouter_ready && [[ ! -f "${OUT_DIR}/tier1_marco_facts_openrouter.json" ]]; then
  run_or_skip_openrouter "tier1_marco_facts_openrouter" \
    python3 -u "${ROOT}/bench/tier1/marco_facts.py" \
      --base-url "$BASE_URL" \
      --run-id "proof_or_marco" \
      --out "${OUT_DIR}/tier1_marco_facts_openrouter.json"
fi

# --- Phase A: inject mode compare (OpenRouter TEXT) ---
if openrouter_ready; then
  if [[ ! -f "${OUT_DIR}/inject_mode_compare_short_openrouter.json" ]]; then
    run_or_skip_openrouter "inject_mode_compare_short_or" \
      python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
        --base-url "$BASE_URL" \
        --run-id "proof_or_short" \
        --noise-turns 0 \
        --out "${OUT_DIR}/inject_mode_compare_short_openrouter.json"
  fi
  if [[ ! -f "${OUT_DIR}/inject_mode_compare_long12_openrouter.json" ]]; then
    run_or_skip_openrouter "inject_mode_compare_long12_or" \
      python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
        --base-url "$BASE_URL" \
        --run-id "proof_or_long12" \
        --noise-turns 12 \
        --out "${OUT_DIR}/inject_mode_compare_long12_openrouter.json"
  fi
fi

# --- Phase A2: long chain with resume-max-tokens 4096 (local TEXT) ---
if [[ ! -f "${OUT_DIR}/inject_mode_compare_long12_resume4096.json" ]]; then
  log "=== inject_mode_compare long12 resume-max-tokens 4096 ==="
  python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
    --base-url "$BASE_URL" \
    --text-backend local \
    --run-id "proof_long4096" \
    --noise-turns 12 \
    --resume-max-tokens 4096 \
    --out "${OUT_DIR}/inject_mode_compare_long12_resume4096.json" \
    || log "inject long4096 failed (continuing)"
fi

# --- Phase C: storage + profile artifacts ---
if [[ ! -f "${OUT_DIR}/store_stats.json" ]]; then
  log "=== collect_store_stats ==="
  DATA_DIR="${NLS_MEMORY_DIR:-/data/pri}"
  if docker inspect pri-inference >/dev/null 2>&1; then
    docker exec pri-inference python3 -u /opt/pri-repo/bench/collect_store_stats.py \
      --base-url "$BASE_URL" \
      --data-dir "$DATA_DIR" \
      --out "/tmp/store_stats.json" \
      --capture-sizes-csv "/tmp/capture_sizes.csv" \
      && docker cp pri-inference:/tmp/store_stats.json "${OUT_DIR}/store_stats.json" \
      && docker cp pri-inference:/tmp/capture_sizes.csv "${OUT_DIR}/capture_sizes.csv" \
      || log "collect_store_stats via docker failed"
  else
    python3 -u "${ROOT}/bench/collect_store_stats.py" \
      --base-url "$BASE_URL" \
      --data-dir "$DATA_DIR" \
      --out "${OUT_DIR}/store_stats.json" \
      --capture-sizes-csv "${OUT_DIR}/capture_sizes.csv" \
      || log "collect_store_stats failed"
  fi
fi

# --- Unit tests (no GPU) ---
log "=== pytest ==="
if python3 -m pytest "${ROOT}/tests/" -q 2>&1 | tee "${OUT_DIR}/pytest.log"; then
  log "pytest PASS"
else
  log "pytest FAIL (non-fatal for proof artifacts)"
fi

# --- Refresh manifest ---
python3 - <<PY
import json, os, subprocess
from pathlib import Path

out = Path(${OUT_DIR@Q})
root = Path(${ROOT@Q})
artifacts = sorted(
    str(p.relative_to(out)).replace("\\\\", "/")
    for p in out.rglob("*") if p.is_file()
)
manifest_path = out / "manifest.json"
manifest = {}
if manifest_path.is_file():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["proof_completion"] = True
manifest["openrouter_configured"] = bool(os.environ.get("OPENROUTER_API_KEY"))
manifest["artifacts"] = artifacts
try:
    manifest["git_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
except Exception:
    pass
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("manifest updated", manifest_path)
PY

log "proof completion done — see ${OUT_DIR}"
