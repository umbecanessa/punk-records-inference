#!/usr/bin/env bash
# Post-RoPE/garbled/OpenRouter-fix verification — inject long12 + OpenRouter arms only.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/bench/load_bench_env.sh"

BASE_URL="${PRI_BASE_URL:-http://127.0.0.1:8000}"
OUT_DIR="${OUT_DIR:-${ROOT}/bench/results/overnight_latest}"
CONTAINER="${PRI_CONTAINER:-pri-inference}"
RUN_TAG="postfix_$(date +%Y%m%d_%H%M%S)"
LOG="${OUT_DIR}/postfix_${RUN_TAG}.log"

mkdir -p "$OUT_DIR"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PRI_BASE_URL="$BASE_URL"
export PRI_API="${BASE_URL%/}/v1/chat/completions"
export NLS_API="$PRI_API"
export PYTHONUNBUFFERED=1

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

log "postfix verify run_tag=${RUN_TAG} out=${OUT_DIR}"

"${ROOT}/bench/wipe_memory.sh" --restart
wait_vllm 120

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker cp "${ROOT}/pri/resume.py" "${CONTAINER}:/opt/pri-repo/pri/resume.py" 2>/dev/null || true
  docker cp "${ROOT}/pri/connector.py" "${CONTAINER}:/opt/pri-repo/pri/connector.py" 2>/dev/null || true
  docker cp "${ROOT}/pri/inject_geometry_audit.py" "${CONTAINER}:/opt/pri-repo/pri/inject_geometry_audit.py" 2>/dev/null || true
fi

python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
  --base-url "$BASE_URL" \
  --text-backend local \
  --run-id "${RUN_TAG}_long12" \
  --noise-turns 12 \
  --garbled-retries 2 \
  --out "${OUT_DIR}/inject_mode_compare_20260624_003614_long12_postfix.json"

if [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  python3 -u "${ROOT}/bench/tier1/marco_facts.py" \
    --base-url "$BASE_URL" \
    --text-backend openrouter \
    --run-id "${RUN_TAG}_or_marco" \
    --out "${OUT_DIR}/tier1_marco_facts_openrouter_postfix.json"

  python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
    --base-url "$BASE_URL" \
    --text-backend openrouter \
    --run-id "${RUN_TAG}_or_short" \
    --noise-turns 0 \
    --garbled-retries 2 \
    --out "${OUT_DIR}/inject_mode_compare_short_openrouter_postfix.json"

  python3 -u "${ROOT}/bench/tier1/inject_mode_compare.py" \
    --base-url "$BASE_URL" \
    --text-backend openrouter \
    --run-id "${RUN_TAG}_or_long12" \
    --noise-turns 12 \
    --garbled-retries 2 \
    --out "${OUT_DIR}/inject_mode_compare_long12_openrouter_postfix.json"
else
  log "SKIP OpenRouter — no OPENROUTER_API_KEY"
fi

log "postfix verify complete log=${LOG}"
