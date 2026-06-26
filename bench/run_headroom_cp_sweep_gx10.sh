#!/usr/bin/env bash
# Headroom + PRI turn sweep on GX10 (Qwen vLLM on :8000).
#
# Usage:
#   ./bench/run_headroom_cp_sweep_gx10.sh
#   CHECKPOINTS=60,80,100 NOISE_MODE=mixed ./bench/run_headroom_cp_sweep_gx10.sh

set -euo pipefail

PRI="${PRI:-/home/wasnaga/punk-records-inference}"
HR_PY="${HR_PY:-/home/wasnaga/headroom-venv/bin/python}"
CHECKPOINTS="${CHECKPOINTS:-60,80,100}"
NOISE_MODE="${NOISE_MODE:-mixed}"
OUT="${OUT:-${PRI}/bench/results/turn_sweep_headroom_${NOISE_MODE}_$(date +%Y%m%d_%H%M%S).json}"

cd "${PRI}"

curl -sf http://127.0.0.1:8000/v1/models >/dev/null || {
  echo "FATAL: vLLM not healthy on :8000 — run: ./bench/deploy_swap_gx10.sh qwen"
  exit 1
}

if [[ ! -x "${HR_PY}" ]]; then
  echo "FATAL: Headroom venv missing at ${HR_PY}"
  exit 1
fi

export PYTHONPATH="${PRI}:${PRI}/bench/opencode:${PRI}/bench/tier1:${PYTHONPATH:-}"
export SWEEP_RESUME_MAX_TOKENS="${SWEEP_RESUME_MAX_TOKENS:-60000}"

echo "[headroom+sweep] checkpoints=${CHECKPOINTS} noise_mode=${NOISE_MODE}"
echo "[headroom+sweep] resume_max_tokens=${SWEEP_RESUME_MAX_TOKENS}"
echo "[headroom+sweep] out=${OUT}"

PYTHONUNBUFFERED=1 "${HR_PY}" -u bench/tier1/turn_sweep_headroom.py \
  --base-url http://127.0.0.1:8000 \
  --checkpoints "${CHECKPOINTS}" \
  --noise-mode "${NOISE_MODE}" \
  --out "${OUT}" \
  --garbled-retries 4 \
  "$@"

echo "[headroom+sweep] done -> ${OUT}"
GEOM="${OUT%.json}_geometry.json"
if [[ -f "${GEOM}" ]]; then
  echo "[headroom+sweep] geometry -> ${GEOM}"
fi
