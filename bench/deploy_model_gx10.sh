#!/usr/bin/env bash
# Swap pri-inference to a new MODEL_PATH (sequential — one model at a time on GX10).
#
# Usage:
#   export MODEL_MOUNT=/path/to/checkpoint/snapshot
#   export PRI_DATA_VOLUME=pri-data-gemma27b   # fresh volume per model (recommended)
#   export GPU_MEMORY_UTILIZATION=0.75
#   ./bench/deploy_model_gx10.sh
#
# Or download first:
#   export HF_REPO=google/gemma-3-27b-it
#   ./bench/deploy_model_gx10.sh --download

set -euo pipefail

PRI="${PRI:-/home/wasnaga/punk-records-inference}"
IMAGE="${PRI_IMAGE:-ghcr.io/punkrecords/inference:dev}"
PRI_DATA_VOLUME="${PRI_DATA_VOLUME:-pri-data}"
# GB10 Spark: 128 GiB unified memory; Qwen ~34 GiB weights leaves large KV headroom.
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
NLS_RESUME_MAX_TOKENS="${NLS_RESUME_MAX_TOKENS:-60000}"
NLS_RESUME_SWISS_MAX_TOKENS="${NLS_RESUME_SWISS_MAX_TOKENS:-512}"
CONTAINER="${PRI_CONTAINER:-pri-inference}"
PORT="${PRI_PORT:-8000}"
DOWNLOAD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download) DOWNLOAD=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

if [[ "$DOWNLOAD" == "1" ]]; then
  : "${HF_REPO:?Set HF_REPO e.g. google/gemma-3-27b-it}"
  python3 - <<PY
from huggingface_hub import snapshot_download
import os
repo = os.environ["HF_REPO"]
print("Downloading", repo, "...")
path = snapshot_download(repo)
print("Cached at:", path)
PY
  MODEL_MOUNT="$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('${HF_REPO}', local_files_only=True))")"
  export MODEL_MOUNT
  exit 0
fi

: "${MODEL_MOUNT:?Set MODEL_MOUNT to checkpoint snapshot dir}"

# HuggingFace hub caches use snapshot→blob symlinks; mount the repo root so
# paths resolve inside the container (snapshot-only mount breaks config.json).
HF_VOLUME_MOUNT="${MODEL_MOUNT}"
MODEL_PATH_CONTAINER="/model"
DOCKER_MODEL_VOL=(-v "${HF_VOLUME_MOUNT}:/model:ro")
if [[ "${MODEL_MOUNT}" == *"/snapshots/"* ]]; then
  HF_VOLUME_MOUNT="${MODEL_MOUNT%%/snapshots/*}"
  MODEL_PATH_CONTAINER="/hf-model${MODEL_MOUNT#"${HF_VOLUME_MOUNT}"}"
  DOCKER_MODEL_VOL=(-v "${HF_VOLUME_MOUNT}:/hf-model:ro")
fi

if [[ ! -f "${MODEL_MOUNT}/config.json" && ! -L "${MODEL_MOUNT}/config.json" ]]; then
  echo "ERROR: ${MODEL_MOUNT}/config.json missing"
  exit 1
fi

sed -i 's/\r$//' "${PRI}/docker/start.sh" 2>/dev/null || true
chmod +x "${PRI}/docker/start.sh" "${PRI}/bench/run_model_matrix.sh" 2>/dev/null || true

echo "[deploy] model=${MODEL_MOUNT}"
echo "[deploy] volume_mount=${HF_VOLUME_MOUNT} -> MODEL_PATH=${MODEL_PATH_CONTAINER}"
echo "[deploy] volume=${PRI_DATA_VOLUME} gpu_mem=${GPU_MEM} max_model_len=${MAX_MODEL_LEN}"
echo "[deploy] resume_max_tokens=${NLS_RESUME_MAX_TOKENS} swiss_max=${NLS_RESUME_SWISS_MAX_TOKENS}"

docker rm -f "${CONTAINER}" 2>/dev/null || true

docker volume create "${PRI_DATA_VOLUME}" 2>/dev/null || true

DOCKER_ENV=(
  -e "GPU_MEMORY_UTILIZATION=${GPU_MEM}"
  -e "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  -e "MODEL_PATH=${MODEL_PATH_CONTAINER}"
  -e "NLS_MODEL_PATH=${MODEL_PATH_CONTAINER}"
  -e NLS_AGENT_SHIM=1
  -e NLS_CHAIN_CAPTURE_MODE=turn
  -e NLS_API_INJECT_MODE=resume_overflow
  -e "NLS_RESUME_MAX_TOKENS=${NLS_RESUME_MAX_TOKENS}"
  -e "NLS_RESUME_SWISS_MAX_TOKENS=${NLS_RESUME_SWISS_MAX_TOKENS}"
)
if [[ -n "${PRI_UPGRADE_TRANSFORMERS:-}" ]]; then
  DOCKER_ENV+=(-e "PRI_UPGRADE_TRANSFORMERS=${PRI_UPGRADE_TRANSFORMERS}")
fi
if [[ -n "${PRI_VLLM_MODEL_IMPL:-}" ]]; then
  DOCKER_ENV+=(-e "PRI_VLLM_MODEL_IMPL=${PRI_VLLM_MODEL_IMPL}")
fi
if [[ -n "${PRI_VLLM_LANGUAGE_MODEL_ONLY:-}" ]]; then
  DOCKER_ENV+=(-e "PRI_VLLM_LANGUAGE_MODEL_ONLY=${PRI_VLLM_LANGUAGE_MODEL_ONLY}")
fi

docker create --name "${CONTAINER}" \
  --gpus all \
  -p "${PORT}:8000" \
  "${DOCKER_MODEL_VOL[@]}" \
  -v "${PRI_DATA_VOLUME}:/data/pri" \
  "${DOCKER_ENV[@]}" \
  --entrypoint /opt/pri-repo/docker/start.sh \
  "${IMAGE}"

echo "[deploy] overlay pri/ from ${PRI}"
docker cp "${PRI}/pri/." "${CONTAINER}:/opt/pri-repo/pri/"
docker cp "${PRI}/docker/start.sh" "${CONTAINER}:/opt/pri-repo/docker/start.sh"
docker cp "${PRI}/bench/." "${CONTAINER}:/opt/pri-repo/bench/" 2>/dev/null || true

docker start "${CONTAINER}"

echo "[deploy] startup_profile + vLLM boot (up to 12 min)..."
for i in $(seq 1 144); do
  if ! docker ps -q --filter "name=^/${CONTAINER}$" --filter status=running | grep -q .; then
    if docker ps -aq --filter "name=^/${CONTAINER}$" --filter status=exited | grep -q .; then
      echo "CONTAINER_EXITED"
      docker logs "${CONTAINER}" 2>&1 | tail -40
      exit 1
    fi
  fi
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "HEALTH_OK"
    docker logs "${CONTAINER}" 2>&1 | grep -E 'startup_profile|PRI_ARCHITECTURE|Uvicorn running|profile ok' | tail -8 || true
    docker exec "${CONTAINER}" cat /data/pri/model_profile.json 2>/dev/null | head -30 || true
    exit 0
  fi
  sleep 5
done

echo "HEALTH_TIMEOUT"
docker logs "${CONTAINER}" 2>&1 | tail -40
exit 1
