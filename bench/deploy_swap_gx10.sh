#!/usr/bin/env bash
# Sequential model swap on GX10 — ONE PRI container on :8000 at a time.
#
# Usage:
#   ./bench/deploy_swap_gx10.sh qwen     # restore Tier A baseline (pri-inference + pri-data)
#   ./bench/deploy_swap_gx10.sh gemma
#   ./bench/deploy_swap_gx10.sh enggpt
#
# Stops any running pri-* container, deploys the requested checkpoint, waits for /health.

set -euo pipefail

MODEL="${1:-}"
if [[ "$MODEL" != "qwen" && "$MODEL" != "gemma" && "$MODEL" != "enggpt" && "$MODEL" != "llama" && "$MODEL" != "llama8b" ]]; then
  echo "Usage: $0 qwen|gemma|enggpt|llama|llama8b"
  exit 1
fi

PRI="${PRI:-/home/wasnaga/punk-records-inference}"
chmod +x "${PRI}/bench/deploy_model_gx10.sh" "${PRI}/docker/start.sh" 2>/dev/null || true
sed -i 's/\r$//' "${PRI}/docker/start.sh" "${PRI}/bench/deploy_model_gx10.sh" 2>/dev/null || true

stop_all_pri_containers() {
  echo "[swap] stopping ALL pri-* containers (free GPU before next model)..."
  # Named containers used by matrix deploys
  docker rm -f pri-gemma pri-llama pri-llama8b pri-enggpt pri-inference 2>/dev/null || true
  # Any other pri-* leftovers
  mapfile -t _pri_left < <(docker ps -aq --filter "name=pri-" 2>/dev/null || true)
  if [[ ${#_pri_left[@]} -gt 0 ]]; then
    docker rm -f "${_pri_left[@]}" 2>/dev/null || true
  fi
  # Release :8000 if something else grabbed it
  mapfile -t _port8000 < <(docker ps -q --filter "publish=8000" 2>/dev/null || true)
  if [[ ${#_port8000[@]} -gt 0 ]]; then
    echo "[swap] stopping containers bound to :8000: ${_port8000[*]}"
    docker rm -f "${_port8000[@]}" 2>/dev/null || true
  fi
  sleep 3
  if docker ps --format '{{.Names}}' | grep -qE '^pri-'; then
    echo "[swap] WARN: pri-* still running:"
    docker ps --filter "name=pri-"
    return 1
  fi
  echo "[swap] no pri-* containers running"
}

stop_all_pri_containers

case "$MODEL" in
  qwen)
    MOUNT="$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-35B-A3B-FP8', local_files_only=True))")"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"
    export PRI_CONTAINER=pri-inference
    export PRI_DATA_VOLUME=pri-data
    ;;
  gemma)
    MOUNT="$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('google/gemma-3-27b-it', local_files_only=True))")"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
    export PRI_CONTAINER=pri-gemma
    export PRI_DATA_VOLUME=pri-data-gemma27b
    ;;
  enggpt)
    MOUNT="$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('engineering-group/EngGPT2-16B-A3B', local_files_only=True))")"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
    export PRI_UPGRADE_TRANSFORMERS=1
    export PRI_CONTAINER=pri-enggpt
    export PRI_DATA_VOLUME=pri-data-enggpt2
    ;;
  llama)
    # Dense global-attention control for Tier B matrix (GB10 ~121GB unified mem).
    # Use RedHatAI compressed-tensors FP8 (vLLM-native). NVIDIA TRT FP8 uses
    # k_scale naming that fails on our pinned vLLM (input_scale/weight_scale).
    HF_REPO="${HF_REPO:-RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic}"
    MOUNT="$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('${HF_REPO}', local_files_only=True))")"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    export PRI_CONTAINER=pri-llama
    export PRI_DATA_VOLUME=pri-data-llama70b
    ;;
  llama8b)
    HF_REPO="${HF_REPO:-meta-llama/Meta-Llama-3-8B-Instruct}"
    MOUNT="$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('${HF_REPO}', local_files_only=True))")"
    export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.22}"
    export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
    export PRI_UPGRADE_TRANSFORMERS=1
    export PRI_CONTAINER=pri-llama8b
    export PRI_DATA_VOLUME=pri-data-llama8b
    ;;
esac

export PRI_PORT=8000
export MODEL_MOUNT="${MOUNT}"

echo "[swap] model=${MODEL} mount=${MOUNT} port=${PRI_PORT} gpu_mem=${GPU_MEMORY_UTILIZATION} ctx=${MAX_MODEL_LEN}"
"${PRI}/bench/deploy_model_gx10.sh"

echo "[swap] ${MODEL} live on http://127.0.0.1:8000"
