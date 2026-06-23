#!/bin/bash
#
# Punk Records Inference — KV-only vLLM startup profile.
#
# Extracted from NLS start_vllm_v3.sh with MoE/router-bias/CAMM/stream-slots removed.
# Model path is required via MODEL_PATH (mount checkpoint at /model in Docker).

set -euo pipefail

export PYTHONPATH=/opt/pri-repo:${PYTHONPATH}

# Runtime deps (idempotent on container restart)
PRI_RUNTIME_DEPS="zstandard sentence-transformers"
for pkg in $PRI_RUNTIME_DEPS; do
  module="${pkg//-/_}"
  if ! python3 -c "import ${module}" >/dev/null 2>&1; then
    echo "[start.sh] installing runtime dep: ${pkg}"
    pip install --quiet --no-input "${pkg}" 2>&1 | tail -3 || {
      echo "[start.sh] WARN: ${pkg} install failed; continuing"
    }
  fi
done
unset PRI_RUNTIME_DEPS pkg module

: "${MODEL_PATH:?MODEL_PATH must point at a mounted checkpoint directory}"
export NLS_MODEL_PATH="${NLS_MODEL_PATH:-${MODEL_PATH}}"

# Persistent memory store (Docker volume default: /data/pri)
export NLS_MEMORY_DIR="${NLS_MEMORY_DIR:-/data/pri}"
export NLS_SNAPSHOT_DIR="${NLS_SNAPSHOT_DIR:-/data/pri/snapshot}"

mkdir -p "${NLS_MEMORY_DIR}" "${NLS_SNAPSHOT_DIR}/captures"

# Model probe + inject-profile env (layer probes, Swiss/neural gating)
PROFILE_ENV="${NLS_MEMORY_DIR}/profile.env"
python3 -m pri.startup_profile \
  --model-path "${MODEL_PATH}" \
  --memory-dir "${NLS_MEMORY_DIR}" \
  --inject-mode "${NLS_API_INJECT_MODE:-resume}" \
  --write-env "${PROFILE_ENV}" \
  --quiet
# shellcheck source=/dev/null
set -a
source "${PROFILE_ENV}"
set +a

export NLS_KV_K_SCALE="${NLS_KV_K_SCALE:-1.3}"
export NLS_KV_V_SCALE="${NLS_KV_V_SCALE:-1.0}"
export NLS_STRIP_ASSISTANT_KEEP_RATIO="${NLS_STRIP_ASSISTANT_KEEP_RATIO:-0}"
export NLS_STRIP_INJECT_SYS_BLOCK_LEN="${NLS_STRIP_INJECT_SYS_BLOCK_LEN:-105}"
export NLS_MAX_MEMORIES="${NLS_MAX_MEMORIES:-250000}"
export NLS_ROLE_FILTER="${NLS_ROLE_FILTER:-user,tool}"

# Agent middleware (strip transcript + capture_start for OpenCode-style agents)
export NLS_AGENT_SHIM="${NLS_AGENT_SHIM:-1}"
# Turn capture + resume chain (Fable C′)
export NLS_CHAIN_CAPTURE_MODE="${NLS_CHAIN_CAPTURE_MODE:-turn}"
export NLS_API_INJECT_MODE="${NLS_API_INJECT_MODE:-resume}"
export NLS_RESUME_ROLES="${NLS_RESUME_ROLES:-turn,tool}"

echo "[start.sh] MODEL_PATH=${MODEL_PATH}"
echo "[start.sh] NLS_MEMORY_DIR=${NLS_MEMORY_DIR}"
echo "[start.sh] NLS_SNAPSHOT_DIR=${NLS_SNAPSHOT_DIR}"
echo "[start.sh] NLS_AGENT_SHIM=${NLS_AGENT_SHIM}"
echo "[start.sh] NLS_CHAIN_CAPTURE_MODE=${NLS_CHAIN_CAPTURE_MODE}"
echo "[start.sh] PRI_INJECT_PROFILE=${PRI_INJECT_PROFILE:-}"
echo "[start.sh] NLS_NEURAL_SCORING=${NLS_NEURAL_SCORING:-0}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --enforce-eager \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.60}" \
  --enable-prefix-caching \
  --mamba-cache-mode all \
  --enable-chunked-prefill \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --middleware pri.middleware.agent_shim.AgentShimMiddleware \
  --middleware pri.admin.NLSAdminMiddleware \
  --no-disable-hybrid-kv-cache-manager \
  --kv-transfer-config "{\"kv_connector\":\"NLSSnapshotConnector\",\"kv_connector_module_path\":\"pri.connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"snapshot_dir\":\"${NLS_SNAPSHOT_DIR}\"}}"
