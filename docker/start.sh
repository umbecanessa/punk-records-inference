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

if [[ "${PRI_UPGRADE_TRANSFORMERS:-0}" == "1" ]]; then
  echo "[start.sh] upgrading transformers>=5.0 for vLLM MoE backend"
  pip install --quiet --no-input "transformers>=5.0.0" 2>&1 | tail -3 || {
    echo "[start.sh] WARN: transformers upgrade failed; continuing"
  }
  echo "[start.sh] upgrading mistral-common for Llama 4 tokenizer"
  pip install --quiet --no-input "mistral-common>=1.11.4" 2>&1 | tail -3 || {
    echo "[start.sh] WARN: mistral-common upgrade failed; continuing"
  }
fi

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
  --inject-mode "${NLS_API_INJECT_MODE:-resume_overflow}" \
  --write-env "${PROFILE_ENV}" \
  --force \
  --quiet
# Preserve container-level vLLM overrides (profile.env may set defaults).
_VLLM_IMPL_OVERRIDE="${PRI_VLLM_MODEL_IMPL:-}"
_VLLM_LM_ONLY_OVERRIDE="${PRI_VLLM_LANGUAGE_MODEL_ONLY:-}"
# shellcheck source=/dev/null
set -a
source "${PROFILE_ENV}"
set +a
if [[ -n "${_VLLM_IMPL_OVERRIDE}" ]]; then
  export PRI_VLLM_MODEL_IMPL="${_VLLM_IMPL_OVERRIDE}"
fi
if [[ -n "${_VLLM_LM_ONLY_OVERRIDE}" ]]; then
  export PRI_VLLM_LANGUAGE_MODEL_ONLY="${_VLLM_LM_ONLY_OVERRIDE}"
fi

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
export NLS_API_INJECT_MODE="${NLS_API_INJECT_MODE:-resume_overflow}"
export NLS_RESUME_ROLES="${NLS_RESUME_ROLES:-turn,tool}"

echo "[start.sh] MODEL_PATH=${MODEL_PATH}"
echo "[start.sh] NLS_MEMORY_DIR=${NLS_MEMORY_DIR}"
echo "[start.sh] NLS_SNAPSHOT_DIR=${NLS_SNAPSHOT_DIR}"
echo "[start.sh] NLS_AGENT_SHIM=${NLS_AGENT_SHIM}"
echo "[start.sh] NLS_CHAIN_CAPTURE_MODE=${NLS_CHAIN_CAPTURE_MODE}"
echo "[start.sh] PRI_INJECT_PROFILE=${PRI_INJECT_PROFILE:-}"
echo "[start.sh] PRI_ARCHITECTURE_FAMILY=${PRI_ARCHITECTURE_FAMILY:-unknown}"
echo "[start.sh] NLS_NEURAL_SCORING=${NLS_NEURAL_SCORING:-0}"

VLLM_EXTRA_ARGS=()
if [[ "${PRI_VLLM_MAMBA_CACHE:-1}" == "1" ]]; then
  VLLM_EXTRA_ARGS+=(--mamba-cache-mode all)
fi
if [[ "${PRI_VLLM_HYBRID_KV:-1}" == "1" ]]; then
  VLLM_EXTRA_ARGS+=(--no-disable-hybrid-kv-cache-manager)
fi
if [[ -n "${PRI_VLLM_TOOL_PARSER}" ]]; then
  VLLM_EXTRA_ARGS+=(--enable-auto-tool-choice)
  VLLM_EXTRA_ARGS+=(--tool-call-parser "${PRI_VLLM_TOOL_PARSER}")
fi
if [[ -n "${PRI_VLLM_REASONING_PARSER:-}" ]]; then
  VLLM_EXTRA_ARGS+=(--reasoning-parser "${PRI_VLLM_REASONING_PARSER}")
fi
if [[ -n "${PRI_VLLM_MODEL_IMPL:-}" ]]; then
  VLLM_EXTRA_ARGS+=(--model-impl "${PRI_VLLM_MODEL_IMPL}")
fi
if [[ "${PRI_VLLM_LANGUAGE_MODEL_ONLY:-0}" == "1" ]]; then
  VLLM_EXTRA_ARGS+=(--language-model-only)
fi

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --enforce-eager \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.60}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --middleware pri.middleware.agent_shim.AgentShimMiddleware \
  --middleware pri.admin.NLSAdminMiddleware \
  --logits-processors pri.microscope_processor:PRIMicroscopeProcessor \
  "${VLLM_EXTRA_ARGS[@]}" \
  --kv-transfer-config "{\"kv_connector\":\"NLSSnapshotConnector\",\"kv_connector_module_path\":\"pri.connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"snapshot_dir\":\"${NLS_SNAPSHOT_DIR}\"}}"
