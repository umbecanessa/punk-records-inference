# Docker

Image: `ghcr.io/punkrecords/inference:<tag>` (build locally as `:dev`).

## Build

```bash
docker build -f docker/Dockerfile -t ghcr.io/punkrecords/inference:dev .
```

Base image: `vllm/vllm-openai:cu130-nightly` pinned by digest in `docker/Dockerfile`
(GX10-validated: `@sha256:a20a9fc7…`). Override with `--build-arg VLLM_BASE=...`.

Build applies `patches/apply_patches.py` for GDN prefix-caching fix on Qwen hybrid.

## Run (compose)

**Vanilla Qwen3.5** — mount the HF snapshot directory directly (no extra cache mount):

```bash
export MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/<revision>
docker compose -f docker/compose.yaml up --build
```

**Symlink-heavy checkpoints** — when weight files in `MODEL_PATH` are symlinks into
`~/.cache/huggingface`, merge the GX10 overlay so targets resolve inside the container:

```bash
export MODEL_PATH=$HOME/.cache/huggingface/hub/my-checkpoint
export HF_CACHE=$HOME/.cache/huggingface
docker compose -f docker/compose.yaml -f docker/compose.gx10.yaml up --build
```

Volumes:

| Mount | Purpose |
|-------|---------|
| `pri-data:/data/pri` | Persistent `.nls` captures + memory index |
| `${MODEL_PATH}:/model:ro` | BYOC checkpoint |
| `${HF_CACHE}:/root/.cache/huggingface:ro` | *(optional, gx10 overlay)* resolves symlinked weights |

## Run (manual)

```bash
docker run --gpus all \
  -v pri-data:/data/pri \
  -v /path/to/checkpoint:/model:ro \
  -e MODEL_PATH=/model \
  -e NLS_AGENT_SHIM=1 \
  -e NLS_CHAIN_CAPTURE_MODE=turn \
  -p 8000:8000 \
  ghcr.io/punkrecords/inference:dev
```

## Health check

```bash
curl -s http://127.0.0.1:8000/v1/models | head
curl -s http://127.0.0.1:8000/health
```

## Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *(required)* | Checkpoint directory |
| `NLS_MEMORY_DIR` | `/data/pri` | Memory store root |
| `NLS_SNAPSHOT_DIR` | `/data/pri/snapshot` | KV connector snapshot dir |
| `NLS_AGENT_SHIM` | `1` | Agent middleware on/off |
| `NLS_CHAIN_CAPTURE_MODE` | `turn` | Turn snapshots for resume |
| `MAX_MODEL_LEN` | `32768` | vLLM context window |
| `GPU_MEMORY_UTILIZATION` | `0.60` | vLLM GPU memory fraction |

## Middleware stack

Registered in `docker/start.sh`:

1. `pri.middleware.agent_shim.AgentShimMiddleware`
2. `pri.admin.NLSAdminMiddleware`

KV connector: `pri.connector.NLSSnapshotConnector` (`kv_connector_module_path=pri.connector`).
