# Installation

Hardware, software, and checkpoint requirements for running Punk Records Inference locally.

---

## Requirements

### Hardware

| Component | Minimum | Validated (v0.1) |
|-----------|---------|------------------|
| GPU | NVIDIA with CUDA, ≥24 GB VRAM for 35B FP8 | Qwen3.5-35B-A3B-FP8 validated |
| RAM | 32 GB host RAM | 64 GB+ recommended |
| Disk | Checkpoint size + `/data/pri` growth | NVMe preferred for `.nls` I/O |

PRI runs **one vLLM process per container**. Multi-GPU tensor parallel is not validated in v0.1.

### Software

| Tool | Version | Notes |
|------|---------|-------|
| Docker | 24+ | With [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| NVIDIA driver | Matching CUDA 13.x stack | Image uses `vllm/vllm-openai:cu130-nightly` |
| Git | Any recent | Clone the repo |
| Python 3.10+ | Host only | For bench scripts and unit tests |

### Model checkpoint (BYOC)

PRI ships **no weights**. You must provide a compatible checkpoint:

- **Tier A:** Qwen3 Next hybrid (FullAttention + Mamba/DeltaNet), e.g. **Qwen3.5-35B-A3B-FP8**
- Mounted read-only at `/model` inside the container
- `trust-remote-code` enabled (default in `docker/start.sh`)

See [Supported models](../SUPPORTED_MODELS.md) for the full tier list and vLLM flag requirements.

---

## Get the checkpoint

### Option A — Hugging Face cache (recommended)

If you already pulled the model with `huggingface-cli`:

```bash
export MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/<revision>
```

List available snapshots:

```bash
ls ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/
```

### Option B — Custom checkpoint directory

Any directory with `config.json` and weight shards works, as long as the architecture is Tier A or B:

```bash
export MODEL_PATH=/path/to/my-qwen35-checkpoint
```

### Symlink-heavy layouts

When weight files in `MODEL_PATH` are symlinks into `~/.cache/huggingface`, use the **HF cache overlay** so targets resolve inside the container:

```bash
export HF_CACHE=$HOME/.cache/huggingface
docker compose -f docker/compose.yaml -f docker/compose.gx10.yaml up --build
```

---

## Build the image

From the repo root:

```bash
git clone https://github.com/umbecanessa/punk-records-inference.git
cd punk-records-inference

docker build -f docker/Dockerfile -t ghcr.io/punkrecords/inference:dev .
```

The Dockerfile pins a vLLM nightly digest and applies `patches/apply_patches.py` for Qwen hybrid prefix-caching. Override the base with `--build-arg VLLM_BASE=...` if needed.

---

## Run with Compose

**Vanilla layout** (direct snapshot mount):

```bash
export MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/<revision>
docker compose -f docker/compose.yaml up --build
```

On first boot:

1. Runtime deps install (`zstandard`, `sentence-transformers`)
2. `pri/startup_profile.py` probes `config.json` and writes `/data/pri/profile.env`
3. vLLM starts on port **8000** with agent middleware + KV connector

### Persistent data

| Volume | Path | Contents |
|--------|------|----------|
| `pri-data` | `/data/pri` | `.nls` captures, memory index, `profile.env` |

To wipe memory and start fresh:

```bash
docker compose -f docker/compose.yaml down -v
```

---

## Verify installation

```bash
# Models endpoint
curl -s http://127.0.0.1:8000/v1/models | jq .

# Health
curl -s http://127.0.0.1:8000/health

# Admin memory stats (debug)
curl -s http://127.0.0.1:8000/admin/memory/stats | jq .
```

Expected: your checkpoint appears as the served model; admin stats show empty or prior captures.

---

## Optional configuration

Common overrides (also in [Environment variables](../reference/env-vars.md)):

```bash
# Larger context (if GPU memory allows)
export MAX_MODEL_LEN=65536

# Overflow profile instead of pure resume
export NLS_API_INJECT_MODE=resume_overflow

# Disable agent middleware (manual kv_transfer_params only)
export NLS_AGENT_SHIM=0
```

Pass via compose environment or `docker run -e`.

---

## Host-side tools

Bench and test scripts run on the **host**, not inside the container:

```bash
pip install requests          # benchmarks
pip install pytest torch zstandard   # unit tests
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `MODEL_PATH must point at...` | Env not set | Export `MODEL_PATH` before compose |
| CUDA / GPU not found | Missing NVIDIA toolkit | Install container toolkit; use `--gpus all` |
| Weight file not found | Symlink layout | Use `compose.gx10.yaml` HF cache overlay + `HF_CACHE` |
| Empty `/v1/models` | vLLM still starting | Wait for startup logs; check GPU memory |
| Resume recall fails | Wrong inject mode / capture | See [Core concepts](concepts.md), [Troubleshooting](../guides/troubleshooting.md) |

---

## Next steps

- [Quickstart](quickstart.md) — first bench run
- [Docker](../DOCKER.md) — manual `docker run`, middleware stack
- [Integrating OpenCode](../guides/integrating-opencode.md) — agent client wiring
