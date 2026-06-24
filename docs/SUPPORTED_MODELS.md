# Supported models

Punk Records Inference ships **no model weights**. Mount your checkpoint at
`MODEL_PATH` (Docker: `/model`).

## Tier A — supported (v0.1 validation target)

| Item | Requirement |
|------|-------------|
| Architecture | Qwen3 Next hybrid (FullAttention + Mamba/DeltaNet) |
| Engine | Pinned vLLM nightly with GDN prefix-caching patch applied |
| Checkpoint | Stock **Qwen3.5-A3B-FP8** (or equivalent hybrid) on NVIDIA GPU ≥24 GB VRAM |
| Context | `--max-model-len 32768` default in `docker/start.sh` |

Required vLLM flags (set in `docker/start.sh`):

- `--enable-prefix-caching`
- `--mamba-cache-mode all`
- `--no-disable-hybrid-kv-cache-manager`
- `--kv-transfer-config` with `NLSSnapshotConnector`

## Tier B — plug-and-play matrix (experimental)

Mount checkpoint at `MODEL_PATH`, restart container. `startup_profile.py` probes
`config.json` and sets layer probes + vLLM flags (see [`MODEL_MATRIX.md`](MODEL_MATRIX.md)).

| Model class | Full Mamba resume | Matrix harness |
|-------------|-------------------|----------------|
| Qwen3.6 hybrid | Yes | `run_model_matrix.sh --model-tag qwen36` |
| **Gemma 3 27B-it** | K/V only | `--model-tag gemma27b` |
| **Llama 3.3 70B** | K/V only | `--model-tag llama70b` |
| MoE on vLLM | If vLLM loads | tag per checkpoint |

Run `./bench/run_model_matrix.sh` twice per model; compare to frozen Qwen baseline in
`bench/results/overnight_20260624_003614/`.

## Tier C — out of scope v0.1

- Dense-only transformers (no hybrid Mamba path)
- Non-vLLM engines (llama.cpp, TensorRT-LLM, etc.)
- MoE expert-slot expansion (256→320) — not shipped in this repo

## BYOC checklist

1. Checkpoint directory mounted read-only at `/model`
2. `MODEL_PATH=/model` set in environment
3. `trust-remote-code` enabled (default in start script)
4. Tool calling: `--tool-call-parser qwen3_coder` for agent harnesses

Report compatibility issues with: model id, vLLM image tag, GPU, and
`bench/results/tier1_*.json` artifact.
