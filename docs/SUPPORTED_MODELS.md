# Supported models



Punk Records Inference ships **no model weights**. Mount your checkpoint at

`MODEL_PATH` (Docker: `/model`).



**Platform context:** how models fit the wider NLS architecture → [`PLATFORM.md`](PLATFORM.md).



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



**Proof:** cp20 RESUME **5/5**, OpenCode **6/6**, ~3700 tokens saved per recall — [`BENCHMARKS.md`](BENCHMARKS.md).



## Tier B — plug-and-play matrix (experimental)



Mount checkpoint at `MODEL_PATH`, restart container. `startup_profile.py` probes

`config.json` and sets layer probes + vLLM flags (see [`MODEL_MATRIX.md`](MODEL_MATRIX.md)).



| Model | Checkpoint | Full Mamba resume | cp20 RESUME (pass 1) | Matrix tag |

|-------|------------|-------------------|----------------------|------------|

| Qwen3.6 hybrid | TBD | Yes | — | `qwen36` |

| EngGPT2-16B-A3B | `engineering-group/EngGPT2-16B-A3B` | K/V only | Queued | `enggpt2` |

| Gemma 3 27B-it | `google/gemma-3-27b-it` | K/V only | **2/5** | `gemma27b` |

| Llama 3.3 70B | `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` | K/V only | **0/5** | `llama70b` |

| Llama 3 8B | `meta-llama/Meta-Llama-3-8B-Instruct` | K/V only | **0/5** (3/5 @ cp5) | `llama8b` |



Run `./bench/run_model_matrix.sh` twice per model; compare to frozen Qwen baseline in

`bench/results/overnight_20260624_003614/`.



**Restore Qwen baseline after any swap:** `./bench/deploy_swap_gx10.sh qwen`



## Tier C — out of scope v0.1



- Dense-only transformers without vLLM hybrid support

- Non-vLLM engines (llama.cpp, TensorRT-LLM, etc.)

- MoE expert-slot expansion (256→320) — not shipped in this repo

- Llama 4 Scout / Maverick — dropped from matrix (GB10 VRAM + vLLM maturity)



## BYOC checklist



1. Checkpoint directory mounted read-only at `/model`

2. `MODEL_PATH=/model` set in environment

3. `trust-remote-code` enabled (default in start script)

4. Tool calling: `--tool-call-parser qwen3_coder` for Qwen agent harnesses (auto from profile)



Report compatibility issues with: model id, vLLM image tag, GPU, and

`bench/results/model_matrix_<tag>/` artifacts.


