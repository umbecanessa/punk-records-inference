# Model matrix — plug-and-play validation

> **Maintainer doc** — multi-model validation matrix. End users: [Benchmarks](BENCHMARKS.md) and [Overview](OVERVIEW.md).

**Goal:** Prove (or falsify) that Punk Records Inference auto-configures from any mounted
checkpoint and runs the **same comparison battery** as the frozen Qwen baseline.

**Frozen baseline:** `bench/results/overnight_20260624_003614/` (Qwen3.5 hybrid).

---

## What auto-configures today

On container start (`docker/start.sh`):

1. **`pri/startup_profile.py`** reads `MODEL_PATH/config.json`
2. Derives layer topology (full-attention vs linear/Mamba), MoE expert count, probe layers
3. Writes `/data/pri/model_profile.json` + `profile.env`
4. Sets inject profile (`NLS_API_INJECT_MODE=resume` by default)
5. Sets **vLLM runtime hints** from architecture family:

| `architecture_family` | Mamba cache | Hybrid KV mgr | Mamba delta-sum | Tool parser |
|----------------------|-------------|---------------|-----------------|-------------|
| `qwen_next_hybrid` | on | on | 1 | qwen3_coder |
| `hybrid_unknown` | on | on | 1 | qwen3_coder |
| `dense_or_unknown` | off | off | 0 | off |

MoE models (e.g. Qwen3.5 MoE) are detected via `num_experts` / `n_routed_experts` in config.

---

## What still requires vLLM to load the checkpoint

Plug-and-play means **change `MODEL_PATH`, restart container** — not zero config ever:

| Requirement | Notes |
|-------------|-------|
| vLLM supports architecture | Gemma 3, Llama 3.x must load on pinned vLLM image |
| HF license + token | Gated models (Gemma, Llama) |
| VRAM | Sequential swap — one model at a time on a single GPU |
| Separate data volume | Use `pri-data-<tag>` per model so `.nls` captures don't mix |

**Full resume stack** (RoPE + Mamba delta-sum + turn capture) = Tier A hybrid only.  
**Tier B** (Gemma 27B, Llama) = K/V inject + RoPE; compare recall curves, not SSM parity.

---

## Comparison battery (`bench/run_model_matrix.sh`)

Same arms as overnight methodology (`research/01_methodology.md`), trimmed for model swap:

| Step | Harness | Primary metrics |
|------|---------|-----------------|
| 1 | `smoke_health.py` | API up |
| 2 | copy `model_profile.json` | topology probe result |
| 3 | `marco_facts.py` | TEXT vs RESUME @ short chain |
| 4 | `turn_sweep.py` cp20–80 | TEXT / RESUME / ARM-D recall @5 |
| 5 | `geometry_audit.py` | RoPE delta_uniformity |
| 6 | `sweep_diagnose.py` | failure taxonomy |
| 7 | `resume_parity_assumption_test.py` @ cp80 | H1–H6 + minimal/full parity |

Run **twice per model** (`--pass 1`, `--pass 2`) for reproducibility.

```bash
# After container restart with new MODEL_PATH
./bench/run_model_matrix.sh --model-tag gemma27b --pass 1
./bench/run_model_matrix.sh --model-tag gemma27b --pass 2 --skip-wipe  # optional same-session pass 2
```

Output: `bench/results/model_matrix_<tag>/pass<N>/manifest.json`

---

## Model queue (sequential validation)

Run one checkpoint at a time on the validation GPU:

| Order | Model | Tag | Qwen data |
|-------|-------|-----|-----------|
| — | Qwen3.5 hybrid FP8 | `qwen35_hybrid` | **Frozen** — no new runs |
| 1 | `google/gemma-3-27b-it` | `gemma27b` | ×2 passes |
| 2 | `meta-llama/Llama-3.3-70B-Instruct` | `llama70b` | ×2 passes (HF access approved) |
| 3 | Other MoE hybrid on vLLM | `*` | if vLLM loads |

### Swap procedure

```bash
docker stop pri-inference

export MODEL_PATH=$HOME/.cache/huggingface/hub/models--google--gemma-3-27b-it/snapshots/<rev>
export HF_CACHE=$HOME/.cache/huggingface
# Use fresh volume per model:
export PRI_DATA_VOLUME=pri-data-gemma27b

docker compose -f docker/compose.yaml -f docker/compose.gx10.yaml up -d

# Verify probe
docker logs pri-inference 2>&1 | tail -20
cat /data/pri/model_profile.json   # inside container

./bench/run_model_matrix.sh --model-tag gemma27b --pass 1
```

---

## Compare results

For each model matrix run, check against frozen Qwen:

| Metric | Qwen (overnight) | New model |
|--------|------------------|-----------|
| turn_sweep RESUME cp80 | 4/5 (post RoPE fix) | ? |
| turn_sweep TEXT cp80 | 5/5 | ? |
| geometry delta_uniformity | 100% | ? |
| minimal↔resume @ cp80 | ~0.89–0.93 | ? (hybrid only) |
| garble / neutral blocks | documented | ? |

Report incompatibilities with: model id, `model_profile.json`, `manifest.json`, vLLM boot log.

---

## References

- [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) — Tier A/B/C policy
- [`BENCHMARKS.md`](BENCHMARKS.md) — harness details
- NLS [`RESUME_RESEARCH_ROADMAP.md`](../../NLS/docs/research/RESUME_RESEARCH_ROADMAP.md) — research framing
