# Model matrix — plug-and-play validation

> **Maintainer doc** — multi-model validation matrix. End users: [Benchmarks](BENCHMARKS.md) and [Overview](OVERVIEW.md).

**Goal:** Prove (or falsify) that Punk Records Inference auto-configures from any mounted
checkpoint and runs the **same comparison battery** as the frozen Qwen baseline.

**Frozen baseline:** `bench/results/overnight_20260624_003614/` (Qwen3.5 hybrid).

---

## Executive summary (2026-06-25)

| Finding | Detail |
|---------|--------|
| **Stack is real** | Resume inject, RoPE geometry, and turn capture work on Tier A Qwen; Tier B models load plug-and-play via `startup_profile.py`. |
| **Not production-closed** | Long-chain RESUME cliff (cp60+) persists even on Qwen; Tier B recall degrades faster than TEXT. |
| **Critical bug fixed** | Resume pack was ordering `[T1 KV][system+user]` instead of `[system][T1][user]` — caused garbled Tier B decode; fix in `pri/resume.py` + `pri/connector.py`. |
| **Llama 4 Scout dropped** | Official BF16 does not fit GB10 (~116 GB at load); FP8 download abandoned to reclaim disk. Not on matrix queue. |

**Production posture:** Technology is **promising for research and early integration**, not yet a closed production guarantee across architectures. Qwen Tier A is the validated reference; Tier B results inform where K/V-only inject helps vs where model limits dominate.

---

## Results snapshot (GX10, pass 1)

Artifacts under `bench/results/model_matrix_<tag>/pass1/` on GX10 unless copied locally.

### Turn sweep — recall @5 (`pass_clean`)

| Model | Tag | cp20 TEXT | cp20 RESUME | cp40 RESUME | cp60 RESUME | cp80 RESUME | Notes |
|-------|-----|-----------|-------------|-------------|-------------|-------------|-------|
| **Qwen3.5-35B-A3B-FP8** | `qwen35_hybrid` | 5/5 | **5/5** | 5/5 | 3/5 | 0/5 | Frozen overnight run; full Mamba resume |
| **Gemma 3 27B-it** | `gemma27b` | 5/5 | **2/5** | — | — | — | K/V only; `turn_sweep_cp20_plugplay.json` |
| **Llama 3.3 70B FP8** | `llama70b` | 5/5 | **0/5** | — | — | — | RedHat compressed-tensors FP8 |
| **Llama 3 8B Instruct** | `llama8b` | 2/5 | **0/5** | — | — | — | Post ordering-fix; TEXT partly refusal |

### Llama 8B scaling curve (post inject-ordering fix)

| cp | Turns | TEXT | RESUME |
|----|-------|------|--------|
| 5 | 8 | 5/5 | 3/5 |
| 10 | 13 | 5/5 | 2/5 |
| 15 | 18 | 5/5 | 1/5 |
| 20 | 23 | 2/5 | 0/5 |

Short chains: RESUME viable. Longer chains: recall collapses before TEXT — model/inject limits, not garble.

### Qwen baseline reference (frozen)

| cp | TEXT | RESUME | ARM-D |
|----|------|--------|-------|
| 20 | 5/5 | 5/5 | 5/5 |
| 40 | 5/5 | 5/5 | 5/5 |
| 60 | 5/5 | 3/5 | 3/5 |
| 80 | 5/5 | 0/5 | 0/5 |

RoPE geometry: **100% delta_uniformity** on audited chains. cp60+ RESUME failures are inject-mediated decode, not pack geometry (documented in overnight `BENCHMARK_SUMMARY.md`).

---

## Code fixes pending commit

These changes are on disk (local + synced to GX10) and should ship together:

| Area | Files | Why |
|------|-------|-----|
| **Resume ordering** | `pri/resume.py`, `pri/connector.py` | Prepend system block to phantom pack; strip duplicate system tokens from live prompt on inject |
| **Dense topology probe** | `pri/startup_profile.py` | Llama/Gemma dense models were mis-probed as hybrid interval-4; fixes layer/expert defaults |
| **Container boot** | `docker/start.sh` | Preserve `PRI_VLLM_*` overrides after `profile.env`; only enable tool parser when set; optional transformers/mistral-common upgrade path |
| **GX10 deploy** | `bench/deploy_model_gx10.sh`, `bench/deploy_swap_gx10.sh` | `--download` exits cleanly; env passthrough; **`qwen` target** restores baseline; scout target removed |
| **Bench harness** | `bench/tier1/sweep_lib.py`, `turn_sweep.py`, `resume_parity_assumption_test.py` | Garble guard, scoring, parity assumptions |

**Defer or split:** `assets/logo-256.png`, large `bench/results/*.json` deltas, one-off diag scripts (`bench/diag_*.py`).

Suggested commit message when ready:

> Fix resume pack system-block ordering and Tier B startup probing; add qwen restore to GX10 swap helper.

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
| `moe_dense` | off | off | 0 | off |
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

| Order | Model | Tag | Status |
|-------|-------|-----|--------|
| — | Qwen3.5 hybrid FP8 | `qwen35_hybrid` | **Frozen** baseline |
| 1 | `engineering-group/EngGPT2-16B-A3B` | `enggpt2` | Queued |
| 2 | `google/gemma-3-27b-it` | `gemma27b` | Pass 1 partial (cp20 plugplay) |
| 3 | `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic` | `llama70b` | Pass 1 cp20 |
| 4 | `meta-llama/Meta-Llama-3-8B-Instruct` | `llama8b` | Pass 1 + scaling curve |
| — | Llama 4 Scout | — | **Dropped** (GB10 VRAM / vLLM maturity) |

### Swap procedure (sequential — one model on GB10 at a time)

GB10 unified memory holds one large checkpoint at a time. Use `:8000` for whichever model is live:

```bash
./bench/deploy_swap_gx10.sh qwen     # restore Tier A baseline → pri-inference + pri-data
./bench/deploy_swap_gx10.sh gemma    # → pri-gemma on :8000
./bench/run_model_matrix.sh --model-tag gemma27b --pass 1

./bench/deploy_swap_gx10.sh enggpt   # stops gemma, → pri-enggpt on :8000
./bench/run_model_matrix.sh --model-tag enggpt2 --pass 1

./bench/deploy_swap_gx10.sh llama    # Llama 3.3 70B FP8
./bench/deploy_swap_gx10.sh llama8b  # Llama 3 8B (Tier B control)
```

---

## Compare results

For each model matrix run, check against frozen Qwen:

| Metric | Qwen (overnight) | Gemma 27B | Llama 70B | Llama 8B |
|--------|------------------|-----------|-----------|----------|
| turn_sweep RESUME cp20 | 5/5 | 2/5 | 0/5 | 0/5 |
| turn_sweep TEXT cp20 | 5/5 | 5/5 | 5/5 | 2/5 |
| turn_sweep RESUME cp80 | 0/5 | — | — | — |
| geometry delta_uniformity | 100% | pass (fair audit) | — | pass post-fix |
| full Mamba parity @ cp80 | ~0.89–0.93 | N/A (Tier B) | N/A | N/A |

Report incompatibilities with: model id, `model_profile.json`, `manifest.json`, vLLM boot log.

---

## References

- [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) — Tier A/B/C policy
- [`BENCHMARKS.md`](BENCHMARKS.md) — harness details
- [`bench/results/overnight_20260624_003614/BENCHMARK_SUMMARY.md`](../bench/results/overnight_20260624_003614/BENCHMARK_SUMMARY.md) — frozen Qwen proof
- NLS [`RESUME_RESEARCH_ROADMAP.md`](../../NLS/docs/research/RESUME_RESEARCH_ROADMAP.md) — research framing
