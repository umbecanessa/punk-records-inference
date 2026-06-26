# Overview

Punk Records Inference is the **main home** for the Neural Ledger inference architecture: open vLLM plugin, Docker, reproducible benchmarks, and the full design narrative (patent pending U.S. **64/050,345**).

This repo ships **code, Docker, proof artifacts, and documentation** — including the [five-phase pipeline](NLS_PIPELINE.md), [research journey](JOURNEY.md), and [economics](ECONOMICS.md). Start with the [documentation map](PLATFORM.md).

---

## What problem this solves

Long agent sessions make every turn re-process the entire transcript through the transformer. That costs:

- **Prompt tokens** (even when history is unchanged)
- **GPU prefill time** (latency scales with context length)
- **VRAM pressure** on the active context window

PRI inverts part of that cost structure: prior turns live as **compressed KV state on disk**. Turn ≥ 2 sends only the new user message plus a small inject footprint — the model continues from stored state instead of re-reading inline history.

Text compression (e.g. Headroom) reduces **what you send**; it does not skip GPU attention over history. PRI reduces **what the GPU re-computes** — stored KV is re-injected, not re-prefilled. See [Economics](ECONOMICS.md) for measured savings from the June 2026 bench.

---

## NLS pipeline vs v0.1 OSS default

| | Full NLS ([NLS_PIPELINE.md](NLS_PIPELINE.md)) | v0.1 OSS default |
|---|-----|-----|
| **Pipeline** | Retrieve → inject → score → monitor → capture | **Capture + chain resume** (+ optional Swiss/overflow) |
| **Cross-session** | Swiss/BM25 fusion across memory pool | Retrieval opt-in; resume = same-session chain |
| **Production demo** | [punkrecords.live](https://punkrecords.live) — full stack | This repo — reproducible resume slice |
| **Code** | Production plugin (hosted API) | `pri/` package — PolyForm NC |

Phases 1–4 (retrieve, score, monitor) are documented in [NLS_PIPELINE.md](NLS_PIPELINE.md) and validated in the April 2026 production run ([BENCHMARKS](BENCHMARKS.md#historical-production-validation-april-2026)). v0.1 OSS proves **chain resume** on stock Qwen3.5 hybrid — the path agent loops need turn-to-turn.

---

## What we measured (2026-06-24)

Validated on **Qwen3.5-35B-A3B-FP8**, NVIDIA GPU ≥24 GB VRAM. Full tables: [Benchmarks](BENCHMARKS.md) · [Summary](../bench/results/overnight_20260624_003614/BENCHMARK_SUMMARY.md).

| Finding | Result |
|---------|--------|
| Recall on long12 chain (5 probes) | TEXT **5/5** · RESUME **5/5** · OVERFLOW **5/5** |
| Prompt tokens per recall (long12 mean) | TEXT **3743** → RESUME **42** (**~98.9%** saved) |
| Mean recall latency (long12) | TEXT **2885 ms** → RESUME **1549 ms** (**~46%** lower) |
| OpenCode agent session (seed 42) | PRI **6/6** · baseline **6/6** |
| RoPE pack geometry (post-fix) | **100%** delta_uniformity — uniform re-rotation delta −22 |
| Turn sweep RESUME at cp20–40 | **5/5** (matches TEXT) |
| Turn sweep RESUME at cp60–80 | **Cliff** — 3/5 then 0/5 while TEXT stays **5/5** |

Extended narrative: [Research — Key findings](../bench/results/overnight_20260624_003614/research/00_findings.md).

---

## Key discoveries (plain language)

### 1. Chain resume works for agent-length sessions

On a 12-noise-turn Marco chain, RESUME matches TEXT recall while sending **~3700 fewer prompt tokens** per probe. OpenCode-style multi-turn recall passes **6/6** against a full-transcript baseline.

### 2. RoPE re-rotation had a real geometry bug — now fixed

Pre-fix geometry audits showed a phantom outlier block (turn 59). After connector/resume fixes, **83/83 blocks** share the same RoPE delta with **100%** uniformity. Garble at long inject depth is **not** explained by pack geometry anymore.

### 3. Long-chain garble is a separate, open issue

At ~17–23k inject tokens (turn sweep cp60+), RESUME decode degrades while inline TEXT stays perfect. Investigation shows:

- **Not** missing inline facts (TEXT 5/5 with same history)
- **Not** tail noise alone (facts-only inject with 3 blocks still garbles)
- **Not** RoPE pack layout (geometry audit passes)

This points to **inject-mediated decode** under very long KV inject — documented as a v0.1 limitation, not a bench harness failure.

### 4. Default inject mode: `resume`

Inject-mode compare shows `resume` matches `resume_overflow` on short/long12 for recall; overflow does **not** recover the cp60+ cliff. **`resume`** is the v0.1 default.

### 5. Storage is cheap relative to re-prefill

A full turn-sweep session produced **143 captures · ~648 MB** on disk — acceptable vs repeated 20k+ token prefills every turn.

### 6. Plug-and-play matrix (Tier B) — mechanical yes, parity no

June 2026 matrix runs on GB10 (`docs/MODEL_MATRIX.md`) mounted Gemma 27B, Llama 3.3 70B FP8, and Llama 3 8B with **zero code changes** — `startup_profile.py` probes topology and boots vLLM. A **resume pack ordering bug** (system block after turn KV instead of before) caused Tier B garble; fixed in `pri/resume.py` + `pri/connector.py`.

| Model | cp20 TEXT | cp20 RESUME |
|-------|-----------|-------------|
| Qwen3.5 MoE (Tier A) | 5/5 | **5/5** |
| Gemma 3 27B | 5/5 | 2/5 |
| Llama 3.3 70B FP8 | 5/5 | 0/5 |
| Llama 3 8B | 2/5 | 0/5 (3/5 @ cp5) |

**Takeaway:** PRI auto-configures across architectures; **full resume parity is Tier A today**. Tier B validates K/V inject and recall curves — useful for research, not a production guarantee on arbitrary checkpoints.

---

## Who this is for

| You want… | Start here |
|-----------|------------|
| Run it locally | [Installation](getting-started/installation.md) → [Quickstart](getting-started/quickstart.md) |
| Understand capture vs resume | [Core concepts](getting-started/concepts.md) |
| Wire an agent client | [Integrating OpenCode](guides/integrating-opencode.md) · [Client contract](CLIENT_CONTRACT.md) |
| Reproduce proof numbers | [Benchmarks](BENCHMARKS.md) · [`bench/README.md`](../bench/README.md) |
| Read charts and per-probe data | [Research index](../bench/results/overnight_20260624_003614/research/README.md) |
| Full NLS pipeline story | [neural-ledger-system](https://github.com/umbecanessa/neural-ledger-system) |
| License / commercial use | [Licensing](LICENSING.md) |

---

## Repository map

```
punk-records-inference/
├── pri/           # vLLM connector, store, resume, agent shim
├── docker/        # Image + compose
├── bench/         # Reproducible harnesses + published results
├── spec/          # .nls manifest schema
├── docs/          # You are here
└── tests/         # Unit tests (no GPU)
```

---

## Related projects

| Project | Relationship |
|---------|--------------|
| [Neural Ledger System](https://github.com/umbecanessa/neural-ledger-system) | Architecture + research narrative (retrieval-first docs; PRI implements the resume slice) |
| [Punk Records demo](https://punkrecords.live) | Hosted product using NLS-style inference (separate from this BYOC repo) |

Prompt-compression tools (e.g. Headroom) reduce **what you send**; PRI reduces **what the GPU re-computes**. They complement each other.
