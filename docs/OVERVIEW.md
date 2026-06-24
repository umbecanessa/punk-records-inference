# Overview

Punk Records Inference (PRI) is the **open reference implementation** of KV-state persistence for vLLM: capture attention and hybrid recurrent state after each agent turn, store it on disk (`.nls`), and re-inject on the next request so the model skips re-prefilling full history.

This repo ships **code, Docker, benchmarks, and reproducible proof data**. For the broader architecture narrative — research history, full pipeline phases, and the hosted demo — see the companion repository [**Neural Ledger System (NLS)**](https://github.com/umbecanessa/neural-ledger-system).

Both projects relate to U.S. Provisional Patent Application No. **64/050,345**.

---

## What problem this solves

Long agent sessions make every turn re-process the entire transcript through the transformer. That costs:

- **Prompt tokens** (even when history is unchanged)
- **GPU prefill time** (latency scales with context length)
- **VRAM pressure** on the active context window

PRI inverts part of that cost structure: prior turns live as **compressed KV state on disk**. Turn ≥ 2 sends only the new user message plus a small inject footprint — the model continues from stored state instead of re-reading inline history.

---

## NLS architecture vs this repository

| | [Neural Ledger System](https://github.com/umbecanessa/neural-ledger-system) | **Punk Records Inference** (this repo) |
|---|-----|-----|
| **Purpose** | Architecture docs, research narrative, hosted demo story | Reference **source code** + Docker + benches |
| **Pipeline** | Full 5-phase pipeline (retrieve → inject → score → monitor → capture) | **v0.1 focus:** turn **capture + chain resume** (+ optional Swiss/overflow) |
| **Default memory path** | Documented as **semantic retrieval first** (Swiss/BM25 fusion) | Bench-validated default: **`resume`** (prior-turn KV chain) |
| **Code** | Architecture only (plugin proprietary in research arc) | `pri/` package, patches, harnesses — PolyForm NC |
| **Proof** | Demo + LongMemEval narrative in NLS repo | Machine-readable JSON + [research pages](../bench/results/overnight_20260624_003614/research/README.md) |

The NLS [ARCHITECTURE.md](https://github.com/umbecanessa/neural-ledger-system/blob/main/ARCHITECTURE.md) describes retrieval-first injection across many sessions. PRI v0.1 proves the **chain-resume** path that agent loops need turn-to-turn: same `memory_base_session`, inject prior `.nls` blocks, skip re-prefill. Semantic retrieval (Swiss) and overflow compose remain available but are not the default.

**If you read NLS first:** treat Phases 1–4 (retrieve, inject, score, monitor) as the full vision; treat this repo as the **production-shaped slice** validated on stock Qwen3.5 hybrid + vLLM.

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
