# Documentation — Punk Records Inference

**This repository is the main entry point** for the Neural Ledger inference architecture: open code, Docker, reproducible benchmarks, and the full design narrative (patent pending U.S. **64/050,345**).

The sibling repo [neural-ledger-system](https://github.com/umbecanessa/neural-ledger-system) is a **legacy docs mirror** — new material lands here first.

---

## Start here

| Audience | Path |
|----------|------|
| **New visitor** | [Overview](OVERVIEW.md) → [Journey](JOURNEY.md) |
| **Implementer** | [Installation](getting-started/installation.md) → [Quickstart](getting-started/quickstart.md) → [Client contract](CLIENT_CONTRACT.md) |
| **Researcher** | [Benchmarks](BENCHMARKS.md) → [overnight research](../bench/results/overnight_20260624_003614/research/README.md) |
| **Architect** | [NLS five-phase pipeline](NLS_PIPELINE.md) → [PRI runtime](ARCHITECTURE.md) |

---

## Architecture & history

| Document | Content |
|----------|---------|
| [Overview](OVERVIEW.md) | Problem, proof headline, key discoveries |
| [NLS pipeline](NLS_PIPELINE.md) | Full architecture — retrieve, inject, score, monitor, capture |
| [PRI runtime](ARCHITECTURE.md) | vLLM plugin layout, `pri/` modules, phase mapping |
| [Journey](JOURNEY.md) | LoRA → MoE → KV injection → OSS (650+ experiments) |
| [Research log](RESEARCH_LOG.md) | Curated experiment timeline |
| [Core concepts](getting-started/concepts.md) | Capture vs resume vs Swiss vs overflow |

---

## Benchmarks & analytics

| Document | Content |
|----------|---------|
| [Benchmarks](BENCHMARKS.md) | Reproduce commands + headline tables |
| [Model matrix](MODEL_MATRIX.md) | Tier B plug-and-play (Gemma, Llama) |
| [Economics](ECONOMICS.md) | Compute, token, energy, and storage savings (June 2026 bench) |
| [Overnight run](../bench/results/overnight_20260624_003614/) | Frozen JSON + [research charts](../bench/results/overnight_20260624_003614/research/README.md) |
| [Demo](https://punkrecords.live) | Live conversational proof + API log |

### Benchmark map

| Question | Where |
|----------|-------|
| Session resume 5/5, ~3700 tok saved? | [BENCHMARKS](BENCHMARKS.md) · [00_findings](../bench/results/overnight_20260624_003614/research/00_findings.md) |
| Cross-session OpenCode 4/4, 99.3% savings? | [BENCHMARKS — April 2026](BENCHMARKS.md#historical-production-validation-april-2026) |
| LongMemEval TEXT = KV parity? | [BENCHMARKS — parity](BENCHMARKS.md#historical-production-validation-april-2026) |
| Turn-sweep cliff cp60+? | [06_turn_sweep_scaling](../bench/results/overnight_20260624_003614/research/06_turn_sweep_scaling.md) |
| Other models plug-and-play? | [MODEL_MATRIX](MODEL_MATRIX.md) |

---

## Operations

| Document | Content |
|----------|---------|
| [Supported models](SUPPORTED_MODELS.md) | Tier A/B/C |
| [Docker](DOCKER.md) | Image, compose, volumes |
| [Limitations](LIMITATIONS.md) | Known cliffs, scope |
| [Licensing](LICENSING.md) | PolyForm NC, commercial path |
| [Integrating OpenCode](guides/integrating-opencode.md) | Agent harness |
| [`.nls` spec](../spec/) | Manifest schema |

---

## Two inject paths (same plugin)

| Path | When | Docs |
|------|------|------|
| **Chain resume** | Same agent session, turn ≥ 2 | Default in OSS · [concepts](getting-started/concepts.md) |
| **Retrieval inject** | Cross-session, memory pool search | [NLS pipeline § Phases 1–4](NLS_PIPELINE.md) · validated Apr 2026 demo |

Both share phantom KV injection, RoPE correction, and `.nls` capture. v0.1 OSS defaults to **chain resume**; full retrieval+scoring ships in production.

---

## Production posture (June 2026)

| Capability | Status |
|------------|--------|
| Qwen3.5 hybrid + chain resume | Validated — reproducible OSS |
| Full five-phase stack | Production demo ([punkrecords.live](https://punkrecords.live)) |
| Long inject cliff (cp60+) | Documented limitation |
| Tier B models | Experimental |
