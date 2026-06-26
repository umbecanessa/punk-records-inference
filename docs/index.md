# Punk Records Inference — Documentation

**Main entry point** for the Neural Ledger inference architecture — code, Docker, benchmarks, and design narrative.

**Start here:** [Overview](OVERVIEW.md) · [Documentation map](PLATFORM.md) · [Install](getting-started/installation.md) → [Quickstart](getting-started/quickstart.md)

---

## I want to…

| Goal | Guide |
|------|-------|
| Understand the problem and proof | [Overview](OVERVIEW.md) |
| Read how we got here | [Journey](JOURNEY.md) |
| Full five-phase architecture | [NLS pipeline](NLS_PIPELINE.md) |
| vLLM plugin / `pri/` layout | [PRI runtime](ARCHITECTURE.md) |
| Install and run | [Installation](getting-started/installation.md) · [Quickstart](getting-started/quickstart.md) |
| Capture, resume, overflow | [Core concepts](getting-started/concepts.md) |
| Integrate an agent client | [Client contract](CLIENT_CONTRACT.md) · [OpenCode](guides/integrating-opencode.md) |
| Reproduce benchmark numbers | [Benchmarks](BENCHMARKS.md) |
| Extended analytics (charts) | [Overnight research](../bench/results/overnight_20260624_003614/research/README.md) |
| Economics & savings | [Economics](ECONOMICS.md) |
| Model matrix (Tier B) | [MODEL_MATRIX.md](MODEL_MATRIX.md) |
| Everything in one map | [PLATFORM.md](PLATFORM.md) |

---

## Reference

| Document | Description |
|----------|-------------|
| [NLS pipeline](NLS_PIPELINE.md) | Retrieve → inject → score → monitor → capture |
| [PRI runtime](ARCHITECTURE.md) | Docker, `pri/`, phase mapping |
| [Journey](JOURNEY.md) | Research arc (LoRA → MoE → KV) |
| [Research log](RESEARCH_LOG.md) | Curated experiment timeline |
| [Benchmarks](BENCHMARKS.md) | Proof tables + reproduce |
| [Limitations](LIMITATIONS.md) | Known cliffs, scope |
| [Licensing](LICENSING.md) | PolyForm NC, patent notice |
| [`.nls` spec](../spec/) | Manifest schema |

---

## For AI agents

[`llms.txt`](../llms.txt) at repo root.
