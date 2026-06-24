# Punk Records Inference

<p align="center">
  <img src="assets/logo-composite.png" alt="Punk Records Inference" width="180" />
</p>

<p align="center">
  <strong>KV-state persistence for vLLM</strong> — capture attention + hybrid recurrent state after each agent turn,<br />
  store on disk (<code>.nls</code>), re-inject on the next request so the model skips re-prefilling full history.
</p>

<p align="center">
  <a href="https://github.com/umbecanessa/punk-records-inference/actions/workflows/ci.yml"><img src="https://github.com/umbecanessa/punk-records-inference/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/vLLM-plugin-teal.svg" alt="vLLM plugin">
  <img src="https://img.shields.io/badge/local--first-black.svg" alt="Local-first">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-PolyForm--NC-lightgrey.svg" alt="License PolyForm Noncommercial"></a>
</p>

<p align="center">
  <a href="docs/OVERVIEW.md"><strong>Overview</strong></a> ·
  <a href="docs/getting-started/installation.md"><strong>Installation</strong></a> ·
  <a href="docs/getting-started/quickstart.md"><strong>Quickstart</strong></a> ·
  <a href="docs/index.md">Docs</a> ·
  <a href="#proof">Proof</a> ·
  <a href="docs/BENCHMARKS.md">Benchmarks</a> ·
  <a href="docs/LICENSING.md">Licensing</a> ·
  <a href="llms.txt">llms.txt</a>
</p>

<p align="center"><sub>
  <b>AI agents / LLMs:</b> read <a href="llms.txt"><code>llms.txt</code></a> for entry points and doc links.
</sub></p>

---

**Press a record of each turn's KV cache. Replay it on the next request instead of re-reading the full transcript.**

PRI is a vLLM plugin: one Docker container, OpenAI-compatible API on port 8000, bring-your-own checkpoint. It pairs with agent clients (OpenCode, custom harnesses) and complements prompt-compression layers — compression shrinks what you send; PRI skips re-computing what the model already processed.

---

## What it does

- **Turn capture** — serialize attention KV + hybrid recurrent state → compressed `.nls` manifests
- **Chain resume** — inject prior turn state on turn ≥ 2; skip expensive re-prefill
- **Agent middleware** — strip transcript, set `memory_capture_start`, enrich `kv_transfer_params` automatically
- **Optional overflow** — resume + semantic retrieval when context trim evicts tokens (opt-in)
- **Model-aware startup** — probes `config.json` and gates layer env by inject mode

## How it works (30 seconds)

```
Agent client (OpenCode, curl, LangChain)
        │   chat completions + kv_transfer_params
        ▼
┌─────────────────────────────────────────────────────┐
│  vLLM OpenAI API :8000                               │
│  AgentShim  →  strip · capture_start · chain meta   │
│  Connector  →  WRITE .nls  |  READ resume/overflow  │
└─────────────────────────────────────────────────────┘
        │
        ▼
/data/pri  (captures/*.nls + index.jsonl)     /model  (BYOC)
```

→ [Architecture](docs/ARCHITECTURE.md) · [Core concepts](docs/getting-started/concepts.md) · [Client contract](docs/CLIENT_CONTRACT.md)

---

## Proof

**Qwen3.5-35B-A3B-FP8 · NVIDIA ≥24 GB VRAM · 2026-06-24**

| Bench | Result | Notes |
|-------|--------|-------|
| Inject mode compare (long12) | TEXT 5/5 · RESUME 5/5 · OVERFLOW 5/5 | **~3701 prompt tokens saved** per recall vs inline TEXT |
| Tier-1 Marco facts | TEXT 5/5 · RESUME 5/5 | Local + OpenRouter TEXT baseline |
| OpenCode long session (seed 42) | PRI 6/6 · baseline 6/6 | Agent-style multi-turn recall |
| Turn sweep cp20–80 | cp20–40: 5/5 RESUME; cp60+: documented cliff | TEXT 5/5 at all checkpoints |
| RoPE geometry audit | pass · 100% delta_uniformity | Garble at cp60+ is inject/decode, not pack geometry |

Default inject mode: **`resume`**. [Overview & findings](docs/OVERVIEW.md) · [Key findings](bench/results/overnight_20260624_003614/research/00_findings.md) · [Full tables](docs/BENCHMARKS.md) · [Summary](bench/results/overnight_20260624_003614/BENCHMARK_SUMMARY.md) · [Research](bench/results/overnight_20260624_003614/research/README.md)

Reproduce:

```bash
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
./bench/run_suite.sh --tier mode-compare --seed 42 --base-url http://127.0.0.1:8000
```

---

## Get started (60 seconds)

```bash
git clone https://github.com/umbecanessa/punk-records-inference.git
cd punk-records-inference

export MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/<revision>
docker compose -f docker/compose.yaml up --build

curl -s http://127.0.0.1:8000/v1/models
```

```bash
pip install requests
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000

pip install pytest torch zstandard && pytest tests/ -q
```

[Installation](docs/getting-started/installation.md) · [Quickstart](docs/getting-started/quickstart.md)

---

## When to use · When to skip

**Great fit if you…**

- run long agent sessions on **self-hosted vLLM** and pay in GPU time for re-prefill
- want **local-first** persistence — captures stay on disk under your control
- use OpenAI-compatible clients with `kv_transfer_params` (or enable agent shim)

**Skip it if you…**

- only need prompt compression — PRI optimizes GPU state, not text size
- run on **hosted APIs** without KV transfer hooks
- need **multi-node** replicated KV — v0.1 is single-node BYOC only

---

## Compared to

PRI persists **model internal state** (KV + hybrid recurrent tensors), not chat text.

| | What it optimizes | Deploy | Needs vLLM |
| --- | --- | --- | --- |
| **PRI** | Skip re-prefill via captured KV state | Docker plugin | Yes |
| Prompt compression tools | Shrink tool outputs and logs before the model | Proxy / library | No |
| Provider prompt cache | Prefix reuse on identical prompts | Provider-native | No |

Prompt compression and PRI stack: compress incoming context, then resume prior turns without re-reading the full transcript.

---

## Scope (v0.1)

| In scope | Out of scope |
|----------|--------------|
| Turn capture → `.nls` | MoE expert slots / router bias |
| Chain resume inject + RoPE | Legacy CAMM, streaming scorer |
| Optional semantic retrieval | Hosted SaaS (separate product) |
| Agent middleware | Model weights (BYOC) |

Environment variables use the `NLS_*` prefix in v0.1. See [env vars](docs/reference/env-vars.md).

---

## Documentation

| Start here | Go deeper |
| --- | --- |
| [Overview](docs/OVERVIEW.md) | [Architecture](docs/ARCHITECTURE.md) |
| [Installation](docs/getting-started/installation.md) | [Client contract](docs/CLIENT_CONTRACT.md) |
| [Quickstart](docs/getting-started/quickstart.md) | [Benchmarks](docs/BENCHMARKS.md) |
| [Integrating OpenCode](docs/guides/integrating-opencode.md) | [Limitations](docs/LIMITATIONS.md) |
| [Licensing](docs/LICENSING.md) | [Research analysis](bench/results/overnight_20260624_003614/research/README.md) |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md)

```bash
pip install pytest torch zstandard && pytest tests/ -q
```

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for research, personal use, and noncommercial organizations. **Commercial production use** requires a separate agreement. See [Licensing](docs/LICENSING.md). U.S. Provisional Patent Application No. 64/050,345.
