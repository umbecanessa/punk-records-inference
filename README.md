# Punk Records Inference

<p align="center">
  <img src="assets/logo-composite.png" alt="Punk Records Inference" width="180" />
</p>

<p align="center">
  <strong>Stateful inference for self-hosted agents</strong><br />
  Skip GPU re-prefill — persist turn KV state to disk (<code>.nls</code>) and re-inject on the next request.
</p>

<p align="center">
  <strong>90–99% fewer prompt prefill tokens · vLLM plugin · Docker · BYOC · reproducible benches</strong>
</p>

<p align="center">
  <a href="https://github.com/umbecanessa/punk-records-inference/actions/workflows/ci.yml"><img src="https://github.com/umbecanessa/punk-records-inference/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <img src="https://img.shields.io/badge/vLLM-plugin-teal.svg" alt="vLLM plugin">
  <img src="https://img.shields.io/badge/local--first-black.svg" alt="Local-first">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-PolyForm--NC-lightgrey.svg" alt="License PolyForm Noncommercial"></a>
</p>

<p align="center">
  <a href="docs/OVERVIEW.md"><strong>Overview</strong></a> ·
  <a href="docs/getting-started/installation.md"><strong>Install</strong></a> ·
  <a href="docs/getting-started/quickstart.md"><strong>Quickstart</strong></a> ·
  <a href="docs/BENCHMARKS.md"><strong>Proof</strong></a> ·
  <a href="docs/ECONOMICS.md"><strong>Economics</strong></a> ·
  <a href="docs/PLATFORM.md"><strong>Docs</strong></a> ·
  <a href="llms.txt"><code>llms.txt</code></a>
</p>

---

Every agent turn re-runs the full transcript on your GPU. Text compression shrinks what you send — it does not skip the attention pass. **PRI reuses computed KV state** so turn ≥ 2 does not re-prefill unchanged history.

Tools like [Headroom](https://github.com/headroomlabs-ai/headroom) compress what the agent reads; PRI skips what the GPU already computed. They complement each other.

PRI is an open **vLLM plugin** and the documentation home for the Neural Ledger inference architecture (patent pending U.S. **64/050,345**). One Docker container, OpenAI-compatible API on port 8000, bring-your-own checkpoint.

Full design narrative: [docs/JOURNEY.md](docs/JOURNEY.md) · [docs/NLS_PIPELINE.md](docs/NLS_PIPELINE.md) · [docs/PLATFORM.md](docs/PLATFORM.md)

```
Agent client (OpenCode, curl, LangChain)
        │  chat completions + kv_transfer_params
        ▼
┌──────────────────────────────────────────────┐
│  vLLM :8000                                   │
│  AgentShim → strip transcript · chain meta   │
│  Connector → WRITE .nls  |  READ resume      │
└──────────────────────────────────────────────┘
        │
        ▼
/data/pri  (captures/*.nls)          /model  (BYOC)
```

---

## Proof (2026-06-24)

**Qwen3.5-35B-A3B-FP8** · NVIDIA ≥24 GB VRAM · default inject mode **`resume`**

| | Inline history (TEXT) | Chain resume (RESUME) |
|---|:---:|:---:|
| Recall on agent-length sessions | 5/5 · 6/6 OpenCode | 5/5 · 6/6 OpenCode |
| Mean prompt tokens per recall | 3,743 | 42 (**98.9% saved**) |
| Mean recall latency | 2,885 ms | 1,549 ms (**46% lower**) |
| Est. GPU energy per recall | 0.200 Wh | 0.108 Wh (**46% lower**) |
| RoPE pack geometry (post-fix) | — | **100%** pass |

Long inject chains (~17k+ tokens) show a documented recall cliff; token savings stay >99% but recall fails. Details: [Key findings](bench/results/overnight_20260624_003614/research/00_findings.md) · [Benchmarks](docs/BENCHMARKS.md) · [Economics](docs/ECONOMICS.md)

```bash
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
./bench/run_suite.sh --tier mode-compare --seed 42 --base-url http://127.0.0.1:8000
```

---

## Quick start

```bash
git clone https://github.com/umbecanessa/punk-records-inference.git
cd punk-records-inference

export MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/<revision>
docker compose -f docker/compose.yaml up --build

curl -s http://127.0.0.1:8000/v1/models
```

→ [Installation guide](docs/getting-started/installation.md) · [Quickstart](docs/getting-started/quickstart.md) · [OpenCode integration](docs/guides/integrating-opencode.md)

---

## Documentation

| | |
|---|---|
| [Documentation map](docs/PLATFORM.md) | **Start here** — architecture, benchmarks, journey |
| [Overview](docs/OVERVIEW.md) | Problem, proof headline, discoveries |
| [Economics](docs/ECONOMICS.md) | Compute, token, energy, and storage savings (June 2026 bench) |
| [NLS pipeline](docs/NLS_PIPELINE.md) | Five-phase architecture (retrieve → capture) |
| [Journey](docs/JOURNEY.md) | Research history (LoRA → MoE → KV) |
| [Architecture](docs/ARCHITECTURE.md) | vLLM runtime, `pri/` modules |
| [Benchmarks](docs/BENCHMARKS.md) | Tables + reproduce commands |
| [Core concepts](docs/getting-started/concepts.md) | Capture, resume, overflow |
| [Client contract](docs/CLIENT_CONTRACT.md) | `kv_transfer_params` and chain metadata |
| [Limitations](docs/LIMITATIONS.md) | Known cliffs, scope boundaries, legal |
| [Licensing](docs/LICENSING.md) | PolyForm NC, commercial path, patent notice |

---

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md)

```bash
pip install pytest torch zstandard requests && pytest tests/ -q
```

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for research, personal use, and noncommercial organizations. **Commercial production use** requires a separate agreement. See [Licensing](docs/LICENSING.md). U.S. Provisional Patent Application No. 64/050,345.
