# Punk Records Inference — Documentation

**KV-state persistence for vLLM** — capture attention and hybrid recurrent state after each agent turn, store on disk (`.nls`), re-inject on the next request so the model skips re-prefilling full history.

**Start here:** [Installation](getting-started/installation.md) → [Quickstart](getting-started/quickstart.md) → [Core concepts](getting-started/concepts.md)

---

## How it works

```text
Agent client (OpenCode, curl, LangChain)
        │
        ▼
vLLM OpenAI API :8000
  AgentShim  →  strip transcript · capture_start · chain metadata
  Connector  →  WRITE turn capture → .nls  |  READ resume / overflow
        │
        ▼
/data/pri  (captures/*.nls + index.jsonl)
```

Details: [Architecture](ARCHITECTURE.md)

---

## I want to…

| Goal | Guide |
|------|-------|
| Install GPU, Docker, and a model checkpoint | [Installation](getting-started/installation.md) |
| Run the server and smoke-test | [Quickstart](getting-started/quickstart.md) |
| Understand capture, resume, and overflow | [Core concepts](getting-started/concepts.md) |
| Integrate an OpenCode-style agent | [Integrating OpenCode](guides/integrating-opencode.md) |
| Integrate a custom HTTP client | [Client contract](CLIENT_CONTRACT.md) |
| Look up environment variables | [Environment variables](reference/env-vars.md) |
| Deploy with Docker | [Docker](DOCKER.md) |
| Choose a supported model | [Supported models](SUPPORTED_MODELS.md) |
| Reproduce published benchmark numbers | [Benchmarks](BENCHMARKS.md) |
| Read extended analysis (charts, tables) | [Benchmark research](../bench/results/overnight_20260624_003614/research/README.md) |
| Understand the license | [Licensing](LICENSING.md) |
| Know scope limits and known issues | [Limitations](LIMITATIONS.md) |
| Fix a failure | [Troubleshooting](guides/troubleshooting.md) |

---

## Reference

| Document | Description |
|----------|-------------|
| [Architecture](ARCHITECTURE.md) | Runtime flow, package layout |
| [Client contract](CLIENT_CONTRACT.md) | `kv_transfer_params` fields |
| [Docker](DOCKER.md) | Image, compose, volumes |
| [Benchmarks](BENCHMARKS.md) | Proof results and reproduction |
| [Limitations](LIMITATIONS.md) | v0.1 scope and known gaps |
| [Licensing](LICENSING.md) | PolyForm Noncommercial + commercial path |
| [`.nls` spec](../spec/) | Manifest schema and validator |

---

## For AI agents

Read [`llms.txt`](../llms.txt) at the repo root for a compact index.
