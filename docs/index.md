# Punk Records Inference — Documentation

**KV-state persistence for vLLM** — capture attention and hybrid recurrent state after each agent turn, store on disk (`.nls`), re-inject on the next request so the model skips re-prefilling full history.

New users: [Installation](getting-started/installation.md) → [Quickstart](getting-started/quickstart.md) → [Core concepts](getting-started/concepts.md).

**Public release:** [Announcement playbook](ANNOUNCE.md) · [Licensing](LICENSING.md) · [Changelog](../CHANGELOG.md) · [License](../LICENSE)

---

## How it works (30 seconds)

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

| Goal | Start here |
|------|------------|
| Install prerequisites and checkpoint | [Installation](getting-started/installation.md) |
| Run the server locally | [Quickstart](getting-started/quickstart.md) |
| Understand capture vs resume vs overflow | [Core concepts](getting-started/concepts.md) |
| Wire an OpenCode-style agent client | [Integrating OpenCode](guides/integrating-opencode.md) |
| Integrate a custom client | [Client contract](CLIENT_CONTRACT.md) |
| Look up env vars | [Environment variables](reference/env-vars.md) |
| Deploy with Docker | [Docker](DOCKER.md) |
| Pick a supported model | [Supported models](SUPPORTED_MODELS.md) |
| Reproduce benchmark results | [Benchmarks](BENCHMARKS.md) |
| Read bench research analysis | [overnight run research/](../bench/results/overnight_20260624_003614/research/README.md) |
| Announce or launch publicly | [Announcement playbook](ANNOUNCE.md) |
| Fix something broken | [Troubleshooting](guides/troubleshooting.md) |
| Know what v0.1 does *not* do | [Limitations](LIMITATIONS.md) |
| Understand community vs commercial use | [Licensing](LICENSING.md) |

---

## Documentation map

### Getting started

| Guide | Description |
|-------|-------------|
| [Installation](getting-started/installation.md) | GPU, Docker, checkpoint, verify |
| [Quickstart](getting-started/quickstart.md) | Build, run, health check, first bench |
| [Core concepts](getting-started/concepts.md) | Capture, resume, Swiss, overflow, `.nls` |

### Guides

| Guide | Description |
|-------|-------------|
| [Guides index](guides/index.md) | Integration and operations |
| [Integrating OpenCode](guides/integrating-opencode.md) | Agent shim, KVP, reference harness |
| [Troubleshooting](guides/troubleshooting.md) | Common failures and diagnostics |

### Reference

| Guide | Description |
|-------|-------------|
| [Reference index](reference/index.md) | API and configuration |
| [Environment variables](reference/env-vars.md) | Consolidated `NLS_*` list |
| [Architecture](ARCHITECTURE.md) | Runtime flow, package layout, subsystems |
| [Client contract](CLIENT_CONTRACT.md) | `kv_transfer_params` fields and examples |
| [Docker](DOCKER.md) | Image, compose, env vars, volumes |
| [Supported models](SUPPORTED_MODELS.md) | Qwen hybrid, topology requirements |
| [Benchmarks](BENCHMARKS.md) | Tier-1, OpenCode, inject compare, reproduction commands |
| [Limitations](LIMITATIONS.md) | Scope, legal, known gaps |
| [Licensing](LICENSING.md) | Community vs commercial use, patent notice |

### Spec

| Path | Description |
|------|-------------|
| [`spec/`](../spec/) | `.nls` manifest schema and validator |

### Maintainers

Planning docs for release and benchmarking — see [internal/README.md](internal/README.md).

| Guide | Description |
|-------|-------------|
| [Ship plan](internal/SHIP_PLAN.md) | Scope, naming, release checklist |
| [Bench data plan](internal/BENCH_DATA_PLAN.md) | Measurement matrix for value case |

---

## Repository layout

| Path | Role |
|------|------|
| `pri/` | Python package — connector, store, resume, agent shim, admin |
| `patches/` | vLLM source patches (build time) |
| `docker/` | Dockerfile, compose, `start.sh` |
| `bench/` | Tier-1 + OpenCode harnesses |
| `spec/` | `.nls` manifest schema |
| `tests/` | Unit tests (no GPU) |
| `assets/` | Brand assets — `logo.png`, social preview (see `assets/README.md`) |

---

## For AI agents

Read [`llms.txt`](../llms.txt) at the repo root for a compact index of entry points and doc links.
