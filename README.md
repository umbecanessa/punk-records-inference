# Punk Records Inference

**KV-state persistence for vLLM** — capture attention + hybrid recurrent state after each
agent turn, store on disk (`.nls`), re-inject on the next request so the model does not
re-prefill full history.

> **Status:** Private development repo. History will be squashed before public release.

## Proof (GX10, stock Qwen3.5-35B-A3B-FP8, 2026-06-23)

| Bench | Result | Artifact |
|-------|--------|----------|
| Tier-1 Marco facts (seed 42) | TEXT 5/5 · RESUME 5/5 | `bench/results/tier1_marco_facts_42.json` |
| OpenCode long session (seed 42) | RECALL 6/6 | `bench/results/opencode_long_session.json` |
| Manifest proof turn 2 (KL #648) | `rope_start=24` | `bench/results/manifest_opencode_t2.json` |

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for reproduction commands.

## What this is

| In scope | Out of scope (legacy) |
|----------|----------------------|
| Turn capture → `.nls` | MoE expert slots / router bias |
| Chain resume inject + RoPE | MoE router bias, legacy CAMM, streaming scorer |
| Optional Swiss retrieval | Hosted Punk Records SaaS |
| Agent middleware (strip + capture) | Model weights (BYOC) |

See [docs/SHIP_PLAN.md](docs/SHIP_PLAN.md) for the full release plan.

## Layout

```
pri/        Python package (connector, store, resume, agent_shim, admin, …)
patches/    vLLM source patches (build time)
docker/     Dockerfile + compose + start.sh
bench/      Tier-1 + OpenCode harnesses
spec/       .nls manifest schema + validator
tests/      Unit tests (no GPU)
docs/       Architecture, client contract, Docker
```

## Quick start

```bash
# Build + run (requires NVIDIA GPU + checkpoint on host)
export MODEL_PATH=$HOME/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/<revision>
docker compose -f docker/compose.yaml up --build

# Health check
curl -s http://127.0.0.1:8000/v1/models

# Tier-1 bench (host Python, live server)
pip install requests
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000

# OpenCode long session (direct vLLM)
./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000 --seed 42

# Unit tests (no vLLM)
pip install pytest torch zstandard
pytest tests/ -q
```

Env vars keep the `NLS_*` prefix for migration; `PRI_*` rename planned for v0.2.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Client contract](docs/CLIENT_CONTRACT.md) — `kv_transfer_params`
- [Docker](docs/DOCKER.md)
- [Supported models](docs/SUPPORTED_MODELS.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Limitations](docs/LIMITATIONS.md)

## License

TBD — community license + patent notice (provisional 64/050,345).
