# Punk Records Inference

**KV-state persistence for vLLM** — capture attention + hybrid recurrent state after each
agent turn, store on disk (`.nls`), re-inject on the next request so the model does not
re-prefill full history.

> **Status:** Private development repo. History will be squashed before public release.

## What this is

| In scope | Out of scope (legacy) |
|----------|----------------------|
| Turn capture → `.nls` | MoE expert slots / router bias |
| Chain resume inject + RoPE | Hippocampus, CAMM, streaming-scorer hot-swap |
| Optional Swiss retrieval | Hosted Punk Records SaaS |
| Agent middleware (strip + capture) | Model weights (BYOC) |

See [docs/SHIP_PLAN.md](docs/SHIP_PLAN.md) for the full release plan.

## Layout

```
pri/              Python package (canonical; migration from nls_vllm_plugin in progress)
nls_vllm_plugin/  Compatibility shim for vLLM module paths (temporary)
patches/          vLLM source patches (build time)
docker/           Container entrypoint + Dockerfile
bench/            Replication harnesses
spec/             .nls format spec
docs/             Architecture + client contract (stubs)
```

## Quick start (development)

Extracted from NLS branch `exp/chain-of-latest` @ `71a65774`.

```bash
# After Docker image is built (Phase 2):
docker run --gpus all \
  -v pri-data:/data/pri \
  -v /path/to/checkpoint:/model:ro \
  -e MODEL_PATH=/model \
  -p 8000:8000 \
  ghcr.io/punkrecords/inference:dev
```

Env vars keep the `NLS_*` prefix for migration; `PRI_*` rename planned for v0.2.

## Development workflow

1. Work in this repo only (private until launch).
2. Phase 0: KV-only cleanup — remove MoE wiring, parameterize `MODEL_PATH`.
3. Phase 1–3: rename to `pri/`, Docker image, docs, tier-1 bench.
4. Before public release: squash history → single clean initial commit.

## License

TBD — community license + patent notice (provisional 64/050,345).
