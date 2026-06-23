# Limitations (v0.1)

## Product scope

- **KV capture/resume only** — not text compression, not MoE routing, not hosted SaaS
- **Single-node** — one vLLM process per container; no distributed KV transfer
- **BYOC** — you provide and license the base model checkpoint

## Legacy subsystems (not shipped)

MoE router bias, CAMM, streaming-scorer hot-swap, and thalamus paths exist in the
NLS research branch but are **not included** in this repo. The default image runs
KV capture/resume only.

## Agent middleware

`AgentShimMiddleware` derives turn index from client user-message count (no DB).
Hosted Punk Records uses Prisma for sticky chain ids and compaction detection;
standalone inference uses header/`nls_chain_id` hints and transcript heuristics.

v0.1 clients may send full `kv_transfer_params` in the request body (Nest proxy
pattern) or rely on agent shim to inject them. Both paths are supported; KL #648
`memory_capture_start` is required for `rope_start > 0` in capture manifests.
See `bench/opencode/nls_kvp_helpers.py` for harness parity with production.

## Memory store

- Index is JSONL on disk under `NLS_MEMORY_DIR` — no replication or HA
- Swiss retrieval requires `sentence-transformers` (installed at container start)
- Very large indexes (>250k entries) need tuning via `NLS_MAX_MEMORIES`

## Resume inject

- Resume walks `base_session_id` chain blocks — cross-chain retrieval does not apply
- RoPE re-rotation assumes Qwen hybrid head layout on pinned vLLM
- Overflow profile (`resume_overflow`) is opt-in and less validated than pure resume

## Benchmarks

Tier-1 Marco facts is a **smoke/regression** harness, not a published leaderboard.
OpenCode harness requires live GPU + stable vLLM; garbled-output detection is
best-effort via `pri.text_quality`.

## Legal

Provisional patent 64/050,345 covers the method. Community license TBD before
public release. Commercial use requires separate license (counsel pending).
