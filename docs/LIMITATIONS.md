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
- **Long-chain cliff (GX10, 2026-06-24):** at ~17–23k inject tokens, RESUME recall
  degrades (cp60 **3/5**, cp80 **0/5**) while full inline TEXT stays **5/5**. RoPE pack
  geometry audit passes at 100% delta_uniformity — failure mode is inject-mediated decode,
  not manifest geometry. Facts-only inject (`max_blocks=3`) still garbles. See
  [`PHASE_E_SUMMARY.md`](../bench/results/overnight_20260624_003614/PHASE_E_SUMMARY.md).

## Benchmarks

Tier-1 Marco facts and OpenCode harnesses are **regression + value-case** tools, not a
published leaderboard. GX10 proof run (2026-06-24): inject compare, Marco, OpenCode,
turn sweep, and geometry audit — see [Benchmarks](BENCHMARKS.md).

## Legal

This project is licensed under the [Punk Records Community License 1.0](../LICENSE) (PRC-1.0). See [Licensing](LICENSING.md) for what is free vs what requires a commercial agreement.

U.S. Provisional Patent Application No. 64/050,345 describes related methods. No broad patent license is granted for commercial use — contact via GitHub issue with label `licensing`.
