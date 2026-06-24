# Limitations (v0.1)

## Product scope

- **KV capture/resume only** — not text compression, not MoE routing, not hosted SaaS
- **Single-node** — one vLLM process per container; no distributed KV transfer
- **BYOC** — you provide and license the base model checkpoint

## Not included in this repository

MoE router bias, CAMM, streaming-scorer hot-swap, and related legacy paths are **out of scope**. The default image runs KV capture/resume only.

## Agent middleware

`AgentShimMiddleware` derives turn index from client user-message count (no external database). Clients may send full `kv_transfer_params` in the request body or rely on agent shim to inject them.

`memory_capture_start` is required for `rope_start > 0` in capture manifests. See [Client contract](CLIENT_CONTRACT.md) and `bench/opencode/nls_kvp_helpers.py` for harness parity.

## Memory store

- Index is JSONL on disk under `NLS_MEMORY_DIR` — no replication or HA
- Semantic retrieval requires `sentence-transformers` (installed at container start)
- Very large indexes (>250k entries) need tuning via `NLS_MAX_MEMORIES`

## Resume inject

- Resume walks `base_session_id` chain blocks — cross-chain retrieval does not apply
- RoPE re-rotation assumes Qwen hybrid head layout on pinned vLLM
- Overflow profile (`resume_overflow`) is opt-in and less validated than pure resume
- **Long-chain cliff (2026-06-24 bench):** at ~17–23k inject tokens, RESUME recall degrades (cp60 **3/5**, cp80 **0/5**) while full inline TEXT stays **5/5**. RoPE pack geometry audit passes at 100% delta_uniformity — failure mode is inject-mediated decode, not manifest geometry. See [Benchmarks — Turn sweep](BENCHMARKS.md#turn-sweep-length-scaling) and [research — Turn sweep](../bench/results/overnight_20260624_003614/research/06_turn_sweep_scaling.md).

## Benchmarks

Harnesses are **regression and value-case tools**, not a competitive leaderboard. Published proof: [Benchmarks](BENCHMARKS.md).

## Legal

Licensed under [PolyForm Noncommercial License 1.0.0](../LICENSE). See [Licensing](LICENSING.md).

U.S. Provisional Patent Application No. 64/050,345. Commercial deployment requires a separate license — open a GitHub issue with label `licensing`.
