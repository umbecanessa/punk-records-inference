# Environment variables

Consolidated reference for `NLS_*` environment variables used in v0.1.

**Inject profiles** are selected via `NLS_API_INJECT_MODE` and applied at startup by `pri/startup_profile.py`. See [Core concepts](../getting-started/concepts.md).

---

## Required

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *(required)* | Checkpoint directory (Docker: `/model`) |

---

## Paths and storage

| Variable | Default | Description |
|----------|---------|-------------|
| `NLS_MODEL_PATH` | `$MODEL_PATH` | Model path used by admin and retrieve |
| `NLS_MEMORY_DIR` | `/data/pri` | Memory store root (index + captures) |
| `NLS_SNAPSHOT_DIR` | `/data/pri/snapshot` | KV connector snapshot directory |
| `NLS_GENESIS_PATH` | — | Optional path to genesis block template |

---

## Inject mode and profiles

Set **`NLS_API_INJECT_MODE`** before container start. `startup_profile.py` writes derived vars to `$NLS_MEMORY_DIR/profile.env`.

| Value | Use case | Neural / Swiss |
|-------|----------|----------------|
| `resume` | **v0.1 default** — chain inject only | Off |
| `resume_overflow` | Resume + Swiss when trim evicts | On |
| `swiss` | Legacy pool retrieval primary | On |

| Variable | Default (`resume`) | Description |
|----------|-------------------|-------------|
| `NLS_API_INJECT_MODE` | `resume` | Startup inject profile selector |
| `PRI_INJECT_PROFILE` | *(written)* | Cached profile name in `profile.env` |
| `NLS_INJECT_MODE` | `swiss` | Runtime inject mode inside connector (set by profile for `swiss` arm) |
| `NLS_NEURAL_SCORING` | `0` | Enable neural inject scoring |
| `NLS_V_SUPPRESSION` | `0` | V-head suppression during inject |
| `NLS_NEURAL_COARSE_K` | `20` / `10` (overflow) | Coarse retrieval pool size |
| `NLS_NEURAL_FINAL_K` | `5` | Final inject block count |
| `NLS_V_SUPPRESSION_KEEP_K` | `5` | Blocks kept after V-suppression |
| `NLS_NEURAL_SUPPRESS_THRESHOLD` | `0.15` | Suppression score threshold |
| `NLS_META_NEURAL_PENALTY` | `1.0` | Meta-score penalty weight |
| `NLS_NEURAL_SCORE_LAYERS` | *(from topology)* | Comma-separated full-attn layers |
| `NLS_V_SUPPRESSION_AT_LAYER` | *(from topology)* | Layer index for V-suppression |
| `NLS_DELTA_FACT_PROBE_LAYERS` | *(from topology)* | DeltaNet probe layers for capture |
| `NLS_RESUME_SWISS_MAX_TOKENS` | `256` | Max Swiss tokens on overflow path |
| `NLS_RESUME_MAX_BLOCKS` | — | Cap resume chain blocks (agent shim) |
| `NLS_RESUME_MAX_TOKENS` | — | Cap resume inject tokens (agent shim) |
| `NLS_RESUME_ABORT_ON_ROPE_FAIL` | `1` | Abort inject on RoPE pack failure |
| `NLS_RESUME_MAMBA_DELTA_SUM` | `1` | Mamba delta sum on resume pack |
| `NLS_RESUME_ROLES` | `turn,tool` | Block roles included in resume chain |

---

## Capture

| Variable | Default | Description |
|----------|---------|-------------|
| `NLS_CHAIN_CAPTURE_MODE` | `turn` | `turn` = one `.nls` per agent turn |
| `NLS_CAPTURE_MIN_DECODE_TOKENS` | `4` | Minimum decode tokens to capture |
| `NLS_TURN_STRIP_GARBLED_DECODE` | `1` | Skip capture on garbled decode |
| `NLS_CAPTURE_META_MAX` | `0.95` | Meta-token ratio cap for capture |
| `NLS_CAPTURE_MIN_WORDS` | `4` | Minimum words for capture label |
| `NLS_PLUGIN_PASS2` | `1` | Second-pass capture enrichment |
| `NLS_PASS2_TIMEOUT` | `60` | Pass-2 HTTP timeout (seconds) |

---

## Agent middleware

| Variable | Default | Description |
|----------|---------|-------------|
| `NLS_AGENT_SHIM` | `1` | Enable `AgentShimMiddleware` |
| `NLS_STRIP_ASSISTANT_KEEP_RATIO` | `0` | Fraction of assistant text kept on strip |
| `NLS_STRIP_INJECT_SYS_BLOCK_LEN` | `105` | Legacy static sys strip (prefer `memory_capture_start`) |

When `NLS_AGENT_SHIM=1`, OpenCode-style requests get automatic transcript strip and `memory_capture_start`. Clients can also send full `kv_transfer_params` manually — see [Client contract](../CLIENT_CONTRACT.md).

---

## KV inject tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `NLS_KV_K_SCALE` | `1.3` | K tensor scale on inject |
| `NLS_KV_V_SCALE` | `1.0` | V tensor scale on inject |
| `NLS_SNAPSHOT_CACHE_MAX_ENTRIES` | `0` | In-memory snapshot cache entries (0=default) |
| `NLS_SNAPSHOT_CACHE_MAX_BYTES` | `0` | Snapshot cache byte limit |
| `NLS_CUDA_RELEASE_INTERVAL_S` | `5.0` | CUDA cache release interval |
| `NLS_MAMBA_DELTA_SUM` | `0` | Mamba delta sum debug flag |

---

## Retrieval and memory index

| Variable | Default | Description |
|----------|---------|-------------|
| `NLS_MAX_MEMORIES` | `250000` | Max index entries before pruning |
| `NLS_ROLE_FILTER` | `user,tool` | Roles indexed for Swiss retrieval |
| `NLS_SEMANTIC_DEVICE` | `cuda` | Device for sentence-transformers |
| `NLS_RECENCY` | `1` | Recency weighting in retrieval |
| `NLS_RECENCY_DECAY` | `0.002` | Recency decay factor |
| `NLS_RECENCY_FLOOR` | `1.0` | Minimum recency multiplier |
| `NLS_DELTA_FACT` | `1` | Delta-fact probe scoring |
| `NLS_DELTA_FACT_BOOST` | `0.35` | Delta-fact score boost |
| `NLS_DELTA_SIGNAL_ENABLED` | `1` | Delta signal in retrieval |
| `NLS_DELTA_SIGNAL_SHARPNESS` | `15.0` | Delta signal sharpness |
| `NLS_TEMPORAL_INDEX` | `1` | Temporal index for memories |
| `NLS_META_PENALTY_WEIGHT` | `0.85` | Meta penalty in ranking |
| `NLS_COMPACTION_CONTEXT_BOOST` | `0.25` | Boost for compaction-tagged blocks |
| `NLS_TURN_RECENCY_BONUS` | `0.0` | Turn recency bonus |
| `NLS_CHAIN_DECAY` | `0.90` | Chain walk decay |
| `NLS_CHAIN_WALK` | `1` | Enable chain walk retrieval |
| `NLS_CHAIN_WALK_ROLES` | `tool` | Roles for chain walk |
| `NLS_CHAIN_HOPS` | `2` | Chain walk hop depth |
| `NLS_ASST_FUNNEL` | `0` | Assistant funnel retrieval (off v0.1) |
| `NLS_ASST_FUNNEL_TOP_K` | `100` | Assistant funnel top-K |

---

## vLLM server (docker/start.sh)

These are standard vLLM flags, not `NLS_*`, but commonly overridden:

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_MODEL_LEN` | `32768` | Context window |
| `GPU_MEMORY_UTILIZATION` | `0.60` | GPU memory fraction |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Chunked prefill batch size |

---

## Bench harness (host)

| Variable | Description |
|----------|-------------|
| `PRI_BASE_URL` | vLLM base URL (default `http://127.0.0.1:8000`) |
| `PRI_API` | Chat completions URL |
| `NLS_API` | Alias for `PRI_API` |
| `OPENROUTER_API_KEY` | Optional TEXT baseline via OpenRouter |
| `PUNK_API_KEY` | Hosted Punk Records API key (OpenCode harness) |

Copy `bench/env.example` to `bench/.env` for local bench secrets (gitignored).

---

## Per-request overrides (`kv_transfer_params`)

Many behaviors are controlled per request rather than by env vars. Full field list: [Client contract](../CLIENT_CONTRACT.md).

Key request fields: `memory_inject_mode`, `memory_off`, `memory_no_capture`, `memory_capture_start`, `memory_base_session`, `memory_turn_index`.

---

## See also

- [Docker](../DOCKER.md) — compose, volumes, manual run
- [Architecture](../ARCHITECTURE.md) — how subsystems interact
- [Integrating OpenCode](../guides/integrating-opencode.md) — client wiring
