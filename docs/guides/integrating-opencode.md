# Integrating OpenCode-style agents

How to wire tool-calling agent clients to Punk Records Inference for turn capture and chain resume.

PRI supports two integration paths:

1. **Agent shim (recommended)** — send normal OpenAI chat requests; middleware injects `kv_transfer_params`
2. **Explicit KVP** — client sets `kv_transfer_params` on every request (production Nest proxy pattern)

---

## Architecture

```
OpenCode / custom agent client
        │  POST /v1/chat/completions
        │  (tools, full transcript each turn)
        ▼
AgentShimMiddleware
        │  strip prior assistant noise
        │  memory_capture_start, memory_sys_prompt_hash
        │  memory_base_session, memory_turn_index, chain link
        ▼
NLSSnapshotConnector
        │  WRITE .nls after decode
        │  READ resume inject on turn ≥ 2
        ▼
Model generates with prior KV already loaded
```

With `NLS_AGENT_SHIM=1` (default), you do **not** need a separate proxy service.

---

## Quick integration (agent shim)

### 1. Start PRI with defaults

```bash
export MODEL_PATH=/path/to/qwen35-fp8
docker compose -f docker/compose.yaml up --build
```

Confirm:

```bash
curl -s http://127.0.0.1:8000/v1/models
```

### 2. Point your agent at the vLLM API

```bash
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=unused   # vLLM does not require a key locally
```

Use a Qwen3-compatible tool-call parser on the server side (already set in `docker/start.sh`: `--tool-call-parser qwen3_coder`).

### 3. Send stable chain identity

The shim derives chain metadata from headers and transcript. For deterministic bench parity, send:

| Header / field | Purpose |
|----------------|---------|
| `x-nls-chain-id` or body hint | Stable `memory_base_session` |
| Consistent `memory_user` | Partition key (via KVP if not using shim defaults) |

In practice, the harness uses a generated chain id per session and resends the full transcript each HTTP call — same as OpenCode.

---

## Explicit `kv_transfer_params` (production pattern)

When not using the shim, or when mirroring hosted Punk Records, set KVP on each request. Critical fields for correct capture geometry (KL #648):

| Field | Turn 1 | Turn ≥ 2 |
|-------|--------|----------|
| `memory_user` | ✓ | ✓ |
| `memory_session` | new UUID per turn | new UUID per turn |
| `memory_base_session` | chain id | same chain id |
| `memory_turn_index` | `1` | increment |
| `memory_capture_start` | token offset after system | same |
| `memory_sys_prompt_hash` | SHA256(system)[:16] | same |
| `memory_inject_mode` | — | `resume` |
| `memory_silo` | `1` | — |
| `memory_prev_hash` | — | prior block hash |

Helper module used by benches: `bench/opencode/nls_kvp_helpers.py`

```python
from nls_kvp_helpers import enrich_kv_params, compute_capture_start

kvp = enrich_kv_params(
    {"memory_user": "bench_user", "memory_base_session": chain_id},
    system_prompt=SYSTEM_PROMPT,
    api_root="http://127.0.0.1:8000",
    model="/model",
)
```

Full reference: [Client contract](../CLIENT_CONTRACT.md)

---

## Run the reference harness

The OpenCode long-session harness simulates a multi-turn agent session with planted facts and recall probes:

```bash
./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000 --seed 42
```

Output: `bench/results/opencode_long_session.json`

What it does:

1. **Plant** — seed random stack facts (ports, DB names) the model must remember
2. **Work turns** — scaffold, API layout, env draft, etc. (full transcript resend)
3. **Recall probes** — ask for exact planted values after many turns

Success: exact weird values recalled (not generic guesses like port 3000).

### Manifest proof

Verify turn-2 captures have `rope_start > 0`:

```bash
python bench/opencode/manifest_proof.py --base-url http://127.0.0.1:8000
```

---

## Inject modes for agents

| Mode | When to use |
|------|-------------|
| `resume` | **Default v0.1** — short-to-medium sessions; GX10 proof: 5/5 recall, ~3.7k tokens saved vs TEXT on long12 ([Benchmarks](../BENCHMARKS.md)) |
| `resume_overflow` | Long sessions where context trim evicts tokens; adds Swiss backfill — same recall as resume on short/long12; does not fix cp60+ cliff |

Set container env before start:

```bash
export NLS_API_INJECT_MODE=resume_overflow
docker compose -f docker/compose.yaml up --build
```

Or per-request: `memory_inject_mode=resume_overflow` in KVP.

See [Core concepts](../getting-started/concepts.md) for capture vs resume vs overflow.

---

## Hosted Punk Records API (optional)

The same harness can target the hosted API instead of direct vLLM:

```bash
export PUNK_API_KEY=nls_live_...
python bench/opencode/opencode_long_session_harness.py
```

Direct vLLM is the open-source path; hosted API adds DB-backed chain ids and compaction detection.

---

## Common pitfalls

| Issue | Cause | Fix |
|-------|-------|-----|
| `rope_start=0` in manifests | Missing `memory_capture_start` | Use shim or `nls_kvp_helpers.enrich_kv_params` |
| Resume recall degrades at long sessions | Pure resume window exceeded (~17k+ inject tok) | Documented cliff — see [Limitations](../LIMITATIONS.md#resume-inject); try `resume_overflow` for trim scenarios |
| Garbled captures pollute chain | Bad decode captured | `NLS_TURN_STRIP_GARBLED_DECODE=1` (default) |
| Cross-session bleed on turn 1 | Missing silo | Set `memory_silo=1` on turn 1 |
| System preamble in captures | No strip / wrong capture_start | Enable agent shim |

More: [Troubleshooting](troubleshooting.md)

---

## Next steps

- [Client contract](../CLIENT_CONTRACT.md) — full KVP field list
- [Benchmarks](../BENCHMARKS.md) — reproduction commands and artifacts
- [Environment variables](../reference/env-vars.md) — container tuning
