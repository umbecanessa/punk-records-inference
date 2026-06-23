# Client contract

Punk Records Inference extends the OpenAI chat completions API with
`kv_transfer_params` on each request. The agent middleware (`NLS_AGENT_SHIM=1`)
can populate these automatically for tool-calling clients; direct callers set them
explicitly.

## Core fields

| Field | Required | Description |
|-------|----------|-------------|
| `memory_user` | yes | Partition key for memory index and prefix cache |
| `memory_session` | yes | Unique block id for this capture (per turn in agent mode) |
| `memory_base_session` | agent | Stable chain id for resume inject scope |
| `memory_text` | capture | Text label stored with the block (BM25/embedding) |
| `memory_ring` | no | Memory ring (`general` default) |
| `memory_block_role` | no | `user`, `tool`, or `turn` |

## Capture boundary (agent mode)

| Field | Description |
|-------|-------------|
| `memory_capture_start` | Token offset where user content begins (after system+tools) |
| `memory_sys_prompt_hash` | SHA256(system prompt)[:16] — dedup genesis block |
| `memory_capture_end` | Optional upper bound (dual-emit leg mode only) |

Without `memory_capture_start`, captured `.nls` blocks include the full system
preamble and pollute future retrieval.

## Chain resume (turn ≥ 2)

| Field | Description |
|-------|-------------|
| `memory_turn_index` | 1-based user turn counter |
| `memory_prev_hash` | Prior block content hash (chain linking) |
| `memory_inject_mode` | `resume` or `resume_overflow` |
| `memory_silo` | `1` on turn 1 — skip cross-session retrieval |
| `memory_deltanet_init_session` | Prior turn session for Mamba seeding |

## Suppression flags

| Field | Effect |
|-------|--------|
| `memory_off` | No retrieval, no inject |
| `memory_no_capture` | No new `.nls` write |
| `memory_no_retrieval` | Skip Swiss retrieval only |

## Response metadata

When using hosted Punk Records API, responses may include an `nls` field with
retrieval stats. Direct vLLM calls expose `/admin/memory/*` for debug.

## Example (manual resume turn)

```json
{
  "model": "/model",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "What's my dog's name?"}
  ],
  "kv_transfer_params": {
    "memory_user": "user_abc",
    "memory_base_session": "oc_chain123",
    "memory_inject_mode": "resume",
    "memory_no_capture": "1"
  }
}
```

With `NLS_AGENT_SHIM=1`, OpenCode-style requests with tools get strip +
`capture_start` automatically — no NestJS proxy required.
