# Core concepts

Punk Records Inference (PRI) persists **model internal state** across agent turns — not chat text. Each turn's KV cache is captured to disk; the next request re-injects that state so vLLM skips re-prefilling unchanged history.

---

## The problem

Long agent sessions accumulate context. Every new turn typically **re-prefills** the entire message history through the transformer. That costs latency, GPU memory, and tokens — even when most of the history is unchanged.

PRI captures **attention KV state** (and hybrid recurrent state where applicable) after each turn, stores it in a compact `.nls` file, and **re-injects** it on the next request.

---

## Subsystems

These are **separate mechanisms**. Do not conflate them in client code or benchmarks.

| Subsystem | Direction | v0.1 default | Purpose |
|-----------|-----------|--------------|---------|
| **Capture** | Write | On | Save turn state → `.nls` on disk |
| **Resume** | Read | On (turn ≥ 2) | Inject prior turn KV; skip re-prefill |
| **Swiss** | Read | Off | Semantic retrieval when resume alone is insufficient |
| **Overflow** | Read | Opt-in | Resume + Swiss when trim evicts tokens |

### Capture

After generation completes, the connector serializes KV blocks and hybrid state into a `.nls` manifest under `/data/pri/captures/`.

**Agent mode:** `AgentShimMiddleware` strips the transcript to user/tool content and sets `memory_capture_start` so captures exclude the system preamble (required for correct RoPE alignment).

### Resume

On turn 2+, set `memory_inject_mode=resume` and `memory_base_session` to the chain id. The connector loads prior `.nls` blocks and injects KV — prior context is **already computed**, not re-tokenized.

**Default inject mode in v0.1:** `resume` (validated on short-to-medium sessions; see [Benchmarks](../BENCHMARKS.md)).

### Swiss retrieval

When resume cannot cover evicted or cross-session context, **Swiss** ranks stored memories by embedding + BM25. Off by default in v0.1.

### Overflow (`resume_overflow`)

Combines resume with Swiss when trim evicts tokens from the inject window. Opt-in profile — not the default.

---

## Chain metadata

| Field | Role |
|-------|------|
| `memory_user` | Partition key (tenant/user) |
| `memory_base_session` | Stable chain id for resume scope |
| `memory_session` | Unique block id per turn |
| `memory_turn_index` | 1-based user turn counter |
| `memory_prev_hash` | Content hash linking to prior block |

Turn 1 uses `memory_silo=1` to skip cross-session retrieval.

Full reference: [Client contract](../CLIENT_CONTRACT.md).

---

## `.nls` files

Each capture is a compressed manifest (zstd) with KV tensor shapes, RoPE metadata (`rope_start`, etc.), hybrid recurrent state, and a text label for retrieval indexing.

Schema: [`spec/`](../../spec/).

---

## What PRI is not

| Not PRI | Notes |
|---------|-------|
| Text/context compression | PRI optimizes GPU re-prefill, not prompt size |
| Long-term agent memory (facts, profiles) | Out of scope — use external memory products |
| MoE expert routing | Out of scope |
| Hosted inference SaaS | This repo is BYOC self-host only |

See [Limitations](../LIMITATIONS.md).
