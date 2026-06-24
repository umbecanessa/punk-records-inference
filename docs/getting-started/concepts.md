# Core concepts

Punk Records Inference (PRI) persists **model internal state** across agent turns — not chat text. Think of it as pressing a vinyl record of each turn's KV cache and replaying it on the next request instead of re-reading the full transcript.

---

## The problem

Long agent sessions accumulate context. Every new turn typically **re-prefills** the entire message history through the transformer. That costs latency, GPU memory, and tokens — even when most of the history is unchanged.

PRI captures the **attention KV state** (and hybrid recurrent state where applicable) after each turn, stores it in a compact `.nls` file, and **re-injects** it on the next request so vLLM continues from where the model left off.

---

## Three orthogonal subsystems

These are **separate mechanisms**. Do not conflate them in client code or benchmarks.

| Subsystem | Direction | v0.1 default | Purpose |
|-----------|-----------|--------------|---------|
| **Capture** | Write | On | Save turn state → `.nls` on disk |
| **Resume** | Read | On (turn ≥ 2) | Inject prior turn KV; skip re-prefill |
| **Swiss** | Read | Off | Semantic retrieval when resume alone is insufficient |
| **Overflow** | Read | Opt-in | Resume + Swiss when trim evicts tokens |

### Capture

After generation completes, the connector serializes the relevant KV blocks and hybrid state into a `.nls` manifest under `/data/pri/captures/`. Each block is keyed by `memory_user`, `memory_session`, and chain metadata.

**Agent mode:** `AgentShimMiddleware` strips the transcript to user/tool content and sets `memory_capture_start` so captures exclude the system preamble (critical for correct RoPE alignment).

### Resume

On turn 2+, the client sets `memory_inject_mode=resume` and `memory_base_session` to the chain id. The connector loads prior `.nls` blocks and injects KV into vLLM's cache — the model sees prior context as **already computed**, not as new tokens.

v0.1 ships with **pure resume** as the default inject mode. This minimizes variables and has validated well on short-to-medium agent sessions.

### Swiss retrieval

When resume cannot cover evicted or cross-session context, **Swiss** ranks stored memories by embedding + BM25 and injects selected blocks. This is the research path for very long sessions; v0.1 keeps it off by default.

### Overflow (`resume_overflow`)

Combines resume with Swiss **only when** trim evicts tokens from the inject window, or when explicitly forced. Arm D in the bench matrix — flip to default only after Phase A benchmarks prove no recall regression.

---

## Chain metadata

Agent sessions form a **chain** of linked `.nls` blocks:

| Field | Role |
|-------|------|
| `memory_user` | Partition key (tenant/user) |
| `memory_base_session` | Stable chain id for resume scope |
| `memory_session` | Unique block id per turn |
| `memory_turn_index` | 1-based user turn counter |
| `memory_prev_hash` | Content hash linking to prior block |

Turn 1 uses `memory_silo=1` to skip cross-session retrieval — a fresh chain starts clean.

Full field reference: [Client contract](../CLIENT_CONTRACT.md).

---

## `.nls` files

Each capture is a compressed manifest (zstd) describing:

- KV tensor shapes and layer groups (from vLLM `kv_cache_groups` — model-agnostic at runtime)
- RoPE position metadata (`rope_start`, etc.)
- Hybrid recurrent state (Mamba/DeltaNet legs on Qwen3.5)
- Text label for retrieval indexing

Schema and validator live in [`spec/`](../../spec/).

---

## What PRI is not

| Not PRI | Alternative |
|---------|-------------|
| Text/context compression | See [Headroom](https://github.com/headroomlabs-ai/headroom) |
| Long-term agent memory (facts, soul) | External agent-memory products — out of scope for PRI |
| MoE expert routing / router bias | Legacy NLS research — out of scope |
| Hosted SaaS | BYOC self-host only in this repo |

See [Limitations](../LIMITATIONS.md) for the full v0.1 scope boundary.
