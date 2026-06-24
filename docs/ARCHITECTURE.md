# Architecture

Punk Records Inference is a **KV-state persistence layer** for vLLM. One container,
one process: OpenAI-compatible API on port 8000, plugin loaded via `--kv-transfer-config`.

For the full multi-phase pipeline narrative (retrieval, scoring, monitoring), see the companion [**Neural Ledger System**](https://github.com/umbecanessa/neural-ledger-system) architecture docs. This repo implements the **reference code** for capture, chain resume, and optional Swiss/overflow — see [Overview](OVERVIEW.md) for how the two relate.

## Runtime flow

```
Client (OpenCode, curl, LangChain)
        │
        ▼
┌───────────────────────────────────────────┐
│  vLLM OpenAI API :8000                     │
│  ┌─────────────────────────────────────┐  │
│  │ AgentShimMiddleware                  │  │
│  │  strip transcript · capture_start      │  │
│  │  chain_id · turn_index · silo        │  │
│  └─────────────────────────────────────┘  │
│  ┌─────────────────────────────────────┐  │
│  │ NLSSnapshotConnector (pri.connector)│  │
│  │  WRITE: turn capture → .nls          │  │
│  │  READ:  resume | swiss | overflow   │  │
│  └─────────────────────────────────────┘  │
└───────────────────────────────────────────┘
        │
        ▼
/data/pri  (captures/*.nls + index.jsonl)
        │
        ▼
/model  (BYOC checkpoint — not baked in)
```

## Subsystems (orthogonal)

| Subsystem | Direction | Config | v0.1 default |
|-----------|-----------|--------|--------------|
| **Capture** | Write `.nls` per turn | `NLS_CHAIN_CAPTURE_MODE=turn` | On |
| **Resume** | Inject prior turn KV | `memory_inject_mode=resume`, turn ≥ 2 | On |
| **Swiss** | Semantic retrieval | auto when not resume/silo | Off (bench profile) |
| **Overflow** | Resume + Swiss compose | `memory_inject_mode=resume_overflow` | Opt-in |

Capture ≠ resume ≠ overflow. Do not conflate them in client or bench code.

## Package layout (`pri/`)

| Module | Role |
|--------|------|
| `connector.py` | vLLM KV connector — inject + capture |
| `capture.py` | Turn vs dual capture mode |
| `resume.py` | Chain block collection + resume inject config |
| `store.py` | On-disk memory index |
| `retrieve.py` | Swiss retrieval (optional) |
| `format.py` | `.nls` read/write |
| `scorer.py` | Neural inject V-suppression |
| `admin.py` | `/admin/memory/*` middleware |
| `middleware/agent_shim.py` | Agent transcript strip + KVP enrichment |

## Mapping to NLS pipeline phases

| [NLS phase](https://github.com/umbecanessa/neural-ledger-system/blob/main/ARCHITECTURE.md) | PRI module | v0.1 |
|-----|-----|-----|
| Capture | `connector.py`, `capture.py`, `format.py` | ✅ Default |
| Injection (chain resume) | `resume.py`, `connector.py` | ✅ Default |
| Injection (RoPE correction) | `resume.py` | ✅ Validated |
| Retrieval (Swiss/BM25) | `retrieve.py`, `store.py` | Opt-in |
| Scoring (V-suppression) | `scorer.py` | Opt-in |
| Monitoring (hot-swap) | — | Not shipped |

`pri/` is the sole Python package. vLLM loads middleware and the KV connector via
module paths in `docker/start.sh` (`pri.middleware.agent_shim`, `pri.admin`, `pri.connector`).
