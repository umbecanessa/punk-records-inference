# Architecture

> Stub — see [SHIP_PLAN.md](SHIP_PLAN.md) §3–§4 for the canonical diagram.

## Subsystems (orthogonal)

| Subsystem | Direction | v0.1 default |
|-----------|-----------|--------------|
| **Capture** | Write `.nls` per turn | On (`NLS_CHAIN_CAPTURE_MODE=turn`) |
| **Resume** | Read prior turn KV | On (turn ≥ 2) |
| **Swiss** | Semantic retrieval | Off (benchmark profile) |
| **Overflow** | Resume + Swiss compose | Opt-in |

Capture ≠ resume ≠ overflow. Do not conflate them in client or bench code.
