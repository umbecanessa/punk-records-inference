# 8 — Failure modes & garble root cause

## Executive summary

| Category | Status |
|----------|--------|
| Harness bugs (OpenRouter, garbled guard) | **Fixed** in postfix run |
| RoPE geometry | **Pass** at 100% delta_uniformity |
| Long-chain RESUME recall | **Open product issue** at cp60+ |

## Documented limitations

- Turn sweep cp60+: RESUME garbled decode while TEXT 5/5 (~17–23k inject tokens) — inject-mediated, not RoPE geometry.
- Facts-only inject (max_blocks=3) still garbles at cp60+ — not tail-noise text pollution alone.
- OpenRouter TEXT requires reasoning.effort=none (see openrouter_client.py).
- Long-chain RESUME cliff is a product limitation until inject/decode fix lands.

## Garble investigation highlights

From `turn_sweep_cp60_80_garble_inv_garble_cause_cp60.json`:

1. **TEXT 5/5 + RESUME fail** → failure is inject-mediated decode, not absent inline facts.
2. **Facts-only inject (`max_blocks=3`)** still garbles — not tail-noise text pollution alone.
3. **Hotel probe** fails isolated and after probes 1–3 at ~17–18k inject tokens.
4. **21 neutral-substitute blocks** in tail; TEXT still 5/5 with same inline history.

## Session isolation

Each harness uses unique `memory_user` / `memory_base_session`. No cross-test `.nls` bleed.

## Pre-fix lessons (now in canonical artifacts)

| Topic | Resolution |
|-------|------------|
| OpenRouter TEXT with reasoning enabled | Distorts baseline — use `reasoning.effort=none` |
| Unguarded garbled plant turns | Fixed with garbled-turn capture guard in sweep |

Post-fix canonical artifacts: [`canonical_artifacts.json`](../canonical_artifacts.json)

## Recommended research follow-ups

1. Profile vLLM decode under long KV inject (cp60+).
2. MoE model matrix — same harness, compare cliff location.
3. Optional: GPU profiler + energy per recall arm.
