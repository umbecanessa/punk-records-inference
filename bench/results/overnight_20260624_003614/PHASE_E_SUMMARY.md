# Phase E — Bench proof summary

Generated: 2026-06-24T11:18:55.461291+00:00
Run dir: `bench/results/overnight_20260624_003614`
Git SHA: `9944c727779bfe722139b0851ce804149790c7d4`
Default inject mode: **resume**

## Headline (long chain, local PRI)

| Metric | TEXT | RESUME | OVERFLOW | Δ vs TEXT (prompt tok) |
|--------|------|--------|----------|-------------------------|
| Recall @5 | 5/5 | 5/5 | 5/5 | — |
| Mean prompt tok | 3743.2 | 42.2 | 42.2 | 3701.0 |
| Mean latency ms | 2884.9 | 1548.5 | 1472.9 | — |

## Inject mode compare

| Scenario | TEXT | RESUME | OVERFLOW | Δ prompt (mean) |
|----------|------|--------|----------|-----------------|
| short chain (0 noise) (local) | 5/5 | 5/5 | 5/5 | 404.0 |
| long12 chain (local) | 5/5 | 5/5 | 5/5 | 3701.0 |
| short OpenRouter TEXT (openrouter) | 5/5 | 5/5 | 5/5 | 360.0 |
| long12 OpenRouter TEXT (openrouter) | 5/5 | 5/5 | 5/5 | 3701.0 |
| long12 resume_max_tokens=4096 (local) | 5/5 | 5/5 | 5/5 | 3701.0 |

## Tier-1 Marco

| Run | TEXT | RESUME |
|-----|------|--------|
| Marco local (local) | 5/5 | 5/5 |
| Marco OpenRouter (openrouter) | 5/5 | 5/5 |

## OpenCode long session (seed 42)

| Arm | RECALL |
|-----|--------|
| PRI | 6/6 |
| baseline (memory_off) | 6/6 |

## Turn sweep (TEXT / RESUME / ARM-D)

| cp | inject tok | TEXT | RESUME | ARM-D |
|----|------------|------|--------|-------|
| v5_overnight cp20 | 6225 | 5/5 | 5/5 | 5/5 |
| v5_overnight cp40 | 11981 | 5/5 | 5/5 | 5/5 |
| v5_overnight cp60 | 17131 | 5/5 | 3/5 | 3/5 |
| v5_overnight cp80 | 23543 | 5/5 | 0/5 | 0/5 |
| garble_investigation cp60 | 17680 | 5/5 | 2/5 | 1/5 |
| garble_investigation cp80 | 18629 | 5/5 | 0/5 | 0/5 |

## RoPE pack geometry (delta_uniformity KPI)

| Chain | Verdict | Blocks | delta_uniformity | mode Δ |
|-------|---------|--------|------------------|--------|
| turn_sweep v5 (RoPE fix audit) | pass | 83 | 100.0% | -22 |
| garble_inv chain | pass | 83 | 100.0% | -22 |

## Storage (Phase C snapshot)

- Captures: 143 files, **648.11 MB**
- Index rows: 143
- Data dir: 650.83 MB

## Known limitations (documented, not bench failures)

- Turn sweep cp60+: RESUME garbled decode while TEXT 5/5 (~17–23k inject tokens) — inject-mediated, not RoPE geometry.
- Facts-only inject (max_blocks=3) still garbles at cp60+ — not tail-noise text pollution alone.
- OpenRouter TEXT requires reasoning.effort=none (see openrouter_client.py).
- Long-chain RESUME cliff is a product limitation until inject/decode fix lands.

## Canonical artifacts

See `canonical_artifacts.json` in this folder for full paths.

## Research analysis (extended)

Detailed pages with charts: [`research/README.md`](research/README.md)

| Topic | File |
|-------|------|
| Token efficiency | [research/02_token_efficiency.md](research/02_token_efficiency.md) |
| Latency | [research/03_latency_analysis.md](research/03_latency_analysis.md) |
| Computational cost | [research/04_computational_cost.md](research/04_computational_cost.md) |
| Storage | [research/05_storage_footprint.md](research/05_storage_footprint.md) |
| Turn-sweep scaling | [research/06_turn_sweep_scaling.md](research/06_turn_sweep_scaling.md) |
| RoPE geometry | [research/07_rope_geometry.md](research/07_rope_geometry.md) |
| Failure modes | [research/08_failure_modes.md](research/08_failure_modes.md) |
