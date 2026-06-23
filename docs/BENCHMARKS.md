# Benchmarks

Reproduce proof artifacts with one command against a live vLLM instance.

**Full measurement matrix:** [BENCH_DATA_PLAN.md](BENCH_DATA_PLAN.md) (maintainer doc — Phases A–E for inject-mode default, value case, storage, latency).

---

## Publication results

Proof tables in this doc and the [README](../README.md) are updated when bench sweeps complete on GX10. Current artifacts below are **smoke/regression** baselines, not a final publication set.

| Status | Bench | Notes |
|--------|-------|-------|
| ✅ Green | Tier-1 Marco facts | TEXT 5/5 · RESUME 5/5 |
| ✅ Green | OpenCode long session (seed 42) | RECALL 6/6 |
| 🔄 In progress | Turn sweep cp20–80 (clean) | Inject-mode + garbled-capture hygiene |
| 🔄 In progress | Inject mode compare | `resume` vs `resume_overflow` value case |
| 📋 Planned | Token/latency/disk matrix | See BENCH_DATA_PLAN Phase B–D |

When the bench pass lands, this section gets expanded tables (tokens saved, latency p50/p95, disk per turn) linked to JSON artifacts under `bench/results/`.

---

## Prerequisites

- Running Punk Records Inference container (see [DOCKER.md](DOCKER.md))
- GPU with hybrid Qwen checkpoint mounted
- Python 3.10+ with `requests` on the host (for bench scripts)

## Run suite

```bash
# Tier 1: health + Marco facts TEXT vs RESUME
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000

# OpenCode long-session harness (direct vLLM)
./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000

# Turn sweep cp20-80 (production-length session)
./bench/run_suite.sh --tier sweep --base-url http://127.0.0.1:8000

# Geometry audit (after sweep — reads chain from admin API)
./bench/run_suite.sh --tier geometry --base-url http://127.0.0.1:8000 \
  --sweep-json bench/results/turn_sweep_cp20_80.json
```

Results land in `bench/results/`.

## Tier 1 — Marco facts

`bench/tier1/marco_facts.py` plants three identity/noise turns, then scores recall
on five probes under two arms:

| Arm | Behavior |
|-----|----------|
| **text** | Full inline history, `memory_off=1` |
| **resume** | Latest message only, `memory_inject_mode=resume` |

Success criterion (v0.1): `resume_pass >= text_pass` on cp3 plant.

```bash
python bench/tier1/marco_facts.py \
  --base-url http://127.0.0.1:8000 \
  --seed 42 \
  --out bench/results/tier1_marco_facts_42.json
```

## Proof artifacts (GX10, stock Qwen3.5-35B-A3B-FP8)

| Date | Bench | Model | Result | Artifact |
|------|-------|-------|--------|----------|
| 2026-06-23 | tier1 marco_facts (seed 42) | `/model` | TEXT 5/5, RESUME 5/5 | `bench/results/tier1_marco_facts_42.json` |
| 2026-06-23 | opencode long session (seed 42) | `/model` | RECALL 6/6 | `bench/results/opencode_long_session.json` |
| 2026-06-23 | turn sweep cp20–80 | `/model` | cp20–40: 5/5; cp60: 4/5; cp80: 1/5 RESUME | `bench/results/turn_sweep_cp20_80.json` |
| 2026-06-23 | turn sweep diagnose | `/model` | see diagnose JSON | `bench/results/turn_sweep_cp20_80_diagnose.json` |

## OpenCode harness

`bench/opencode/` — multi-turn agent recall after 8+ turns. For direct vLLM:

```bash
./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000 --seed 42
```

For hosted Punk Records API, set `PUNK_API_KEY` instead of `--base-url`.

### Manifest proof (KL #648)

Verifies turn-2 capture manifests have `rope_start > 0` when the client sends
`memory_capture_start` (production Nest proxy pattern):

```bash
python bench/opencode/manifest_proof.py --base-url http://127.0.0.1:8000
```

Last known good (NLS monorepo): chain `oc_4eab7d5da27584e2` — RECALL 4/6 strict,
functionally 6/6. PRI repo (GX10 direct vLLM): chain `long_76c9e092e24b` — 6/6.

## Turn sweep (cp20–80)

`bench/tier1/turn_sweep.py` — Marco facts + cumulative noise at checkpoints
20/40/60/80. Scores TEXT vs RESUME (and optional `resume_overflow` arm D) on five
recall probes at each checkpoint. Requires `NLS_CHAIN_CAPTURE_MODE=turn`.

```bash
python bench/tier1/turn_sweep.py \
  --base-url http://127.0.0.1:8000 \
  --checkpoints 20,40,60,80 \
  --out bench/results/turn_sweep_cp20_80.json
```

Success criterion: `resume_pass_clean >= text_pass_clean` at cp20–60; cp80 cliff
documented (NLS post-fix: 4/5 resume at ~22k inject tokens).

## Geometry audit

`bench/tier1/geometry_audit.py` — offline RoPE pack + Mamba provenance audit for
a turn chain. Uses admin API or local `NLS_MEMORY_DIR` index.

```bash
python bench/tier1/geometry_audit.py \
  --from-sweep bench/results/turn_sweep_cp20_80.json \
  --base-url http://127.0.0.1:8000 \
  --out bench/results/geometry_audit_turn_sweep.json
```

Verdict `pass` means inject geometry is consistent for resume pack plan.

## Unit tests (no GPU)

```bash
pip install pytest torch zstandard
pytest tests/ -q
```

Covers `.nls` round-trip, manifest validation, agent shim helpers.
