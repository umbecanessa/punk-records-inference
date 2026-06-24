# Benchmarks

Reproduce proof artifacts with one command against a live vLLM instance.

**Full measurement matrix:** [BENCH_DATA_PLAN.md](BENCH_DATA_PLAN.md) (maintainer doc — Phases A–E for inject-mode default, value case, storage, latency).

---

## Publication results

**GX10 · Qwen3.5-35B-A3B-FP8 · 2026-06-24** — full proof run
[`bench/results/overnight_20260624_003614/`](../bench/results/overnight_20260624_003614/)

Summary table: [`PHASE_E_SUMMARY.md`](../bench/results/overnight_20260624_003614/PHASE_E_SUMMARY.md) ·
machine-readable: [`phase_e_summary.json`](../bench/results/overnight_20260624_003614/phase_e_summary.json) ·
canonical paths: [`canonical_artifacts.json`](../bench/results/overnight_20260624_003614/canonical_artifacts.json)

**Recommended default inject mode:** `resume` (same recall as `resume_overflow` on short/long12; overflow does not recover the cp60+ turn-sweep cliff).

### Phase E headline (long12 chain, local PRI)

| Metric | TEXT | RESUME | OVERFLOW | Δ vs TEXT (prompt tok) |
|--------|------|--------|----------|-------------------------|
| Recall @5 | 5/5 | 5/5 | 5/5 | — |
| Mean prompt tok | 3743 | 42 | 42 | **3701 saved** |
| Mean latency ms | 2885 | 1549 | 1473 | — |

### Inject mode compare

| Scenario | TEXT | RESUME | OVERFLOW | Δ prompt (mean) |
|----------|------|--------|----------|-----------------|
| Short chain (local) | 5/5 | 5/5 | 5/5 | 404 |
| Long12 chain (local) | 5/5 | 5/5 | 5/5 | 3701 |
| Short OpenRouter TEXT | 5/5 | 5/5 | 5/5 | 360 |
| Long12 OpenRouter TEXT | 5/5 | 5/5 | 5/5 | 3701 |
| Long12 `resume_max_tokens=4096` | 5/5 | 5/5 | 5/5 | 3701 |

### Tier-1 Marco · OpenCode

| Bench | TEXT / baseline | RESUME / PRI |
|-------|-----------------|--------------|
| Marco local | 5/5 | 5/5 |
| Marco OpenRouter | 5/5 | 5/5 |
| OpenCode long session (seed 42) | 6/6 (memory_off) | 6/6 |

### Turn sweep (length scaling)

Marco facts + cumulative noise at cp 20/40/60/80. TEXT = full inline; RESUME = chain inject; ARM-D = `resume_overflow`.

| cp | inject tok | TEXT | RESUME | ARM-D |
|----|------------|------|--------|-------|
| 20 | 6225 | 5/5 | 5/5 | 5/5 |
| 40 | 11981 | 5/5 | 5/5 | 5/5 |
| 60 | 17131 | 5/5 | 3/5 | 3/5 |
| 80 | 23543 | 5/5 | 0/5 | 0/5 |

RoPE pack geometry audit: **100%** delta_uniformity (83 blocks, verdict `pass`) — garble at cp60+ is inject/decode, not pack geometry. See [Limitations](LIMITATIONS.md#resume-inject).

### Storage (full turn-sweep session)

| Metric | Value |
|--------|-------|
| Captures | 143 files |
| Capture disk | 648 MB |
| Index rows | 143 |

### Phase completion (BENCH_DATA_PLAN)

| Phase | Status |
|-------|--------|
| A — Inject mode default | ✅ Complete |
| B — Standard vs PRI | ✅ Complete |
| C — Storage snapshot | ✅ Complete |
| D — Latency breakdown | Optional (prompt/latency in JSON per run) |
| E — Publication summary | ✅ Complete |

Maintainer matrix: [BENCH_DATA_PLAN.md](BENCH_DATA_PLAN.md)

### Research analysis (extended)

Full publication pages with Mermaid charts and per-probe tables:

**[`bench/results/overnight_20260624_003614/research/README.md`](../bench/results/overnight_20260624_003614/research/README.md)**

| Page | Topics |
|------|--------|
| [Token efficiency](../bench/results/overnight_20260624_003614/research/02_token_efficiency.md) | Prompt tokens saved, per-scenario, turn-sweep scaling |
| [Latency](../bench/results/overnight_20260624_003614/research/03_latency_analysis.md) | Mean/p95 ms, ms per 1k prompt tokens |
| [Computational cost](../bench/results/overnight_20260624_003614/research/04_computational_cost.md) | Prefill proxy units, cost vs correctness |
| [Energy & cost](../bench/results/overnight_20260624_003614/research/09_energy_and_cost.md) | GPU Wh, electricity $, cloud API $, annual projections |
| [Storage](../bench/results/overnight_20260624_003614/research/05_storage_footprint.md) | `.nls` sizes, MB/turn, bytes/token |
| [Turn-sweep scaling](../bench/results/overnight_20260624_003614/research/06_turn_sweep_scaling.md) | Recall cliff, inject depth charts |
| [RoPE geometry](../bench/results/overnight_20260624_003614/research/07_rope_geometry.md) | delta_uniformity KPI, post-fix audit |
| [Failure modes](../bench/results/overnight_20260624_003614/research/08_failure_modes.md) | Garble root cause, isolation |

Machine-readable rollup: [`research_data.json`](../bench/results/overnight_20260624_003614/research/research_data.json)

Regenerate after a new run:

```bash
python bench/build_phase_e_summary.py --run-dir bench/results/<run_folder>
python bench/build_research_reports.py --run-dir bench/results/<run_folder>
```

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

## Inject mode compare

`bench/tier1/inject_mode_compare.py` — plants Marco facts (+ optional noise turns), then scores
TEXT vs RESUME vs RESUME_OVERFLOW on five recall probes.

```bash
./bench/run_suite.sh --tier mode-compare --base-url http://127.0.0.1:8000 --seed 42

# Long chain stress (12 noise turns)
NOISE_TURNS=12 ./bench/run_suite.sh --tier mode-compare --base-url http://127.0.0.1:8000 --seed 42

# OpenRouter TEXT baseline (requires bench/.env — see bench/env.example)
python bench/tier1/inject_mode_compare.py --text-backend openrouter ...
```

## Proof artifacts (GX10, stock Qwen3.5-35B-A3B-FP8)

Canonical run folder: `bench/results/overnight_20260624_003614/` (see `canonical_artifacts.json`).

| Date | Bench | Result | Artifact |
|------|-------|--------|----------|
| 2026-06-24 | inject mode compare (short + long12) | 5/5/5 all arms | `inject_mode_compare_*_postfix.json` |
| 2026-06-24 | inject mode compare (OpenRouter TEXT) | 5/5/5 | `inject_mode_compare_*_openrouter_reasoning_none.json` |
| 2026-06-24 | tier1 marco_facts | TEXT 5/5, RESUME 5/5 | `tier1_marco_facts_*_marco.json` |
| 2026-06-24 | opencode long session (seed 42) | PRI 6/6, baseline 6/6 | `opencode_long_session_*.json` |
| 2026-06-24 | turn sweep cp20–80 (v5) | cp20–40: 5/5; cp60: 3/5; cp80: 0/5 RESUME | `turn_sweep_cp20_80_v5.json` |
| 2026-06-24 | geometry audit (RoPE fix) | pass, 100% delta_uniformity | `geometry_audit_turn_sweep_v5_fixed.json` |
| 2026-06-23 | smoke marco_facts (seed 42) | TEXT 5/5, RESUME 5/5 | `bench/results/tier1_marco_facts_42.json` |

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

Success criterion: `resume_pass >= text_pass` at cp20–40 (value case proven); cp60+
RESUME garble while TEXT stays 5/5 is a **documented product limitation** (~17–23k
inject tokens), not a bench failure. RoPE geometry audit passes at 100%.

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

## Parity assumption test (attn vs SSM)

`bench/tier1/resume_parity_assumption_test.py` — on a **frozen** turn-sweep chain,
measures query-token parity for `attn_input_hs`, `ssm_state`, and `deltanet_out_hs`,
plus inject-scope and decode-stability A/B (prefix vs full, isolated vs sequential hotel).

Requires the chain still on disk and microscope support in vLLM. Run inside the
container (torch + `/tmp/nls_microscope`):

```bash
./bench/run_parity_assumption_test.sh \
  bench/results/overnight_20260624_003614/turn_sweep_cp60_80_garble_inv.json 80
```

For accurate `full_vs_resume`, export turns during sweep:

```bash
python bench/tier1/turn_sweep.py ... --export-turns bench/results/turns_cp80.json
TURNS_JSON=bench/results/turns_cp80.json ./bench/run_parity_assumption_test.sh ...
```

Output: `*_parity_assumption_cp80.json` with `assumption_verdicts` for H1–H6.

## Unit tests (no GPU)

```bash
pip install pytest torch zstandard
pytest tests/ -q
```

Covers `.nls` round-trip, manifest validation, agent shim helpers.
