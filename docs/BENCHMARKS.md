# Benchmarks

Reproduce proof numbers on your own hardware with the harnesses in [`bench/`](../bench/). Published results are from a single validated run (2026-06-24).

**Run overview:** [`bench/results/overnight_20260624_003614/README.md`](../bench/results/overnight_20260624_003614/README.md)  
**Summary tables:** [`BENCHMARK_SUMMARY.md`](../bench/results/overnight_20260624_003614/BENCHMARK_SUMMARY.md)  
**Key findings (narrative):** [`research/00_findings.md`](../bench/results/overnight_20260624_003614/research/00_findings.md)  
**Extended analysis:** [`research/README.md`](../bench/results/overnight_20260624_003614/research/README.md)  
**Architecture context:** [Overview](OVERVIEW.md) · [Documentation map](PLATFORM.md) · [NLS pipeline](NLS_PIPELINE.md)

---

## Historical production validation (April 2026)

Before the June 2026 reproducible OSS run, the full five-phase stack was validated in production behind [Punk Records Demo](https://punkrecords.live) and the OpenCode agent harness.

### OpenCode cross-session recall (2026-04-27)

Real coding agent (OpenCode TUI) → multi-phase scaffold → recall across **full TUI restarts** with zero chat history.

| # | Question | Phantom tokens injected | User prompt tok | Correct? |
|---|----------|------------------------:|----------------:|:--------:|
| Q1 | Backend port from phase 1? | ~10,000 | — | ✓ (3001) |
| Q3 | Frontend port? (cold, post-restart) | **18,751** | **124** | ✓ (3000) |

**99.3% prompt-token savings** on the recall path. **4/4** questions correct including frontend/backend disambiguation.

### LongMemEval parity (April 2026)

| Condition | Qwen 3.5 | Qwen 3.6 |
|-----------|----------|----------|
| TEXT golden | 8/18 | 9/18 |
| KV golden | 8/18 | 9/18 |
| KV + 5 distractors | 8/18 | 9/18 |

**TEXT = KV** at the text ceiling — remaining gap is model capacity, not KV fidelity.

### Conversational production

5/5 recall on standard `prod_conversation_test` (5 facts, fresh session probe).

Full narrative: [JOURNEY.md](JOURNEY.md) · Economics: [ECONOMICS.md](ECONOMICS.md).

---

## Reproducible OSS proof (June 2026)

The section below is the **frozen run** anyone can reproduce with this repo's harnesses and Docker image.

### Validated configuration

| Component | Value |
|-----------|-------|
| Model | Qwen3.5-35B-A3B-FP8 (stock hybrid checkpoint) |
| GPU | NVIDIA, ≥24 GB VRAM |
| Date | 2026-06-24 |
| Default inject mode | `resume` |

---

## Headline results

### Inject mode compare (long12 chain)

| Metric | TEXT | RESUME | OVERFLOW | Δ vs TEXT (prompt tok) |
|--------|------|--------|----------|-------------------------|
| Recall @5 | 5/5 | 5/5 | 5/5 | — |
| Mean prompt tok | 3743 | 42 | 42 | **3701 saved** |
| Mean latency ms | 2885 | 1549 | 1473 | — |

### Marco facts · OpenCode

| Bench | TEXT / baseline | RESUME / PRI |
|-------|-----------------|--------------|
| Marco local | 5/5 | 5/5 |
| Marco OpenRouter TEXT | 5/5 | 5/5 |
| OpenCode long session (seed 42) | 6/6 | 6/6 |

### Turn sweep (length scaling)

Marco facts + cumulative noise at checkpoints 20/40/60/80.

| cp | inject tok | TEXT | RESUME | OVERFLOW |
|----|------------|------|--------|----------|
| 20 | 6225 | 5/5 | 5/5 | 5/5 |
| 40 | 11981 | 5/5 | 5/5 | 5/5 |
| 60 | 17131 | 5/5 | 3/5 | 3/5 |
| 80 | 23543 | 5/5 | 0/5 | 0/5 |

RoPE geometry audit: **100%** delta_uniformity (verdict `pass`). Garble at cp60+ is documented under [Limitations — Resume inject](LIMITATIONS.md#resume-inject).

### Model matrix — Tier B plug-and-play (2026-06-25)

Same harness (`turn_sweep.py`), different checkpoints on GB10. Compare to frozen Qwen above. Full narrative: [`MODEL_MATRIX.md`](MODEL_MATRIX.md).

| Model | Tag | cp20 TEXT | cp20 RESUME | Artifact |
|-------|-----|-----------|-------------|----------|
| Qwen3.5-35B-A3B-FP8 | `qwen35_hybrid` | 5/5 | **5/5** | `overnight_20260624_003614/` (frozen) |
| Gemma 3 27B-it | `gemma27b` | 5/5 | 2/5 | `model_matrix_gemma27b/pass1/turn_sweep_cp20_plugplay.json` |
| Llama 3.3 70B FP8 | `llama70b` | 5/5 | 0/5 | `model_matrix_llama70b/pass1/turn_sweep_cp20.json` |
| Llama 3 8B Instruct | `llama8b` | 2/5 | 0/5 | `model_matrix_llama8b/pass1/turn_sweep_cp20.json` |

**Llama 8B scaling curve** (post resume pack ordering fix): cp5 RESUME **3/5** → cp20 **0/5** while TEXT stays 5/5 until cp20. See `model_matrix_llama8b/pass1/turn_sweep_cp5_20_curve.json`.

Tier B = K/V inject + RoPE only (no Mamba delta-sum). Matrix proves auto-config boots; **recall parity is Tier A today**.

```bash
# Sequential swap on validation GPU (one model at a time)
./bench/deploy_swap_gx10.sh qwen    # restore baseline
./bench/deploy_swap_gx10.sh gemma
./bench/run_model_matrix.sh --model-tag gemma27b --pass 1
```

### Storage (full turn-sweep session)

| Metric | Value |
|--------|-------|
| Capture files | 143 |
| Capture disk | 648 MB |
| Index rows | 143 |

---

## Reproduce locally

### Prerequisites

- Running PRI container ([Docker](DOCKER.md))
- Hybrid Qwen checkpoint mounted
- Python 3.10+ with `requests` on the host

### Commands

```bash
# Tier 1: Marco facts TEXT vs RESUME
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000

# Inject mode compare (short + long12)
./bench/run_suite.sh --tier mode-compare --seed 42 --base-url http://127.0.0.1:8000

# OpenCode-style multi-turn recall
./bench/run_suite.sh --tier opencode --seed 42 --base-url http://127.0.0.1:8000

# Turn sweep cp20–80
./bench/run_suite.sh --tier sweep --base-url http://127.0.0.1:8000

# RoPE geometry audit (after sweep)
./bench/run_suite.sh --tier geometry --base-url http://127.0.0.1:8000 \
  --sweep-json bench/results/turn_sweep_cp20_80.json
```

Results write to `bench/results/`. See [`bench/README.md`](../bench/README.md) for tier details.

### OpenRouter TEXT baseline (optional)

Copy `bench/env.example` → `bench/.env` and set `OPENROUTER_API_KEY`, then run inject compare with `--text-backend openrouter`.

---

## Harness reference

| Harness | Script | Purpose |
|---------|--------|---------|
| Marco facts | `bench/tier1/marco_facts.py` | TEXT vs RESUME on planted facts |
| Inject compare | `bench/tier1/inject_mode_compare.py` | TEXT vs RESUME vs OVERFLOW |
| Turn sweep | `bench/tier1/turn_sweep.py` | Recall vs chain length at cp20–80 |
| Geometry audit | `bench/tier1/geometry_audit.py` | RoPE pack consistency |
| OpenCode session | `bench/opencode/opencode_long_session_harness.py` | Agent-style multi-turn recall |
| Manifest proof | `bench/opencode/manifest_proof.py` | Verify `rope_start > 0` on turn 2+ |

---

## Canonical artifacts

All paths listed in [`canonical_artifacts.json`](../bench/results/overnight_20260624_003614/canonical_artifacts.json). Key JSON files:

| Bench | Artifact |
|-------|----------|
| Inject compare (local) | `inject_mode_compare_*_postfix.json` |
| Inject compare (OpenRouter) | `inject_mode_compare_*_openrouter_reasoning_none.json` |
| Marco facts | `tier1_marco_facts_*_marco.json` |
| OpenCode | `opencode_long_session_*.json` |
| Turn sweep | `turn_sweep_cp20_80_v5.json` |
| Geometry audit | `geometry_audit_turn_sweep_v5_fixed.json` |
| Model matrix (Tier B) | `model_matrix_*/pass1/turn_sweep_*.json` |

Regenerate summary and research pages after a new run:

```bash
python bench/build_phase_e_summary.py --run-dir bench/results/<run_folder>
python bench/build_research_reports.py --run-dir bench/results/<run_folder>
```

---

## Unit tests (no GPU)

```bash
pip install pytest torch zstandard
pytest tests/ -q
```

Covers `.nls` round-trip, manifest validation, and agent shim helpers.
