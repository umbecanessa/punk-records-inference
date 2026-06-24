> **Maintainer doc** — bench measurement matrix for choosing inject-mode default and building proof tables. End users: [Benchmarks](BENCHMARKS.md).

# Benchmark data plan — Punk Records Inference value case

This document defines the measurement matrix for comparing **PRI KV resume**
against **standard inline-history** baselines, and for choosing the default
inject mode (`resume` vs `resume_overflow`).

Artifacts land in `bench/results/` with JSON schemas suitable for README proof
tables and external charts.

**Latest proof run:** `bench/results/overnight_20260624_003614/` ·
[`PHASE_E_SUMMARY.md`](../bench/results/overnight_20260624_003614/PHASE_E_SUMMARY.md)

| Phase | Status | Outcome |
|-------|--------|---------|
| A — Inject mode default | ✅ | Default stays **`resume`** |
| B — Standard vs PRI | ✅ | Marco + OpenCode + turn sweep measured |
| C — Storage snapshot | ✅ | 143 captures, 648 MB (full sweep session) |
| D — Latency breakdown | Optional | Per-request ms/tokens in harness JSON |
| E — Publication summary | ✅ | Tables in BENCHMARKS.md + PHASE_E_SUMMARY.md |

---

## Goals

1. **Correctness** — recall under agent-style sessions (tier1 + OpenCode)
2. **Efficiency** — prompt tokens avoided vs TEXT/inline baseline
3. **Latency** — end-to-end request time and prefill savings
4. **Storage** — `.nls` capture size, index growth, disk on `/data/pri`
5. **Operational** — startup profile correctness across checkpoints

---

## Arms (every study)

| Arm | Description | Client pattern |
|-----|-------------|----------------|
| **TEXT (OpenRouter)** | Full inline history via cloud API | Same model slug, **no PRI** — isolated baseline |
| **TEXT (local)** | Full inline on same vLLM | `memory_off=1` — shares GPU with PRI arms |
| **RESUME** | Chain KV inject, no Swiss | `memory_inject_mode=resume` |
| **RESUME_OVERFLOW** | Chain + Swiss backfill on trim | `memory_inject_mode=resume_overflow` |
| **OPENCODE_PRI** | Agent harness → PRI vLLM | agent shim + resume/overflow |
| **OPENCODE_BASELINE** | Same harness → stock vLLM | no KV plugin, full history |

---

## OpenRouter TEXT baseline (isolated)

Use when you want TEXT arm off the PRI box (same model family, no KV plugin bleed).

```bash
export OPENROUTER_API_KEY=sk-or-v1-...   # never commit — see bench/env.example
export OPENROUTER_MODEL=qwen/qwen3.5-35b-a3b

./bench/run_suite.sh --tier mode-compare --base-url http://127.0.0.1:8000 --seed 42
# or explicitly:
python bench/tier1/inject_mode_compare.py --text-backend openrouter ...
```

- **Plant + RESUME/OVERFLOW** → local PRI (`PRI_BASE_URL`)
- **TEXT recall** → OpenRouter with full inline `messages` (no `kv_transfer_params`)
- JSON records `text_backend` + `openrouter_model` per run

Rotate any key that was pasted into chat; store only in `bench/.env` (gitignored).

---

## Phase A — Inject mode default (GX10, priority)

**Owner:** bench agent on room `7APVGK`  
**Harness:** `bench/tier1/inject_mode_compare.py`

### A1 Short chain (smoke)

```bash
# Container A: NLS_API_INJECT_MODE=resume (default)
./bench/run_suite.sh --tier mode-compare --base-url http://127.0.0.1:8000 --seed 42

# Container B: recreate with NLS_API_INJECT_MODE=resume_overflow, restart
./bench/run_suite.sh --tier mode-compare --base-url http://127.0.0.1:8000 --seed 42
```

Record: `bench/results/inject_mode_compare_42.json`

### A2 Long chain (overflow stress)

```bash
NOISE_TURNS=12 ./bench/run_suite.sh --tier mode-compare --base-url http://127.0.0.1:8000 --seed 42

python bench/tier1/inject_mode_compare.py \
  --noise-turns 12 \
  --resume-max-tokens 4096 \
  --base-url http://127.0.0.1:8000 \
  --out bench/results/inject_mode_compare_42_long.json
```

### A3 Decision rule

| Criterion | Weight |
|-----------|--------|
| Recall pass rate (5 probes) | Must not regress vs RESUME |
| Mean prompt tokens on recall | Lower is better |
| Mean latency | Lower is better |
| Swiss activation rate (overflow only) | Log when trim evicts |

**Flip default to `resume_overflow` only if** A1+A2 show ≥ same recall and
meaningful benefit (trim backfill or future compaction scenario).

**2026-06-24 result:** RESUME and OVERFLOW tie on recall (5/5) for short + long12;
OVERFLOW does not recover cp60+ turn-sweep cliff. **Keep `resume` as default.**

## Phase B — Standard vs PRI (value case)

### B1 Tier-1 Marco (existing)

```bash
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
```

Metrics from JSON:
- `text_pass` vs `resume_pass`
- Per-request `usage.prompt_tokens` delta

### B2 OpenCode long session

```bash
./bench/run_suite.sh --tier opencode --base-url http://127.0.0.1:8000 --seed 42
```

Run twice:
1. PRI container (resume or winning mode from Phase A)
2. Stock vLLM baseline (same model, no middleware, full transcript)

Capture: `RECALL x/y`, chain_id, garbled flags.

### B3 Turn sweep (length scaling)

```bash
./bench/run_suite.sh --tier sweep --base-url http://127.0.0.1:8000
```

Plot vs checkpoint: TEXT vs RESUME vs arm_d at cp 20/40/60/80.

---

## Phase C — Storage & profile artifacts

| Artifact | Path | Notes |
|----------|------|-------|
| Model profile | `/data/pri/model_profile.json` | layer probes, fingerprint |
| Profile env | `/data/pri/profile.env` | inject gating |
| Captures | `/data/pri/snapshot/captures/*.nls` | per-turn size |
| Index | `/data/pri/index.jsonl` | row count |
| Admin stats | `GET /admin/memory/stats?user_id=` | aggregate bytes |

Per-run folder convention:

```
bench/results/runs/<YYYYMMDD>_<run_id>/
  manifest.json          # git sha, image tag, seed, inject_mode
  inject_mode_compare.json
  tier1_marco_facts.json
  opencode_long_session.json
  store_stats.json
  capture_sizes.csv      # optional script: du per .nls
```

---

## Phase D — Latency breakdown (optional deep dive)

For each recall request log:
- HTTP total ms (harness)
- vLLM `usage.prompt_tokens` / `completion_tokens`
- Inject mode from connector logs (`RESUME-INJECT`, `ARM-D`)

Compare TEXT prefill token count vs RESUME last-message-only + phantom inject tokens.

---

## Phase E — Publication-ready summary table

Source: `bench/results/overnight_20260624_003614/phase_e_summary.json` (long12 chain, local PRI).

| Metric | TEXT | RESUME | OVERFLOW | Δ vs TEXT |
|--------|------|--------|----------|-----------|
| Recall @5 | 5/5 | 5/5 | 5/5 | — |
| Mean prompt tok | 3743 | 42 | 42 | **3701 saved** |
| Mean latency ms | 2885 | 1549 | 1473 | — |
| Capture disk MB | — | — | — | 648 (143 captures, full sweep) |
| OpenCode RECALL | 6/6 baseline | 6/6 PRI | — | — |

Turn sweep RESUME at cp60/cp80: **3/5** and **0/5** while TEXT **5/5** — documented
limitation; RoPE geometry audit **pass** at 100% delta_uniformity.

Rebuild after a new run:

```bash
python bench/build_phase_e_summary.py --run-dir bench/results/<run_folder>
python bench/build_research_reports.py --run-dir bench/results/<run_folder>
./bench/run_phase_e_completion.sh   # optional: marco OR, resume4096, store stats
```

Extended analysis pages (charts, per-probe tables): `<run_folder>/research/README.md`

## Startup profile (automatic)

On container boot, `docker/start.sh` runs:

```bash
python3 -m pri.startup_profile --model-path "$MODEL_PATH" ...
source /data/pri/profile.env
```

Verify in logs:
- `PRI_INJECT_PROFILE=resume|resume_overflow`
- `NLS_NEURAL_SCORING=0|1`
- `NLS_DELTA_FACT_PROBE_LAYERS=...` derived from `config.json`

---

## Reporting

Post to Agent Room `7APVGK`:
- PASS/FAIL per phase
- JSON paths under `bench/results/`
- Recommended default inject mode with evidence
- Blockers for overflow (if Swiss never activates without trim)
