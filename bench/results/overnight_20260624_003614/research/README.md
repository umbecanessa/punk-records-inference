# PRI benchmark research — analysis index

**Run:** [`../README.md`](../README.md) · **Model:** Qwen3.5-35B-A3B-FP8 · **Date:** 2026-06-24

Extended analysis from the published proof run. Start with **[Key findings](00_findings.md)** for narrative context and NLS cross-reference.

Each page embeds **Mermaid charts** (GitHub `xychart-beta` / `pie`) with explicit y-axis ranges.

## Quick headline

| Metric | TEXT (inline) | RESUME (KV inject) | Savings |
|--------|--------------:|-------------------:|--------:|
| Recall @5 (long12) | 5/5 | 5/5 | — |
| Mean prompt tokens | 3743.2 | 42.2 | **98.9%** |
| Mean latency (ms) | 2884.9 | 1548.5 | **46.3%** |
| Est. GPU energy / recall | 0.200 Wh | 0.108 Wh | **46.3%** |
| Capture disk | — | 648 MB (143 files) | — |

**Default inject mode:** `resume` · Summary: [`../BENCHMARK_SUMMARY.md`](../BENCHMARK_SUMMARY.md)

## Analysis pages

| # | Topic | File |
|---|-------|------|
| 0 | **Key findings & NLS context** | **[00_findings.md](00_findings.md)** |
| 1 | Methodology & arms | [01_methodology.md](01_methodology.md) |
| 2 | Token efficiency | [02_token_efficiency.md](02_token_efficiency.md) |
| 3 | Latency | [03_latency_analysis.md](03_latency_analysis.md) |
| 4 | Computational cost | [04_computational_cost.md](04_computational_cost.md) |
| 5 | Storage footprint | [05_storage_footprint.md](05_storage_footprint.md) |
| 6 | Turn-sweep scaling | [06_turn_sweep_scaling.md](06_turn_sweep_scaling.md) |
| 7 | RoPE geometry | [07_rope_geometry.md](07_rope_geometry.md) |
| 8 | Failure modes | [08_failure_modes.md](08_failure_modes.md) |
| 9 | Energy & cost | [09_energy_and_cost.md](09_energy_and_cost.md) |

## Regenerate

```bash
python bench/build_research_reports.py --run-dir bench/results/overnight_20260624_003614
```

## Machine-readable

- [`research_data.json`](research_data.json) — extracted metrics
- [`../phase_e_summary.json`](../phase_e_summary.json) — rollup JSON
- [`../canonical_artifacts.json`](../canonical_artifacts.json) — artifact index

## Architecture context

Full pipeline narrative (retrieval-first design): [Neural Ledger System](https://github.com/umbecanessa/neural-ledger-system) · [docs/OVERVIEW.md](../../../docs/OVERVIEW.md)

Historical engineering triage (superseded): [`../internal/`](../internal/)
