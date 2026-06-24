# PRI bench research — analysis index

**Run:** `bench/results/overnight_20260624_003614`  
**Generated:** 2026-06-24T11:32:12.269292+00:00  
**Model:** `/model` · Git `9944c727779bfe722139b0851ce804149790c7d4`

Publication-oriented analysis from the GX10 overnight proof run. Each page embeds
**Mermaid charts** (GitHub-native `xychart-beta` / `pie`) with explicit y-axis ranges.
For interactive exploration in Cursor, open the [PRI bench research canvas](/Users/umber/.cursor/projects/c-Users-umber-Documents-GitHub-punk-records-inference/canvases/pri-bench-research.canvas.tsx) beside the chat.

## Quick headline

| Metric | TEXT (inline) | RESUME (KV inject) | Savings |
|--------|--------------:|-------------------:|--------:|
| Recall @5 (long12) | 5/5 | 5/5 | — |
| Mean prompt tokens | 3743.2 | 42.2 | **98.9%** |
| Mean latency (ms) | 2884.9 | 1548.5 | **46.3%** |
| Est. GPU energy / recall | 0.20034 Wh | 0.107535 Wh | **46.3%** |
| Capture disk | — | 648.11 MB (143 files) | — |

**Default inject mode:** `resume` · Executive summary: [`../PHASE_E_SUMMARY.md`](../PHASE_E_SUMMARY.md)

## Analysis pages

| # | Topic | File |
|---|-------|------|
| 1 | Methodology & arms | [01_methodology.md](01_methodology.md) |
| 2 | Token efficiency | [02_token_efficiency.md](02_token_efficiency.md) |
| 3 | Latency | [03_latency_analysis.md](03_latency_analysis.md) |
| 4 | Computational cost (prefill proxy) | [04_computational_cost.md](04_computational_cost.md) |
| 5 | Storage footprint | [05_storage_footprint.md](05_storage_footprint.md) |
| 6 | Turn-sweep scaling | [06_turn_sweep_scaling.md](06_turn_sweep_scaling.md) |
| 7 | RoPE geometry | [07_rope_geometry.md](07_rope_geometry.md) |
| 8 | Failure modes & garble | [08_failure_modes.md](08_failure_modes.md) |
| 9 | Energy & cost | [09_energy_and_cost.md](09_energy_and_cost.md) |

## Chart index

Research pages embed Mermaid diagrams inline (no separate `charts/` folder). Regenerate with:

```bash
python bench/build_research_reports.py --run-dir bench/results/<run_folder>
```

## Machine-readable

- [`research_data.json`](research_data.json) — all extracted metrics for charts / MoE comparison
- [`../phase_e_summary.json`](../phase_e_summary.json) — Phase E rollup
- [`../canonical_artifacts.json`](../canonical_artifacts.json) — artifact path index

## Related audits (historical)

Pre-fix investigation notes (superseded where noted):

- [`../FAILURE_AUDIT.md`](../FAILURE_AUDIT.md) — harness bug triage (OpenRouter, garbled guard)
- [`../ROPE_DELTA_AUDIT.md`](../ROPE_DELTA_AUDIT.md) — pre-RoPE-fix microscope (now **100%** — see page 7)
