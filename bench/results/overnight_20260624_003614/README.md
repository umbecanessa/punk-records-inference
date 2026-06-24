# Published benchmark run — 2026-06-24

**Model:** Qwen3.5-35B-A3B-FP8 · **GPU:** NVIDIA ≥24 GB VRAM · **PRI version:** v0.1.x

This folder contains the canonical proof run referenced from the README and [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md).

---

## Start here

| Document | Purpose |
|----------|---------|
| [**BENCHMARK_SUMMARY.md**](BENCHMARK_SUMMARY.md) | Headline tables (recall, tokens, latency, storage) |
| [**research/00_findings.md**](research/00_findings.md) | Narrative findings + NLS phase mapping |
| [**research/README.md**](research/README.md) | Extended analysis with Mermaid charts |
| [**canonical_artifacts.json**](canonical_artifacts.json) | Index of all canonical JSON artifacts |
| [**benchmark_summary.json**](phase_e_summary.json) | Machine-readable rollup |

---

## Headline (long12 chain)

| Metric | TEXT | RESUME | OVERFLOW |
|--------|------|--------|----------|
| Recall @5 | 5/5 | 5/5 | 5/5 |
| Mean prompt tokens | 3743 | 42 | 42 |
| Token savings vs TEXT | — | **98.9%** | **98.9%** |

**Default inject mode:** `resume`

---

## What's in this folder

| Path | Description |
|------|-------------|
| `inject_mode_compare_*.json` | TEXT vs RESUME vs OVERFLOW scenarios |
| `tier1_marco_facts_*.json` | Marco facts recall arms |
| `opencode_long_session_*.json` | Multi-turn agent recall |
| `turn_sweep_cp20_80_v5.json` | Length scaling at cp20–80 |
| `geometry_audit_turn_sweep_v5_fixed.json` | Post-fix RoPE geometry (100% pass) |
| `research/` | Ten analysis pages (start at `00_findings.md`) + `research_data.json` |
| `internal/` | Historical engineering notes (superseded) — not required reading |

---

## Reproduce

```bash
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000
./bench/run_suite.sh --tier mode-compare --seed 42 --base-url http://127.0.0.1:8000
```

See [bench/README.md](../../bench/README.md) for all tiers.
