# Benchmarks

Reproducible harnesses for validating PRI on a live vLLM instance. Published proof artifacts live under [`results/overnight_20260624_003614/`](results/overnight_20260624_003614/).

Full documentation: [docs/BENCHMARKS.md](../docs/BENCHMARKS.md)

---

## Quick start

```bash
# Prerequisites: running container on :8000, hybrid Qwen checkpoint mounted
pip install requests

# Smoke: Marco facts TEXT vs RESUME
./bench/run_suite.sh --tier 1 --base-url http://127.0.0.1:8000

# Value case: TEXT vs RESUME vs OVERFLOW on short and long chains
./bench/run_suite.sh --tier mode-compare --seed 42 --base-url http://127.0.0.1:8000

# Agent-style multi-turn recall
./bench/run_suite.sh --tier opencode --seed 42 --base-url http://127.0.0.1:8000
```

---

## Harness tiers

| Tier | Script | What it measures |
|------|--------|------------------|
| `1` | `tier1/marco_facts.py` | Marco facts recall: TEXT vs RESUME |
| `mode-compare` | `tier1/inject_mode_compare.py` | TEXT vs RESUME vs OVERFLOW on planted facts |
| `opencode` | `opencode/opencode_long_session_harness.py` | Multi-turn agent recall after 8+ turns |
| `sweep` | `tier1/turn_sweep.py` | Length scaling at checkpoints cp20–80 |
| `geometry` | `tier1/geometry_audit.py` | RoPE pack geometry audit on a sweep chain |

---

## Published proof run (2026-06-24)

| Resource | Path |
|----------|------|
| Run overview | [`results/overnight_20260624_003614/README.md`](results/overnight_20260624_003614/README.md) |
| Summary tables | [`BENCHMARK_SUMMARY.md`](results/overnight_20260624_003614/BENCHMARK_SUMMARY.md) |
| Canonical artifact list | [`canonical_artifacts.json`](results/overnight_20260624_003614/canonical_artifacts.json) |
| Extended analysis | [`research/README.md`](results/overnight_20260624_003614/research/README.md) |

**Validated configuration:** Qwen3.5-35B-A3B-FP8 · NVIDIA GPU ≥24 GB VRAM · stock vLLM plugin image

Regenerate summary and research pages after a new run:

```bash
python bench/build_phase_e_summary.py --run-dir bench/results/<run_folder>
python bench/build_research_reports.py --run-dir bench/results/<run_folder>
```

---

## Configuration

- OpenRouter TEXT baseline (optional): copy `bench/env.example` → `bench/.env` and set `OPENROUTER_API_KEY`
- Harness KVP parity with production: `bench/opencode/nls_kvp_helpers.py`

---

## Unit tests (no GPU)

```bash
pip install pytest torch zstandard
pytest tests/ -q
```
