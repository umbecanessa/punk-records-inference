# Benchmarks

Reproduce proof artifacts with one command against a live vLLM instance.

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
| 2026-06-23 | manifest proof turn 2 (KL #648) | `/model` | `rope_start=24` | `bench/results/manifest_opencode_t2.json` |

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

## Unit tests (no GPU)

```bash
pip install pytest torch zstandard
pytest tests/ -q
```

Covers `.nls` round-trip, manifest validation, agent shim helpers.
