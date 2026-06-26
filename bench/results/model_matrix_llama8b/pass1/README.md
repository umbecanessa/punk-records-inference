# Llama 3 8B — model matrix pass 1

**Model:** `meta-llama/Meta-Llama-3-8B-Instruct`  
**Host:** GX10 (GB10, ~121 GB unified)  
**Date:** 2026-06-25

| Artifact | Description |
|----------|-------------|
| `turn_sweep_cp20.json` | cp20 after resume ordering fix |
| `turn_sweep_cp5_20_curve.json` | Scaling curve cp5/10/15/20 |

Compare against frozen Qwen: `bench/results/overnight_20260624_003614/`.
