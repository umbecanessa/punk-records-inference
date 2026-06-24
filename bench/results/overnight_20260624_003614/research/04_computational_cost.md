# 4 — Computational cost (prefill proxy)

## Model

Linear prefill proxy: cost ∝ prompt_tokens (standard KV-cache resume assumption)

We report **relative prefill units** = `prompt_tokens` at recall time. For transformer decoders,
attention prefill scales ~O(n) per layer w.r.t. sequence length when KV is cold; RESUME inject
reuses stored KV so the live prefill length equals last-message tokens (~42) not chain length (~3743).

Decode cost is small here (≤200 completion tokens) and similar across arms.

## Long12 recall — prefill units (mean per probe)

| Arm | Prefill proxy units | Relative cost |
|-----|--------------------:|--------------:|
| TEXT | 3743.2 | 1.00× |
| RESUME | 42.2 | 0.011× |

**Prefill reduction:** 98.9%

## Turn sweep — cost vs correctness trade-off

At cp80 the RESUME arm still pays **~42 prompt tokens** (99%+ prefill savings) but recall is **0/5**.
Computational efficiency does not imply semantic recovery at extreme inject depth.

| cp | inject tok | TEXT prefill (mean) | RESUME prefill (mean) | TEXT pass | RESUME pass |
|----|----------:|--------------------:|----------------------:|----------:|------------:|
| 20 | 6225 | 6209.2 | 42.2 | 5/5 | 5/5 |
| 40 | 11981 | 11906.2 | 42.2 | 5/5 | 5/5 |
| 60 | 17131 | 17003.2 | 42.2 | 5/5 | 3/5 |
| 80 | 23543 | 23362.2 | 42.2 | 5/5 | 0/5 |

See [09_energy_and_cost.md](09_energy_and_cost.md) for Wh, electricity $, and cloud API $ models.