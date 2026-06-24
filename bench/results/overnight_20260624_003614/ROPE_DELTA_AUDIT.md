# RoPE delta microscope — turn_sweep v5 (83 blocks)

**Chain:** `turn_sweep_0b3f0e4cb1` / `chain_thread_eb71920e0d87`  
**Tool:** `bench/tier1/rope_delta_microscope.py`  
**Input:** `geometry_audit_turn_sweep_v5.json` + `turn_sweep_cp20_80_v5.json`

---

## KPI summary

| Metric | Value | Target |
|--------|------:|--------|
| **delta_uniformity** | **98.8%** (82/83) | ≥ 99% excellent · ≥ 92% research baseline |
| **mode RoPE delta** | **-22** on 82 blocks | uniform resume phantom-aware rotate |
| **outlier blocks** | **1** (turn 59) | 0 |
| **delta stdev** | 2.18 | 0 at perfect uniformity |
| **geometry verdict** | `fail_rope_pack` | would be `pass` at 100% uniform |

**Grade: `research_baseline`** — you are at **0.988**, just below the 0.99 “clean harness” bar. One bad block drives the fail verdict.

This aligns with the pre-release research band (~0.92–0.93) when garbled/neutral-fallback captures pollute phantom metadata; a clean chain should approach **0.99+**.

---

## The one outlier (turn 59)

| Field | Value |
|-------|------:|
| turn_index | 59 |
| label | `noise-N59` (neutral fallback after 2 garbled probes) |
| rope_delta | **-42** (expected **-22**) |
| pack_offset | 16029 |
| manifest rope_start | 22 |
| **capture_num_phantom** | **16049** ← root issue |
| rope_old_effective | 16071 (= 22 + 16049) |
| num_tokens | 84 |

**All other 82 blocks:** `capture_num_phantom=0` → audit falls back to cumulative `total_tokens` before block → uniform **-22**.

**Turn 59 alone** has a non-zero `capture_num_phantom` written at capture time. The geometry audit trusts that manifest field:

```
rope_old = rope_start + capture_num_phantom  →  22 + 16049 = 16071
rope_new = pack_offset                       →  16029
delta    = -42
```

If phantom were **0** (like every other block), the audit would use `total_tokens=16029`:

```
rope_old = 22 + 16029 = 16051  →  delta = -22  ✓
```

---

## Connector trace

`pri/connector.py` writes manifest phantom from capture registry:

```python
capture_phantom = int(info.get("num_phantom", 0) or 0)
if capture_phantom > 0:
    extra["capture_num_phantom"] = capture_phantom
```

`num_phantom` is set during inject Phase 3 (`_capture_registry[request_id]["num_phantom"] = max(num_snap, 0)`).

**Hypothesis:** On neutral-fallback turn 59, the request ran with resume inject context (~16k phantom tokens). That live `num_phantom` was **persisted into the .nls manifest** even though the block’s pack position is computed from cumulative chain length (16029). The stored 16049 is ~20 tokens off from pack offset — likely stale/wrong inject phantom at capture, not the pack-plan phantom the rotate math expects.

**This is a connector/capture bug**, not cross-test memory bleed.

---

## Correlation with garbled / neutral fallback

Turn sweep JSON shows **5 neutral fallbacks** at turns 9, 57, 58, 59, 61. Only **turn 59** produced the bad phantom manifest.

| Turn | Neutral? | rope_delta |
|------|----------|------------|
| 9 | yes | -22 |
| 57 | yes | -22 |
| 58 | yes | -22 |
| **59** | **yes** | **-42** |
| 61 | yes | -22 |

Garbled guard prevents chain gaps; it does **not** yet prevent wrong `capture_num_phantom` on substitute captures after long inject context.

---

## manifest rope “gaps” (82) — not pack gaps

`chain_continuity.gap_count=82` is **manifest** `rope_end[i] ≠ rope_start[i+1]` because every block has `rope_start=22` (per-request resume-shaped capture). **Pack offsets are contiguous** (`pack_continuous=true`). This is expected for turn-capture mode; do not confuse with inject pack holes.

---

## Recommended connector fixes (priority)

1. **Capture phantom semantics** — When writing `capture_num_phantom`, store **phantom at first token of this block’s own prefill**, not live request inject phantom if it includes unrelated chain state. Or omit field when it would disagree with pack-plan `total_tokens`.

2. **Geometry audit tolerance** — Optional: if `capture_num_phantom` disagrees with pack offset by > threshold, fall back to `total_tokens` and emit `warn` not `fail_rope_pack`.

3. **Neutral fallback captures** — After garbled plant, capture substitute with `memory_no_capture` probe first or reset phantom registry before clean capture.

4. **Re-run microscope after fix** — Target **delta_uniformity ≥ 0.99** on same 83-block sweep.

---

## Garbled guard (inject compare) — now added

`inject_mode_compare.py` now uses `plant_turn_hygiene` (same as turn sweep) for facts + noise. Prevents poisoned `.nls` from unguarded noise plants; should improve long12 RESUME once re-run.

---

*Generated 2026-06-24 from overnight v5 artifacts.*
