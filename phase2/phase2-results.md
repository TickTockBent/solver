# LNHM Phase 2 — Results

Methodology: [phase2-spec.md](phase2-spec.md). This file is filled as runs land.

**Status:** _pending GPU._ Design = 2×2 (depth × data), train-to-convergence,
2 seeds/cell to start. Cells: **C00** control (3–12, 4k) · **C10** depth (3–20, 4k)
· **C01** data (3–12, 20k) · **C11** both (3–20, 20k).

---

## 1. Held-out one-shot quality — p = 1/(1+gap)

Averaged over seeds; per eval level. Greedy and sampled (best-of-16). In-range
{5,8,10,12} · extension {16,20} · extrapolation {25,30}.

### Sampled best-of-16 (deployment mode)
| cell | n5 | n8 | n10 | n12 | n16 | n20 | n25 | n30 |
|---|---|---|---|---|---|---|---|---|
| C00 control | _ | _ | _ | _ | _ | _ | _ | _ |
| C10 depth | _ | _ | _ | _ | _ | _ | _ | _ |
| C01 data | _ | _ | _ | _ | _ | _ | _ | _ |
| C11 both | _ | _ | _ | _ | _ | _ | _ | _ |

### Greedy (pure one-shot model)
| cell | n5 | n8 | n10 | n12 | n16 | n20 | n25 | n30 |
|---|---|---|---|---|---|---|---|---|
| C00 control | _ | _ | _ | _ | _ | _ | _ | _ |
| C10 depth | _ | _ | _ | _ | _ | _ | _ | _ |
| C01 data | _ | _ | _ | _ | _ | _ | _ | _ |
| C11 both | _ | _ | _ | _ | _ | _ | _ | _ |

---

## 2. Compute per run (train_summary.json)

Convergence is data-dependent, so record what each cell cost. This is how we read a
quality interaction against a compute interaction.

| cell | seed | levels | total steps | instances seen | wall (s) |
|---|---|---|---|---|---|
| C00 | 0 | 3–12 | _ | _ | _ |
| C00 | 1 | 3–12 | _ | _ | _ |
| C10 | 0 | 3–20 | _ | _ | _ |
| C10 | 1 | 3–20 | _ | _ | _ |
| C01 | 0 | 3–12 | _ | _ | _ |
| C01 | 1 | 3–12 | _ | _ | _ |
| C11 | 0 | 3–20 | _ | _ | _ |
| C11 | 1 | 3–20 | _ | _ | _ |

---

## 3. Compounding — interaction term

Per eval level, on p (sampled). `interaction = Δ_both − (Δ_depth + Δ_data)`.
Positive = super-additive, ≈0 = independent, negative = sub-additive.

| n | Δ_depth (C10−C00) | Δ_data (C01−C00) | Δ_both (C11−C00) | interaction |
|---|---|---|---|---|
| 5 | _ | _ | _ | _ |
| 8 | _ | _ | _ | _ |
| 10 | _ | _ | _ | _ |
| 12 | _ | _ | _ | _ |
| 16 | _ | _ | _ | _ |
| 20 | _ | _ | _ | _ |
| 25 | _ | _ | _ | _ |
| 30 | _ | _ | _ | _ |

---

## 4. Verdict

_pending — score against the criteria in the spec: compounds / one axis dominates /
neither helps._

---

## Run log

- 2026-07-01: methodology fixed, harness built + smoke-tested on CPU. Awaiting GPU.
