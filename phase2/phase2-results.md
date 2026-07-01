# LNHM Phase 2 — Results

Methodology: [phase2-spec.md](phase2-spec.md).

**Status:** 2 seeds/cell complete (8 runs) on the GPU box. Verdict below.
Cells: **C00** control (3–12, 4k) · **C10** depth (3–20, 4k) · **C01** data (3–12, 20k)
· **C11** both (3–20, 20k).

---

## 1. Held-out one-shot quality — p = 1/(1+gap), sampled best-of-16, mean over 2 seeds

In-range {5,8,10,12} · extension {16,20} · extrapolation {25,30} (no cell trained here).

| cell | n5 | n8 | n10 | n12 | n16 | n20 | n25 | n30 |
|---|---|---|---|---|---|---|---|---|
| C00 control | 1.000 | 1.000 | 0.999 | 0.995 | 0.971 | 0.926 | 0.861 | 0.799 |
| C10 depth | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.995 | 0.988 | 0.975 |
| C01 data | 1.000 | 1.000 | 0.998 | 0.995 | 0.972 | 0.927 | 0.864 | 0.802 |
| C11 both | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.995 | 0.986 | 0.971 |

_Greedy (pure one-shot) view still to be pulled (`--mode greedy`); best-of-16
saturates the in-range sizes, hiding any small in-range transfer effect._

---

## 2. Compute per run (train_summary.json)

| cell | seed | total steps | instances seen | wall (s) |
|---|---|---|---|---|
| C00 control | 0 | 2200 | 563,200 | 43.2 |
| C00 control | 1 | 2100 | 537,600 | 47.4 |
| C10 depth | 0 | 24,900 | 6,374,400 | 853.2 |
| C10 depth | 1 | 22,300 | 5,708,800 | 735.9 |
| C01 data | 0 | 2200 | 563,200 | 40.5 |
| C01 data | 1 | 2100 | 537,600 | 41.6 |
| C11 both | 0 | 19,900 | 5,094,400 | 647.5 |
| C11 both | 1 | 19,900 | 5,094,400 | 657.9 |

Depth costs **~15–17× control's wall** (8 extra levels to graduate). Data costs
**~0 extra** at 3–12 (C01 ≡ C00 in steps/wall — the model isn't data-limited).
Note: C11 (both, 20k) converged in *fewer* steps than C10 (depth, 4k) — more data
graduated each level slightly faster. So data's only effect was mild convergence
speed-up, **not** final quality.

---

## 3. Compounding — interaction term (Δp vs control, sampled)

`interaction = Δ_both − (Δ_depth + Δ_data)`.

| n | Δ_depth | Δ_data | Δ_both | interaction |
|---|---|---|---|---|
| 5 | +0.000 | +0.000 | +0.000 | −0.000 |
| 8 | +0.000 | −0.000 | +0.000 | +0.000 |
| 10 | +0.001 | −0.000 | +0.001 | +0.000 |
| 12 | +0.005 | −0.000 | +0.005 | +0.000 |
| 16 | +0.028 | +0.001 | +0.028 | −0.001 |
| 20 | +0.069 | +0.001 | +0.069 | −0.001 |
| 25 | +0.127 | +0.003 | +0.125 | −0.004 |
| 30 | +0.176 | +0.003 | +0.173 | −0.007 |

---

## 4. Verdict — depth dominates (via extrapolation); data is inert; no compounding

Scored against the spec's criteria: **"one axis dominates" — decisively depth.**

1. **Depth's value is extrapolation, and it is enormous.** In-range (n≤12) every cell
   is saturated (~1.0 under best-of-16), so depth looks like nothing there — which is
   why Phase 0 (in-range only) called the effect "modest." But out-of-range it
   explodes: **+0.069 at n=20, +0.127 at n=25, +0.176 at n=30.** Control collapses to
   p=0.799 (25% gap) at n=30; the depth model holds **0.975 (2.6% gap)** — on a size
   *no cell trained on*. Training on 3–20 teaches a generalization that extrapolates
   well past the training range. This *inverts* my prior (I expected depth to be small
   — it is, but only in the regime Phase 0 measured).
2. **Data is inert.** Δ_data is +0.000…+0.003 everywhere. The 710K model is **not
   data-starved** at 4,000 instances/level; 5× more buys no quality (and doesn't even
   change convergence at 3–12). The "more problems" axis of overtraining does nothing
   here.
3. **No compounding.** Interaction ≈ 0, turning *slightly negative* at the
   extrapolation frontier (−0.007 at n=30): adding data to the depth model marginally
   *hurt* extrapolation (C11 0.971 vs C10 0.975). Since data's main effect is ~0,
   C11 ≈ C10. The axes do not reinforce; if anything they mildly antagonize at the edge.
4. **Cost.** The only axis that pays (depth) is also the expensive one (~16×); the
   cheap axis (data) is worthless. So the honest trade is clear: **spend compute on
   depth, not data.**

### Implications

- **Train deeper, not wider-in-data.** To improve the standalone model, extend the
  size range; do not add instances per level. We can likely *lower* the data budget
  below 4k with no loss (worth pinning the floor).
- **Sharpens the model's niche:** a depth-20 model is near-frontier one-shot
  (best-of-16) up to ~n=30 by *extrapolation*. Strong support for the "fast one-shot
  small-/moderate-n solver" thesis.
- **Does not change the static-composition verdict** (Phase 1). Composition leaves are
  n≤~12, already saturated — depth can't help there. Consistent with the k-sweep
  (composition is cleanup-bound).

### Caveats / next

- 2 seeds. The depth effect (+0.176) dwarfs any seed noise; the data null and the
  ~0 interaction are consistent across seeds. A 3rd seed would only be to nail the
  small negative interaction (~−0.005), which is minor either way.
- Pull the **greedy** table to see the pure-model in-range transfer (Phase 0's regime),
  currently hidden by best-of-16 saturation.
- **Open question worth its own run:** how far does depth extrapolate? Does "train to
  N" buy near-frontier out to ~1.5N regardless of N (train 3–30 → good to ~45)? Where
  does it break? And the data floor (2k? 1k?).

---

## Run log

- 2026-07-01: methodology fixed, harness built + smoke-tested on CPU.
- 2026-07-01: 8 runs (4 cells × 2 seeds) complete on the Windows Docker GPU box.
  Verdict: depth dominates via extrapolation, data inert, no compounding.
