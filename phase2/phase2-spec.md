# LNHM Phase 2 — Two Axes of "Overtraining": Depth × Data

**Date:** 2026-07-01
**Status:** methodology fixed; runs pending GPU. Results appended as they land.

## The question

Phase 0 showed the model learns TSP and has a *modest* cross-level transfer effect
(training on larger n slightly helps smaller n). Phase 1 showed composition scales a
tiny model to a million cities but doesn't beat classical methods on **static** TSP —
and that the model's real value is as a fast one-shot small-n solver plus the
transfer thesis, not as a composition local-solver (composition quality is
cleanup-bound). So the live question is: **can we make the model itself
meaningfully better, and along which axis?**

There are two distinct ways to "train harder," and they are not the same thing:

- **Depth (higher n):** extend the training curriculum to larger instances
  (levels 3–12 → 3–20). Tests whether harder anchors transfer down (the Phase 0
  effect, pushed further) and whether the model gets better in the extension range.
- **Data (more per level):** train on *more distinct instances* at each size (not
  more epochs over the same set — that's mere memorization). Tests whether the base
  model is **data-starved** at its current per-level volume.

And the real prize: **do they compound?** Is depth+data better than the sum of each
alone?

## Design: a 2×2 factorial

Same shape as Phase 0's A/B/C/D — two factors, four cells, and "compounding" is the
**interaction term**.

| cell | id | levels (depth) | train instances/level (data) |
|---|---|---|---|
| control | C00 | 3–12 | 4,000 |
| depth | C10 | 3–20 | 4,000 |
| data | C01 | 3–12 | 20,000 (5×) |
| both | C11 | 3–20 | 20,000 |

Architecture is fixed across all cells (d_model=128, 3 encoder layers, 8 heads,
ff=512 — the Phase 0 model). The only variables are the two axes.

### Why train-to-convergence, not compute-matched

We considered holding gradient steps fixed across cells (so "compounding" couldn't
be a compute artifact). We rejected it, for concrete reasons:

1. **The harness is convergence-native.** The additive curriculum graduates each
   level on a validation-accuracy threshold and advances. Forcing fixed steps means
   fighting it (disabling graduation, pinning epochs, bypassing staging) — which
   introduces a *new* regime confound to remove the compute one.
2. **Phase 0's effect was found under convergence.** Backward transfer is *visible*
   precisely because continued training on later levels pushes earlier levels'
   final gap below their graduation bar. Fixed-step would suppress the very signal
   we're after.
3. **Fixed-step has an underfit trap.** At a fixed budget, the depth cell spreads
   the same steps over 18 levels instead of 10 — it can look worse simply because
   *everything* got less training, not because depth fails to transfer.
4. **Convergence answers the decision-relevant question:** "if I train the
   deployable model on more data / higher n, does the *best achievable* model get
   better?"

The one thing compute-matching bought — clean attribution of compounding — we
recover with **discipline instead of regime surgery**:

- **Compute tracking.** Every run records total epochs, total gradient steps,
  instances seen, and wall (`train_summary.json`). A super-additive *quality*
  interaction is then read against whether it cost super-additive *compute*. (More
  informative than fixed-step, which hides the cost.)
- **Seed replication.** Phase 0's effect was ~3–5pp; the interaction term is
  smaller still. We run **2 seeds/cell to start (8 runs)** and add a 3rd only where
  the interaction sits inside seed noise.

"Convergence" is defined identically for every cell: run the additive curriculum to
completion — each level graduates at **50% within-1%-of-optimal accuracy** or hits
the **150-epoch/level cap** — same threshold and cap everywhere.

## Data mechanics (why the axes are clean)

Instances are **deterministic by index**: instance *i* of (level, split) is fully
fixed by `sha256(base_seed : level : split : i)`. Two consequences:

- **Data axis is exact nesting.** The 4,000-instance set is the *first 4,000* of the
  20,000 — a strict subset. `--train-limit` selects it. So "more data" adds
  instances without changing the ones already there: no seed/quality confound.
- **Held-out test set is disjoint by construction.** Generated with a *different*
  `base_seed` (12345), so no test instance can appear in any training pool.

Training pool: `base_seed=0`, ≥20,000 train instances/level for all levels 3–20
(levels 8–12 already have 50k; the rest are topped up). Optimal labels are exact
Held-Karp for n≤12 and LKH-3 for n>12, baked into the data.

*Note on the depth axis:* extending 3–12 → 3–20 inherently adds training data at the
*new* sizes (8 extra levels). That extra volume is intrinsic to "training on harder
problems" and is accounted for via the compute-tracking columns; we do not try to
subtract it out.

## Evaluation — the model's one-shot product

We measure the **standalone model**, not composition (Phase 1 established that
composition quality is set by the cleanup, so it would not reflect model gains).

- **Held-out test set:** `base_seed=12345`, 1,000 instances/level, same set for all
  cells. Optimal labels baked in — evaluation is pure decode-and-compare, no
  re-solving.
- **Two decode modes:** greedy (one shot) and **sampled best-of-16** (the realistic
  high-throughput deployment: draw 16 rollouts, keep the shortest).
- **Metric:** `p = 1/(1 + mean optimality gap)` per level, plus raw gap%.
- **Eval levels:** {5, 8, 10, 12} in-range · {16, 20} extension (only the depth
  cells trained here) · {25, 30} extrapolation (no cell trained here). This
  separates within-range improvement, extension-range learning, and pure
  generalization beyond all training.

### The compounding metric

Per eval-level, with quality = p (or gap):
```
Δ_depth = C10 − C00
Δ_data  = C01 − C00
Δ_both  = C11 − C00
interaction = Δ_both − (Δ_depth + Δ_data)
```
`interaction > 0` → super-additive (the axes reinforce); `≈ 0` → independent;
`< 0` → sub-additive (both chasing the same ceiling). Averaged over seeds, reported
with spread, and read alongside each cell's compute.

## Harness

- `training/train.py` — writes `train_summary.json` (compute tracking) next to the
  self-describing checkpoint. Convergence knobs via `--max-epochs-per-level`,
  `--steps-per-epoch`; cells via `--levels`, `--train-limit`, `--seed`, `--run-name`.
- `analysis/eval_heldout.py` — loads a checkpoint, decodes the held-out set (greedy
  + sampled best-of-K) per level, writes `heldout.json` with gap/p.
- Test set: `python data/generate.py --levels ... --base-seed 12345
  --output-dir data/phase0_test`.

Smoke-tested end-to-end on CPU (train → summary → checkpoint → held-out eval) before
committing GPU time.

## Priors (stated up front, to be scored against)

- Depth alone: **small** (Phase 0 says so).
- Data alone: **unknown** — if the base is data-starved this could exceed depth; if
  it already generalizes, ~flat.
- Interaction: I weakly expect **sub-additive** (both approach the same architectural
  ceiling for a fixed 710K-param model). Would enjoy being wrong.
- Caveat that stands regardless: a better standalone model does **not** rescue
  composition on static TSP (settled in Phase 1). This experiment informs the
  model's *actual* niche — fast one-shot small-n + the transfer thesis — not the
  static-composition verdict.

## Success criteria

- **Compounds:** interaction clearly > seed noise on multiple eval levels.
- **One axis dominates:** its main effect is large, the other and the interaction
  are within noise → train along that axis.
- **Neither helps:** both main effects within noise → the 710K model is at its
  data/depth ceiling; the next lever is capacity, not training regime.
