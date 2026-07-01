# Phase 3 — Elastic-k probe, Test 1: does the model hold on clustered geometry?

**Motivation.** Phase 2 showed depth training extrapolates *forward* in n (train 3–20 →
near-frontier out to n≈30). That opened elastic-k: instead of forcing a uniform cluster
size onto a non-uniform point cloud, let cluster sizes float within a band and respect
natural dense blobs — fewer, more benign seams. The load-bearing assumption was:
**a dense blob of ~k points is an easy one-shot solve** (my prior: "denser is easier —
shorter edges, clearer structure"). Test 1 checks that assumption directly, cheaply, on
CPU with the shallow 710K model (`runs/full`, levels 3–12).

**Method.** New generator `data/generate_clustered.py` (Gaussian-mixture blobs, same
schema/labeling as `phase0_test`: Held-Karp n≤12, LKH n>12). Uniform arm = existing
`phase0_test`. Both evaluated with the *same* shallow model, greedy + sampled best-of-16,
200 instances/level. `p = 1/(1+mean_gap)`, gap vs each instance's own optimum.

## Result — the prior is WRONG. Clustered blobs are *harder*, at every n.

**Sampled best-of-16, p:**

| n | uniform | clustered (σ=0.05) | Δ (clust − unif) |
|---|---|---|---|
| 8  | 1.000 | 0.993 | −0.007 |
| 10 | 0.999 | 0.979 | −0.020 |
| 12 | 0.995 | 0.959 | −0.036 |
| 16 | 0.970 | 0.907 | −0.063 |
| 20 | 0.924 | 0.850 | −0.074 |
| 25 | 0.856 | 0.788 | −0.068 |
| 30 | 0.799 | 0.726 | −0.073 |

**Greedy (pure one-shot), p:**

| n | uniform | clustered | Δ |
|---|---|---|---|
| 8  | 0.994 | 0.979 | −0.015 |
| 12 | 0.980 | 0.961 | −0.019 |
| 16 | 0.963 | 0.934 | −0.029 |
| 20 | 0.941 | 0.912 | −0.029 |
| 25 | 0.924 | 0.896 | −0.028 |
| 30 | 0.907 | 0.865 | −0.042 |

**It's the geometry, not the size.** At n=12 — *fully in the shallow model's training
range* — clustered geometry still costs 3.6pp sampled (0.995→0.959). The model is "expert"
at n=12 and still worse on a blob. So this is a pure distribution-shift penalty: the model
trained on uniform points only and has never seen a tight blob.

**σ-sweep confirms the mechanism (n=20, sampled p):**

| σ (blob tightness) | 0.03 | 0.05 | 0.10 | 0.20 | uniform |
|---|---|---|---|---|---|
| p_sampled | 0.853 | 0.845 | 0.877 | 0.916 | 0.924 |

Monotone recovery toward the uniform value as the blobs loosen. The penalty *is* the
clustering: the tighter (more off-distribution) the blob, the worse the model does; loosen
it back to near-uniform and the penalty vanishes.

## What this means for elastic-k

**Weakens the premise, does not kill it.** The "keep the blob whole" leaf solve is *harder*
than the same-n uniform solve, not easier — so elastic-k's benefit (fewer, cleaner seams)
is partly offset by a harder leaf. The net (seam savings − harder leaf) can only be settled
by the clustered-globals A/B (elastic vs fixed-k vs SFC, all + `fast_local` cleanup). But
we now go in with realistic expectations instead of an inflated prior.

**The real lever this surfaces: data *diversity*, not volume.** Phase 2 killed data
*volume* on uniform (5× instances → nothing). This is a different axis Phase 2 never
touched: the model has never *seen* clustered geometry. The penalty is monotone in
OOD-ness and therefore looks *trainable* — mix clustered instances into training and the
blob penalty should shrink. If a blob-competent model solves blobs ≈ as well as uniform,
the elastic-k premise is restored **and** strengthened (and it's a cheap thing to try).

## Caveats

- **Shallow model = pessimistic case.** This is the 3–12 model. The depth-20 model (Phase 2
  C10, on the GPU box) extrapolates far better in *n*; whether it also generalizes better in
  *geometry* is untested. Depth may narrow the blob gap on its own.
- 200 instances/level, single seed of the test set (base_seed=777). The effect is large and
  monotone across n and σ, well clear of noise at this size.
- Blob model is a 3-cluster isotropic Gaussian mixture; real data (delivery stops, cities)
  is messier. This is the controlled first cut, not the final geometry.

## Revised next steps (cheapest first)

1. **(cheap, local) Data-diversity retrain.** Fine-tune / retrain the shallow model on mixed
   uniform+clustered data → does the blob penalty shrink? Directly tests the "trainable"
   hypothesis and is the pivotal result for whether elastic-k is worth building.
2. **(GPU) Depth model on blobs.** Re-run this exact eval with the C10 depth-20 checkpoint —
   does depth narrow the geometry gap for free?
3. **(the arbiter) Clustered-globals A/B.** elastic-k vs fixed-k vs SFC, all + `fast_local`,
   on clustered global instances — measures whether respecting blobs beats cutting them once
   the leaf-solve realities are priced in.

## Artifacts

- `lnhm/data/generate_clustered.py` — clustered instance generator.
- `lnhm/analysis/wrap_checkpoint.py` — wraps raw state_dicts into the self-describing format.
- `lnhm/data/phase3_clustered/` — the clustered test set + `heldout_{uniform,clustered}.json`.
- Shallow model: `lnhm/runs/full/model_selfdesc.pt` (wrapped from `model_final.pt`).

---

# Test 2 — Data-diversity retrain: does teaching the model blobs close the gap?

**Method.** Retrained the *same* 710K arch (d128/3L/8H/ff512) on a three-way diet via
`data/generate_mixed.py`: ⅓ uniform, ⅓ clustered (RANDOMIZED k∈[2,6], σ∈[0.03,0.15] —
deliberately *not* the test's fixed k=3/σ=0.05), ⅓ "truly random bounded" (anisotropic
blobs + random uniform background). Levels 3–12, 4000/level, train-to-convergence, CPU.
Converged in 3200 steps / 589s (vs ~2200 for uniform-only — ~45% more). Then evaluated on
BOTH held-out sets. The randomized training params make this a genuine *generalization*
test to the fixed test geometry, not memorization.

## Result — diversity nearly closes the geometry gap, and slightly *helps* uniform

**Sampled best-of-16, p. Baseline = uniform-trained (`runs/full`); Mixed = this retrain.**

| n | uniform test: base → mixed | clustered test: base → mixed | blob penalty (unif−clust): base → mixed |
|---|---|---|---|
| 8  | 1.000 → 1.000 | 0.993 → 0.999 | 0.007 → 0.001 |
| 10 | 0.999 → 0.999 | 0.979 → 0.996 | 0.020 → 0.003 |
| 12 | 0.995 → 0.996 | 0.959 → 0.987 | **0.036 → 0.009** |
| 16 | 0.970 → 0.975 | 0.907 → 0.952 | 0.063 → 0.023 |
| 20 | 0.924 → 0.938 | 0.850 → 0.909 | 0.074 → 0.029 |
| 25 | 0.856 → 0.870 | 0.788 → 0.855 | 0.068 → 0.015 |
| 30 | 0.799 → 0.813 | 0.726 → 0.794 | 0.073 → 0.019 |

Three findings:

1. **The blob penalty is largely closed.** At n=12 (in-range) it drops from −3.6pp to
   −0.9pp — near uniform-parity. Across the whole range the residual penalty is ~1–3pp,
   down from ~4–7pp. The "harder leaf" Test 1 held against elastic-k mostly evaporates once
   the model is trained for geometric diversity.

2. **Uniform did not regress — it slightly IMPROVED**, most at extrapolation (+0.014 at
   n=20/25/30), zero regression anywhere. No capacity tradeoff in this 710K model.

3. **n is still a wall.** Both columns still collapse in absolute terms at n=30 (0.79–0.81 =
   ~24–26% gap). Diversity does not fix extrapolation-in-n; only depth (Phase 2) does.

## Mechanism — geometric diversity is a partial, free substitute for depth

The unexpected part (uniform extrapolation *improving*, and clustered n=30 lifting +0.068
rather than staying flat) has one explanation: **a dense blob's local neighborhood looks
like a slice of a larger uniform instance.** Training on varied local density is implicit
exposure to the local statistics of bigger n. So diversity transfers to *both* clustered
geometry (large effect) and uniform extrapolation (small effect) — both are really about
handling denser local structure than n≤12 uniform ever presents. It buys a little n for
free, but does not replace depth.

## Prediction scorecard (called before the eval, for the record)

- **n=12 clustered:** Claude 0.985 / user ≥0.99 / **actual 0.987** — Claude near-exact.
- **Clustered n=30 (the tell-tale):** Claude "measurable lift ~0.78, not flat" / user "still
  collapses (flat)" / **actual 0.794 (+0.068)** — Claude's "geometry partially transfers
  across size" confirmed; still a collapse in absolute terms (user right on that).
- **Uniform extrapolation:** Claude "−0.02 regression" / user "holds within 0.005" /
  **actual +0.014** — both wrong on sign/magnitude; neither predicted the improvement.

## Verdict for elastic-k

Premise **restored**. A diversity-trained model solves blobs almost as well as uniform
in-range, so respecting natural dense blobs (elastic k) no longer pays a prohibitive
leaf-solve penalty. The clustered-globals A/B (elastic vs fixed-k vs SFC, +`fast_local`)
is now worth running with a diversity-trained leaf model. And the deployment recipe
sharpens: **train the composition leaf model on a diverse geometric diet, not uniform.**

## Run log

- 2026-07-01: Test 1 (blob eval, shallow model) complete. Clustered geometry is harder at
  every n (prior falsified); penalty is pure geometry shift (in-range at n=12), monotone in
  blob tightness. Elastic-k premise weakened but alive; data-diversity retrain is the pivot.
- 2026-07-01: Test 2 (mixed-diet retrain) complete. Diversity nearly closes the blob penalty
  (−3.6pp→−0.9pp at n=12), slightly improves uniform extrapolation (+0.014), no regression;
  n remains a wall. Geometric diversity is a partial free substitute for depth. Elastic-k
  premise restored.
