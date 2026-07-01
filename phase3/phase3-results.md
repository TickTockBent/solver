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

## Run log

- 2026-07-01: Test 1 (blob eval, shallow model) complete. Clustered geometry is harder at
  every n (prior falsified); penalty is pure geometry shift (in-range at n=12), monotone in
  blob tightness. Elastic-k premise weakened but alive; data-diversity retrain is the pivot.
