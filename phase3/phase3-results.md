# Phase 3 — Elastic-k probe, Test 1: does the model hold on clustered geometry?

**Motivation.** Phase 2 showed depth training extrapolates forward in n (train 3–20 →
near-frontier out to n≈30). That opened elastic-k: instead of forcing a uniform cluster
size onto a non-uniform point cloud, let cluster sizes float within a band and respect
natural dense blobs, giving fewer, more benign seams. The load-bearing assumption was:
**a dense blob of ~k points is an easy one-shot solve** (my prior: "denser is easier —
shorter edges, clearer structure"). Test 1 checks that assumption directly, cheaply, on
CPU with the shallow 710K model (`runs/full`, levels 3–12).

**Method.** New generator `data/generate_clustered.py` (Gaussian-mixture blobs, same
schema/labeling as `phase0_test`: Held-Karp n≤12, LKH n>12). Uniform arm = existing
`phase0_test`. Both evaluated with the same shallow model, greedy + sampled best-of-16,
200 instances/level. `p = 1/(1+mean_gap)`, gap vs each instance's own optimum.

## Result — the prior is wrong. Clustered blobs are harder, at every n.

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

It's the geometry, not the size. At n=12, fully in the shallow model's training
range, clustered geometry still costs 3.6pp sampled (0.995→0.959). The model is "expert"
at n=12 and still worse on a blob, so this is a pure distribution-shift penalty: the model
trained on uniform points only and has never seen a tight blob.

**σ-sweep confirms the mechanism (n=20, sampled p):**

| σ (blob tightness) | 0.03 | 0.05 | 0.10 | 0.20 | uniform |
|---|---|---|---|---|---|
| p_sampled | 0.853 | 0.845 | 0.877 | 0.916 | 0.924 |

Monotone recovery toward the uniform value as the blobs loosen. The penalty is the
clustering itself: the tighter (more off-distribution) the blob, the worse the model does.

## What this means for elastic-k

This weakens the premise without killing it. The "keep the blob whole" leaf solve is
harder than the same-n uniform solve, so elastic-k's benefit (fewer, cleaner seams)
is partly offset by a harder leaf. The net (seam savings − harder leaf) can only be settled
by the clustered-globals A/B (elastic vs fixed-k vs SFC, all + `fast_local` cleanup). But
we now go in with realistic expectations instead of an inflated prior.

The real lever this surfaces is data diversity, not volume. Phase 2 killed data
volume on uniform (5× instances → nothing). This is a different axis Phase 2 never
touched: the model has never seen clustered geometry. The penalty is monotone in
OOD-ness and therefore looks trainable: mix clustered instances into training and the
blob penalty should shrink. If a blob-competent model solves blobs ≈ as well as uniform,
the elastic-k premise is restored and strengthened, and it's a cheap thing to try.

## Caveats

- **Shallow model = pessimistic case.** This is the 3–12 model. The depth-20 model (Phase 2
  C10, on the GPU box) extrapolates far better in n; whether it also generalizes better in
  geometry is untested. Depth may narrow the blob gap on its own.
- 200 instances/level, single seed of the test set (base_seed=777). The effect is large and
  monotone across n and σ, well clear of noise at this size.
- Blob model is a 3-cluster isotropic Gaussian mixture; real data (delivery stops, cities)
  is messier. This is the controlled first cut, not the final geometry.

## Revised next steps (cheapest first)

1. **(cheap, local) Data-diversity retrain.** Fine-tune / retrain the shallow model on mixed
   uniform+clustered data → does the blob penalty shrink? Directly tests the "trainable"
   hypothesis and is the pivotal result for whether elastic-k is worth building.
2. **(GPU) Depth model on blobs.** Re-run this exact eval with the C10 depth-20 checkpoint:
   does depth narrow the geometry gap for free?
3. **(the arbiter) Clustered-globals A/B.** elastic-k vs fixed-k vs SFC, all + `fast_local`,
   on clustered global instances; measures whether respecting blobs beats cutting them once
   the leaf-solve realities are priced in.

## Artifacts

- `lnhm/data/generate_clustered.py` — clustered instance generator.
- `lnhm/analysis/wrap_checkpoint.py` — wraps raw state_dicts into the self-describing format.
- `lnhm/data/phase3_clustered/` — the clustered test set + `heldout_{uniform,clustered}.json`.
- Shallow model: `lnhm/runs/full/model_selfdesc.pt` (wrapped from `model_final.pt`).

---

# Test 2 — Data-diversity retrain: does teaching the model blobs close the gap?

**Method.** Retrained the same 710K arch (d128/3L/8H/ff512) on a three-way diet via
`data/generate_mixed.py`: ⅓ uniform, ⅓ clustered (randomized k∈[2,6], σ∈[0.03,0.15];
deliberately not the test's fixed k=3/σ=0.05), ⅓ "truly random bounded" (anisotropic
blobs + random uniform background). Levels 3–12, 4000/level, train-to-convergence, CPU.
Converged in 3200 steps / 589s (vs ~2200 for uniform-only, ~45% more). Then evaluated on
both held-out sets. The randomized training params make this a genuine generalization
test to the fixed test geometry, not memorization.

## Result — diversity nearly closes the geometry gap, and slightly helps uniform

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

1. The blob penalty is largely closed. At n=12 (in-range) it drops from −3.6pp to
   −0.9pp, near uniform-parity. Across the whole range the residual penalty is ~1–3pp,
   down from ~4–7pp. The "harder leaf" Test 1 held against elastic-k mostly evaporates once
   the model is trained for geometric diversity.

2. Uniform did not regress; it slightly improved, most at extrapolation (+0.014 at
   n=20/25/30). No capacity tradeoff in this 710K model.

3. n is still a wall. Both columns still collapse in absolute terms at n=30 (0.79–0.81 =
   ~24–26% gap). Diversity does not fix extrapolation-in-n; only depth (Phase 2) does.

## Mechanism — geometric diversity is a partial, free substitute for depth

The unexpected part (uniform extrapolation improving, and clustered n=30 lifting +0.068
rather than staying flat) has one explanation: a dense blob's local neighborhood looks
like a slice of a larger uniform instance. Training on varied local density is implicit
exposure to the local statistics of bigger n, so diversity transfers to both clustered
geometry (large effect) and uniform extrapolation (small effect); each is really about
handling denser local structure than n≤12 uniform ever presents. It buys a little n for
free, but does not replace depth.

## Prediction scorecard (called before the eval, for the record)

- **n=12 clustered:** WS 0.985 / Claude ≥0.99 / **actual 0.987**. WS near-exact;
  Claude's full-parity call just missed.
- **Clustered n=30 (the tell-tale):** WS "measurable lift, ~0.78 — blob edge structure is
  n-independent" / Claude "flat, ≈0.726 — only depth fixes n" / **actual 0.794 (+0.068)**.
  WS confirmed: geometry familiarity partially transfers across size. Still a collapse in
  absolute terms on the sampled basis (Claude right on that much).
- **Uniform extrapolation:** WS "−0.02 regression (capacity split)" / Claude "holds within
  0.005" / **actual +0.014**. WS wrong on direction; Claude right that there is no
  regression but outside the stated band. Neither predicted the improvement, which is
  the real finding (see Mechanism above), and it follows from WS's own n=30 mechanism.

## Verdict for elastic-k

Premise **restored**. A diversity-trained model solves blobs almost as well as uniform
in-range, so respecting natural dense blobs (elastic k) no longer pays a prohibitive
leaf-solve penalty. The clustered-globals A/B (elastic vs fixed-k vs SFC, +`fast_local`)
is now worth running with a diversity-trained leaf model. And the deployment recipe
sharpens: **train the composition leaf model on a diverse geometric diet, not uniform.**

_Superseded by T5-Q2 below: the fixed-k damage gate shows fixed-k loses ~nothing on
globby data, so elastic-k is not justified and the A/B need not be built. Test 2's
"premise restored" was about the leaf's blob competence, which T1 then showed is moot for
static composition. The diversity-training result stands on its own as a
distribution-robustness finding (T5-Q1), just not as a reason to build elastic-k._

---

# Adversarial suite (T1–T8) — results so far

Running the pre-registered kill-suite (`lnhm-phase3-test-suite.md`). Each test states its
verdict rule before the run. Kill-table framing: a failure retires a specific claim, not
the methodology, the p/f ruler, the depth-extrapolation science, or the compiled kernel.

## T1 — trivial local-solver ablation (null hypothesis): CONFIRMED INERT

Held the composition pipeline fixed (SFC partition, k_cap=10, `fast_local` cleanup) and
swapped only the leaf solver. Uniform globals, 3 seeds, BHH reference.
`analysis/t1_ablation.py`. p_raw = pre-cleanup proximity, p_clean = post-cleanup, f =
cpu-core-seconds per 1e6 cities / p.

| n | solver | p_raw | p_clean | f |
|---|---|---|---|---|
| 10k | random | 0.386 | 0.931 | 135 |
| 10k | nn | 0.641 | 0.929 | 115 |
| 10k | model | 0.660 | 0.928 | 243 |
| 10k | held_karp | 0.666 | 0.926 | 462 |
| 100k | random | 0.386 | 0.936 | 127 |
| 100k | nn | 0.636 | 0.935 | 122 |
| 100k | model | 0.657 | 0.932 | 243 |
| 100k | held_karp | 0.663 | 0.933 | 463 |

**Verdict rule:** p(model) − p(NN) < 0.005 after cleanup ⇒ model inert. Δp = **−0.0008
(10k), −0.0031 (100k) → INERT.** Pre-cleanup the leaf matters enormously (random 0.39 →
model/HK 0.66); post-cleanup all four collapse to ~0.93 and the order inverts: a random
permutation leaf reaches the same final tour as the model, which lands marginally below
nearest-neighbor at ~2× the cpu (Pareto-dominated). `fast_local` drives any reasonable
(even random) start to the same local optimum, so leaf quality is irrelevant to the static
result. Retires exactly one claim: the model's role in static composition.

## T5-Q2 — fixed-k damage gate (elastic-k go/no-go): NOT JUSTIFIED

Since T1 proved the leaf irrelevant, any excess loss of fixed-k composition on globby data
must come from the partition (seams cut through blobs). Held leaf = Held-Karp, cleanup =
`fast_local`; compared fixed-k composition vs partition-free SFC+fast and NN+fast on uniform
vs clustered globals (tight ~40-pt blobs, larger than k=10: the seam-damage scenario).
`analysis/t5_fixedk_gate.py`. n=10k, 5 seeds. `vs_best%` = tour length over the best
partition-free baseline.

| dist | pipeline | mean len | cpu_s | vs_best% |
|---|---|---|---|---|
| uniform | composition | 76.899 | 4.41 | +2.16% |
| uniform | sfc_fast | 77.688 | 0.04 | +3.21% |
| uniform | nn_fast | 75.274 | 4.28 | +0.00% |
| clustered | composition | 56.524 | 4.30 | +2.19% |
| clustered | sfc_fast | 56.934 | 0.03 | +2.94% |
| clustered | nn_fast | 55.310 | 4.22 | +0.00% |

**Verdict rule:** fixed-k damage < ~1.1pp ⇒ elastic-k not justified. Excess damage on
clustered vs uniform is **+0.03pp** (composition vs best baseline: +2.19% vs +2.16%). And
against SFC+fast directly (the true partition isolation: same cleanup, only the k=10 seams
differ), composition beats SFC on both distributions (−0.72% clustered, −1.02% uniform),
excess **+0.30pp**. Either way far under the gate → **elastic-k is not justified by data.**
Mechanism: the seams fixed-k cuts through a dense blob are local (blob points are mutually
near), and `fast_local`'s neighbor-2opt/Or-opt repairs local damage regardless of source;
there is no un-repairable structural loss for elastic-k to recover. This is the World-A
resolution of the T1 question, and it preemptively closes the clustered-globals A/B.

## T2 (part 1) — greedy decode table (prediction vs search): HEADLINE HOLDS, MAGNITUDE CORRECTED

Re-assembled the Phase 2 held-out `p_greedy` (already computed by `eval_heldout`, no
re-inference), mean over 2 seeds:

| cell | n5 | n8 | n10 | n12 | n16 | n20 | n25 | n30 |
|---|---|---|---|---|---|---|---|---|
| C00 control | 0.998 | 0.994 | 0.986 | 0.977 | 0.961 | 0.944 | 0.924 | 0.906 |
| C10 depth | 0.999 | 0.998 | 0.996 | 0.994 | 0.989 | 0.980 | 0.969 | 0.957 |
| C01 data | 0.999 | 0.994 | 0.985 | 0.978 | 0.960 | 0.944 | 0.924 | 0.906 |
| C11 both | 0.999 | 0.998 | 0.996 | 0.994 | 0.988 | 0.980 | 0.970 | 0.959 |

**Verdict rule:** greedy depth p ≥ 0.94 at n=30 ⇒ extrapolation is model-borne. Depth greedy
n=30 = **0.957 ≥ 0.94 → PASSES.** Near-frontier one-shot prediction to n=30 confirmed;
the product claim survives the search-vs-prediction test.

But greedy deflates the magnitude of the depth advantage the sampled basis reported:

| n | Δ_depth **sampled** | Δ_depth **greedy** |
|---|---|---|
| 16 | +0.028 | +0.028 |
| 20 | +0.069 | +0.036 |
| 25 | +0.127 | +0.046 |
| 30 | **+0.176** | **+0.051** |

The control's sampled "collapse" (0.799 at n=30) was largely a sampling artifact: control
greedy n=30 = 0.906, so best-of-16 hurt the control by −0.107 (16 noisy draws off a
miscalibrated model, all worse than its greedy argmax) while it helped depth by +0.018.
Restated: depth's one-shot prediction advantage at n=30 is **+0.051, not +0.176**.
Both models degrade gracefully in prediction; depth degrades more gracefully and keeps a
usable sampling distribution where the control loses one, a real but separate property from
prediction quality. Data inert and interaction ≈ 0 on greedy too (robust to decode mode).

### T2 (part 2) — best-of-k curve (depth model C10 s0, 1000 instances)

Sampled p as a function of decode budget k, with the greedy (one-shot) baseline:

| n | greedy | k=1 | k=2 | k=4 | k=8 | k=16 |
|---|---|---|---|---|---|---|
| 25 | **0.970** | 0.936 | 0.960 | 0.974 | 0.982 | 0.988 |
| 30 | **0.958** | 0.914 | 0.940 | 0.957 | 0.967 | 0.977 |

Two findings, both favouring the prediction framing:

1. Search starts underwater. A single sample is worse than greedy (n=30: k=1 = 0.914
   vs 0.958). Sampling doesn't break even with the greedy argmax until **k≈4** (n=25 crosses
   ~k=3, n=30 ~k=4). For k=1–2, one-shot beats search.
2. Above break-even the climb is shallow and unsaturated. Marginal gain per doubling at
   n=30: +0.026 (1→2), +0.017 (2→4), +0.010 (4→8), +0.010 (8→16). Best-of-16 adds only
   **+0.019** over greedy and is still rising at k=16: a slow grind, not a cliff.

**T2 complete:** extrapolation is overwhelmingly model-borne. Greedy gives 0.958 at n=30 in
one pass; search buys ~+0.02 more but only after ≥4 passes clear the break-even and 16 passes
to bank the full headline. The product line: **one-shot ≈ 0.958 at n=30; the 0.977 sampled
headline is a 16× compute premium for ~2pp**, exactly the latency caveat the pre-registration
flagged.

## Runnable-vs-remote status

- **Done locally (this box, CPU):** T1, T5-Q2, T2 part 1 (re-assemble only).
- **Needs the depth checkpoint / GPU box:** T2 part 2 (best-of-k curve); full-strength
  T3/T5-Q1 (price the *depth* model — local models collapse at n=30).
- **Heavy builds, not started:** T3 (compute-matched classical rows), T4 (provisioning law;
  needs Concorde 21–30), T6 (dynamic vs incremental repair, where the thesis lives),
  T7 (edge-feature model + asymmetric TSP, the moat test), T8 (data floor).

## Run log

- 2026-07-01: Test 1 (blob eval, shallow model) complete. Clustered geometry is harder at
  every n (prior falsified); penalty is pure geometry shift (in-range at n=12), monotone in
  blob tightness. Elastic-k premise weakened but alive; data-diversity retrain is the pivot.
- 2026-07-01: Test 2 (mixed-diet retrain) complete. Diversity nearly closes the blob penalty
  (−3.6pp→−0.9pp at n=12), slightly improves uniform extrapolation (+0.014), no regression;
  n remains a wall. Geometric diversity is a partial free substitute for depth. Elastic-k
  premise restored.
- 2026-07-01: T1 (local-solver ablation) complete. Model INERT in static composition
  (Δp vs NN = −0.001/−0.003 after cleanup, Pareto-dominated). Leaf irrelevant post-cleanup.
- 2026-07-01: T5-Q2 (fixed-k damage gate) complete. Excess fixed-k damage on globby data
  ≈ 0 (+0.03pp vs best baseline). Elastic-k NOT justified; cleanup repairs local blob seams.
  Supersedes the Test-2 recommendation to build the clustered-globals A/B.
- 2026-07-01: T2 part 1 (greedy table) complete. Depth greedy n=30 = 0.957 ≥ 0.94 → PASSES;
  extrapolation is model-borne. But sampled overstated depth's advantage ~3.5× (control's
  sampled collapse was a best-of-16 artifact); true prediction edge at n=30 is +0.051.
- 2026-07-01: T2 part 2 (best-of-k curve) complete. Search starts underwater (k=1 < greedy),
  breaks even at k≈4, and best-of-16 adds only +0.019 over greedy at n=30 (still rising).
  Extrapolation is model-borne; the sampled headline is a 16× compute premium for ~2pp. T2 closed.
