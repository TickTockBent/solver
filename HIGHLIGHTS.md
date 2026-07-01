# Highlights: what fell out of this that you can reuse

The headline of this repo is a controlled negative result, but several of the
byproducts are independently useful. This is the short list, with pointers.

## 1. A free OOD detector: sign(best-of-k − greedy)

Best-of-k sampling beats greedy decoding when the model's sampling
distribution is calibrated, and falls *below* greedy when the model is out of
distribution: 16 noisy draws from a miscalibrated policy are all worse than
its argmax. Measured on the same checkpoints: the n≤12 model inverts by n=20
(best-of-16 p=0.799 vs greedy 0.906 at n=30) while the n≤20 model at n=30 does
not invert (0.975 vs 0.957). The sign flips exactly at the model's competence
boundary.

Use it anywhere you have a stochastic decoder: compare best-of-k to greedy on
a sample of inputs and you have an in-distribution test that needs no access
to training data. Corollary for evaluation hygiene: include the greedy rollout
in the best-of-k pool, and never publish sampled-basis generalization numbers
without the greedy column (ours were inflated 3.5x).

## 2. Depth extrapolation: train to N, predict to ~1.5N

A curriculum trained to n=20 produces one-shot greedy tours at p=0.957 at
n=30, a size it never saw, versus 0.906 for the n=12 curriculum. The effect
survived a pre-registered decode-mode control (it is prediction, not search:
sampling only breaks even with greedy at k≈4 and adds +0.02 by k=16). Data
volume at fixed sizes contributes nothing; the training *distribution* is the
lever, in size and in geometry both. Whether reach scales multiplicatively
(c·N) or additively (N+c) is the open question this repo leaves on the table
(test-suite T4).

## 3. The diversity free lunch

Retraining the same architecture on a three-way geometric diet (uniform,
Gaussian blobs with randomized count and spread, anisotropic irregular)
closed the clustered-geometry penalty from 3.6pp to 0.9pp in-range, lifted
clustered performance at every size including far out of range, and slightly
*improved* uniform extrapolation. Zero regression anywhere; the only cost was
~45% more convergence steps. Proposed mechanism: a dense blob's local
neighborhood is statistically a slice of a larger uniform instance, so
density diversity is implicit exposure to larger-n local structure. If you
train a geometric model on one distribution, this says the fix is cheap.

## 4. The oracle-row ablation design

The leaf-solver ablation (T1) that killed the model's role in composition
included four arms: random permutation, nearest-neighbor, the model, and
exact Held-Karp. The exact arm is the design contribution. When the oracle
lands at the same post-cleanup quality as everything else, "your model was
too small" is answered before it is asked: no achievable capacity outperforms
exact, and exact does not matter. Any ablation of a component feeding a
strong downstream optimizer should include the oracle arm.

## 5. The landscape framing of the negative result

What died here is not "learned solvers" but a specific match-up: on Euclidean
TSP, cheap improving moves (2-opt, Or-opt) are so dense that any valid tour
descends to the same local optimum, so constructor quality is worthless and
composition's real contribution (its partition/stitch structure retains a
small, scale-stable edge over space-filling curves even with random leaves)
is all that remains. The mechanism validated: a 2.8MB model recursing on its
own output produced coherent million-city tours. The economics failed because
this problem sells a cheaper substitute. Problems whose landscapes lack cheap
improving moves are where the mechanism goes unpriced, and that is a
testable, falsifiable direction (test-suite T7) rather than a consolation.

## 6. Methodology that paid rent

- **Pre-registration in git.** Priors and verdict thresholds committed before
  results, scored in the open, including the ones that lost. The repo's
  credibility is its commit order.
- **The control you almost didn't build.** The naive curriculum-vs-single-level
  comparison (A vs C) produced a false negative; only adding a
  curriculum-up-to-anchor control (D) isolated the real transfer effect.
  The pair of controls that bracket the hypothesis from both sides is the
  experiment.
- **Deterministic-by-index data.** Every instance is a pure function of
  (base_seed, level, split, index), so "5x more data" is a strict superset of
  the baseline pool and held-out sets are disjoint by construction. Nesting
  kills a whole class of data confounds.
- **Convergence-native factorials with compute tracking.** Rather than
  fighting the curriculum harness to hold steps fixed, run every cell to
  convergence and record steps/instances/wall, then read quality effects
  against their measured cost.
- **p/f ruler.** Quality as p = 1/(1+gap) and cost as cpu-core-seconds per
  million cities per unit p. Contention-robust, parallelism-honest, and it
  made "distance from the frontier" a number instead of a mood.

## 7. Engineering artifacts

- `lnhm/analysis/fast_local_search.py`: compiled neighbor-list 2-opt with
  don't-look bits plus reversal-free linked-list Or-opt. Took the million-city
  cleanup from ~50 minutes to ~52 seconds at better quality.
- `lnhm/analysis/compose.py`: recursive composition with batched leaf solves
  (one padded forward pass per level) and DP-optimal port stitching.
- `lnhm/data/generate_mixed.py`: the three-geometry training diet generator.
