# Abstract (draft)

**Working title:** Local Search Erases Learned Constructors: Controlled
Negative and Positive Results for Neural TSP Solvers at Small Scale

**Alt titles:**
- What a 710K-Parameter TSP Model Can and Cannot Do: A Pre-Registered Study
- The Cleanup Ate the Model: Composition, Extrapolation, and Evaluation
  Artifacts in Neural Combinatorial Optimization

---

Learned construction heuristics for combinatorial optimization are usually
evaluated against classical constructors. We instead ask whether they survive
classical *improvement*. Using a 710K-parameter transformer trained by
curriculum on exactly-solved Euclidean TSP instances (n ≤ 12), we pre-registered
predictions and controls for each claim before measuring it, and we publish the
git history as provenance.

Three results are negative with a common mechanism. Recursive decomposition
lets the tiny model produce valid million-city tours in minutes, but a
leaf-solver ablation spanning random permutations, nearest-neighbor, the model,
and an exact Held-Karp oracle shows all four reach the same tour quality after
a standard 2-opt/Or-opt cleanup: constructor quality is erased by local search,
so no constructor of any capacity can earn its compute on this landscape. The
oracle arm answers the scale objection directly, and a 4x-capacity
replication changes none of the verdicts: greedy extrapolation moves by less
than 0.003, so the observed effects are properties of the training
distribution rather than parameter count. Added capacity only accelerates
in-range convergence and improves out-of-distribution sampling calibration,
narrowing the best-of-k inversion without improving the argmax.

Two results are positive. Extending the training curriculum from n ≤ 12 to
n ≤ 20 yields one-shot greedy tours at p = 0.957 (gap ≈ 4.5%) at n = 30, 1.5x
the training ceiling, and this extrapolation survives a decode-mode control
showing it is model-borne rather than search-borne. Mixing clustered and
irregular geometries into training closes the out-of-distribution geometry
penalty from 3.6pp to 0.9pp at no cost to uniform performance.

Finally, an evaluation artifact with field-wide implications: best-of-k
sampling inverts below greedy decoding precisely where the model is out of
distribution, inflating our own extrapolation headline 3.5x before correction.
The sign of (best-of-k − greedy) is a free out-of-distribution detector, and
sampled-basis extrapolation claims in the literature deserve re-examination on
the greedy basis.

We conclude that learned constructors are only worth building for problems
whose improvement landscapes lack cheap improving moves, and we characterize
Euclidean TSP as the wrong landscape by that criterion, chosen deliberately as
the friendliest test and failed for a reason that generalizes.
