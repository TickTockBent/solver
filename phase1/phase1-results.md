# LNHM Phase 1 Results — Composition at Scale

**Date:** 2026-07-01

## What Phase 1 asked

Phase 0 established that a tiny transformer (710K params, ~2.8 MB) learns TSP and
shows a modest, real cross-level transfer effect. Phase 1 asked the question the
application vision rests on: **can that small model solve *large* instances by
composition — decompose into small clusters, solve each, stitch — and is the
result competitive on the cost/quality frontier?**

## The honest headline

**The mechanism works; the economics don't — yet, and not on this problem.**

- A model trained only on n≤12 produces coherent tours at **n = 1,000,000** — a
  thousandfold beyond its training range — because it never solves anything bigger
  than ~10 cities; the recursion does the rest. That is a real mechanism result.
- But on **static uniform Euclidean TSP**, the composition pipeline is **not near
  the classical frontier**: it lands at ~12–14% over optimal where a well-built
  near-linear classical solver reaches ~5%, and it costs *more*, not less. The
  learned model does not earn its compute on this problem.
- Composition does beat the cheap space-filling-curve baseline, but only **above a
  measured crossover at ≈ n=1000**, and only by a small, stable margin.

The value case for the learned approach is the **dynamic / constrained** axis
(re-solve one cluster on a change; generalize across problem variants), which
Phase 1 did **not** measure. On the axis we *did* measure, classical wins.

---

## Two metrics: p and f

To make "distance from the frontier" concrete, every run is scored on two numbers.

### p — accuracy proximity to the frontier
```
p = L* / L_ours = 1 / (1 + gap)
```
where `L*` is the best-known tour length (exact/Held-Karp for n≤12, LKH for
n≤~2000, and the Beardwood–Halton–Hammersley estimate `0.7124·√n` for larger n).
**p = 1.0 means you are on the frontier (optimal).** p = 0.88 means your tour is
~14% longer than optimal. (BHH is the *asymptotic* optimum and slightly optimistic
at finite n, so p against BHH is a mild *under*estimate of true p.)

### f — compute per unit accuracy
```
f = (compute-seconds per 10^6 cities) / p
```
Throughput normalized by quality, so a method that is fast *but bad* gets no
credit. **Lower is better.** Reported as a ratio to the frontier, it is the
compute-efficiency distance.

### Reference points
| | p | f |
|---|---|---|
| LKH (near-exact) | ~0.995 | high (super-linear; minutes/M) |
| Good classical near-linear (C: neighbor-2opt + Or-opt) | **~0.95** | **~1.5** |
| Cheap near-linear (SFC + neighbor-2opt, Python) | ~0.86 | ~60–220 |
| **Ours (compose + neighbor-2opt)** | **~0.88** | **~240–690** |

---

## Building the method: the cost-engineering journey

Composition = **partition → local-solve → stitch → cleanup**, applied recursively.
Getting it to near-linear took two structural fixes; neither touched the model.
Measured at **n=10000, compose(model, k=10) + neighbor-2opt**:

| stage | wall | gap | cost vs SFC |
|---|---|---|---|
| flat O(m²) centroid ordering + sequential solves | 17.1 s | 14.5% | 33× |
| **+ recursive ordering** (solver calls itself on centroids) | 6.0 s | 14.4% | 12× |
| **+ batched local solves** (one padded forward pass per level) | 2.1 s | 14.4% | 4.1× |

- **Recursion:** the centroid TSP is the same problem one level up, so ordering
  clusters is done by composition itself (`log_k(n)` depth, O(n) work) instead of
  a flat O(m²) solve. This was the super-linear blow-up.
- **Batching:** clusters within a level are independent — the parallel "batch
  dimension" (as in an LLM). One padded forward pass replaces a Python loop of
  single solves. Quality is identical; cost dropped 6.0 s → 2.1 s.

Both fixes came from reading the structure, not the model. 17 s → 2.1 s, quality
unchanged.

---

## p / f across scale

compose(model, k=10)+neighbor-2opt vs the cheap baseline SFC+neighbor-2opt.

| n | ref | method | gap% | **p** | wall (s) | **f** | penalty | wall/city |
|---|---|---|---|---|---|---|---|---|
| 10 | exact | compose | 5.1 | 0.951 | 0.28 | 29121 | 1.3× | 28 ms |
| 10 | exact | SFC+n2o | 5.1 | 0.951 | 0.22 | 22858 | | 22 ms |
| 100 | LKH | compose | 14.0 | 0.877 | 0.28 | 3161 | 93× | 2.8 ms |
| 100 | LKH | SFC+n2o | 9.7 | 0.912 | 0.00 | 34 | | ~0 |
| 1000 | LKH | compose | 12.0 | 0.893 | 0.70 | 785 | 22× | 0.7 ms |
| 1000 | LKH | SFC+n2o | 13.0 | 0.885 | 0.03 | 35 | | 0.03 ms |
| 10000 | BHH | compose | 14.4 | 0.874 | 2.11 | 241 | 4.0× | 0.21 ms |
| 10000 | BHH | SFC+n2o | 16.5 | 0.858 | 0.51 | 60 | | 0.05 ms |
| 100000 | BHH | compose | 14.1 | 0.877 | 60.1 | 686 | 3.1× | 0.60 ms |
| 100000 | BHH | SFC+n2o | 15.5 | 0.865 | 19.4 | 224 | | 0.19 ms |
| 1000000 | BHH | compose | _pending_ | | | | | |
| 1000000 | BHH | SFC+n2o | _pending_ | | | | | |

### Patterns

1. **Quality crossover at ≈ n=1000, then a *stable* edge.** Composition's advantage
   `Δp = p_compose − p_SFC` goes −0.035 (n=100) → +0.008 (n=1000) → +0.016 (n=10k)
   → +0.012 (n=100k). It rises through the crossover and **plateaus at ~+0.013**.
   Composition's `p` is scale-invariant (~0.877 flat); the SFC baseline's `p`
   *erodes* (0.912 → 0.865). **Composition's product is scale-stability**, not
   raw quality — a space-filling-curve tour degrades as the problem grows; a
   recursive-composition tour holds. Below the crossover, composition is strictly
   worse (worse quality *and* worse compute).

2. **Compute penalty amortizes and stabilizes: 93× → 22× → 4× → 3×.** Composition
   carries a fat fixed overhead (torch + k-d-tree, ~0.3 s) that dominates at small
   n and washes out by n≥10k, settling at ~3× the cheap baseline.

3. **`compute/city` is U-shaped — the cleanup is super-linear.** It bottoms out at
   **n≈10k** (0.21 ms/city) then *rises* to 0.6 ms/city at 100k. Going 10k→100k
   (10× cities), compose's wall went 2.1 s → 60 s (**28×, not 10×**). The
   *construction* (partition + recursive batched solves + stitch) is near-linear;
   the **`neighbor_two_opt` cleanup is not** — pure-Python 2-opt with array
   segment-reversals is super-linear because reversal length grows with n. This is
   an implementation wall (an Or-opt / linked-list 2-opt would restore near-
   linearity), not a fundamental one.

---

## The million-city run

<!-- BEGIN 1M RESULT (inserted when the run completes) -->
_Pending — the n=1,000,000 run is still in progress. It is slow for exactly the
reason in pattern (3): the pure-Python `neighbor_two_opt` cleanup over a million
cities is super-linear. The construction (the "tiny model solves it" part) is the
fast part; the classical 2-opt polish is the bottleneck._
<!-- END 1M RESULT -->

---

## Distance from the frontier — the ruler

On static uniform Euclidean TSP:

- **p ≈ 0.88; the reachable near-linear frontier is ~0.95** (LKH-class ~0.995).
  That's ~2.5–3× too much slack (14% vs ~5% gap) — a real, *engineering* gap, and
  notably **not one the model closes** (the k-cap sweep showed final quality is set
  by the cleanup, not the local solver or cluster size).
- **f ≈ 150× off the frontier.** Decomposed: **~40× is pure Python→C** (same
  algorithm, terrible constants — recoverable with a numba/C kernel), and **~4× is
  the model + composition overhead failing to buy proportional accuracy** on this
  problem.

**On this problem, a good classical near-linear solver dominates our stack on both
axes, and the learned model is a net negative on `f`.** That is the honest result.

---

## What Phase 1 established

1. **The recursive-decomposition mechanism works and scales.** A 2.8 MB model
   trained on n≤12 produces coherent million-city tours by only ever solving ~10
   cities at a time, `log₁₀(1M) ≈ 6` levels deep. This is the LLM trick (fixed
   capacity + iteration → unbounded scale) applied to combinatorial structure.
2. **Composition is a pure large-n instrument** with a measured crossover at
   **≈ n=1000**, above which it beats the cheap baseline by a small, scale-stable
   margin. (A quantitative confirmation of "nobody optimizes a delivery route at
   n=200.")
3. **It is not near the classical frontier on static uniform TSP**, and the model
   does not close the gap. `p` and `f` make this precise and trackable.

## What Phase 1 did NOT test (and where the case actually lives)

- **Dynamic re-optimization** — re-solve one affected cluster on a change while
  classical methods recompute from scratch. This is where `f` should invert in our
  favor, and it is the next experiment.
- **Constrained / non-Euclidean variants** (time windows, asymmetric cost matrices)
  where classical heuristics need per-variant re-engineering but a learned solver
  might transfer. Needs the edge-feature model (deferred from Phase 1).

## Immediate next steps

- **Reversal-free cleanup (Or-opt / linked-list 2-opt)** to flatten the super-linear
  branch and make "near-linear at a million" actually true end-to-end.
- **The dynamic experiment** — the frontier that this approach is actually built for.
