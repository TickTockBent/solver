# LNHM Phase 1 Results — Composition at Scale

**Date:** 2026-07-01

## What Phase 1 asked

Phase 0 established that a tiny transformer (710K params, ~2.8 MB) learns TSP and
shows a modest, real cross-level transfer effect. Phase 1 asked the question the
application vision rests on: **can that small model solve *large* instances by
composition (decompose into small clusters, solve each, stitch), and is the
result competitive on the cost/quality frontier?**

## The result

**The mechanism works; the economics don't (yet, and not on this problem).**

- A model trained only on n≤12 produces coherent tours at **n = 1,000,000**, a
  thousandfold beyond its training range, because it never solves anything bigger
  than ~10 cities; the recursion does the rest. That is a real mechanism result.
- But on **static uniform Euclidean TSP**, the composition pipeline is **not near
  the classical frontier**: it lands at ~12–14% over optimal where a well-built
  near-linear classical solver reaches ~5%, and it costs more. The
  learned model does not earn its compute on this problem.
- Composition does beat the cheap space-filling-curve baseline, but only **above a
  measured crossover at ≈ n=1000**, and only by a small, stable margin.

The value case for the learned approach is the **dynamic / constrained** axis
(re-solve one cluster on a change; generalize across problem variants), which
Phase 1 did not measure. On the axis we did measure, classical wins.

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
f = (CPU core-seconds per 10^6 cities) / p
```
Compute normalized by quality, so a method that is fast *but bad* gets no credit.
**Lower is better.** Reported as a ratio to the frontier, it is the compute-
efficiency distance.

**Basis: CPU core-seconds, not wall-clock.** f is a cost metric: the compute
actually consumed (`time.process_time()`, summed across all threads of the process).
A background task stealing the CPU cannot corrupt it, and parallel work is
charged for every core it uses. Wall-clock (latency) is reported alongside but does
not feed f: it is contention-sensitive and hides parallelism (a 4-core burst
looks 4× cheaper than it is). The two diverge exactly where our pipeline fans out:
the torch model construction and the k-d-tree query use all cores, so their
cpu-seconds exceed their wall-seconds; the single-threaded local-search cleanup has
cpu ≈ wall. (Caveat: LKH is a subprocess and is not counted; fine, since it is
only ever the untimed reference, never a timed method.)

_Historical note: the two tables above (the original p/f sweep and the million-city
row) were measured on **wall-clock**. For their single-threaded pure-Python cleanups
that ≈ cpu-seconds when uncontended, but the compose construction fans out, so their
compose-`f` slightly understates true compute cost. The "Kernel rewrite" section
below uses the corrected cpu-seconds basis and reports both columns._

### Reference points
| | p | f |
|---|---|---|
| LKH (near-exact) | ~0.995 | high (super-linear; minutes/M) |
| Good classical near-linear (C: neighbor-2opt + Or-opt) | **~0.95** | **~1.5** |
| Cheap near-linear (SFC + neighbor-2opt, Python) | ~0.86 | ~60–220 |
| **Ours (compose + neighbor-2opt)** | **~0.88** | **~240–690** |

---

## Building the method: cost engineering

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
- **Batching:** clusters within a level are independent (the parallel "batch
  dimension", as in an LLM). One padded forward pass replaces a Python loop of
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
| 1000000 | BHH | compose | 14.0 | 0.877 | 3159.1 | 3600 | 3.0× | 3.16 ms |
| 1000000 | BHH | SFC+n2o | 15.3 | 0.868 | 1057.8 | 1219 | | 1.06 ms |

### Patterns

1. **Quality crossover at ≈ n=1000, then a stable edge.** Composition's advantage
   `Δp = p_compose − p_SFC` goes −0.035 (n=100) → +0.008 (n=1000) → +0.016 (n=10k)
   → +0.012 (n=100k). It rises through the crossover and **plateaus at ~+0.013**.
   Composition's `p` is scale-invariant (~0.877 flat); the SFC baseline's `p`
   erodes (0.912 → 0.865). **Composition's product is scale-stability**, not
   raw quality: a space-filling-curve tour degrades as the problem grows; a
   recursive-composition tour holds. Below the crossover, composition is strictly
   worse (worse quality and worse compute).

2. **Compute penalty amortizes and stabilizes: 93× → 22× → 4× → 3×.** Composition
   carries a fat fixed overhead (torch + k-d-tree, ~0.3 s) that dominates at small
   n and washes out by n≥10k, settling at ~3× the cheap baseline.

3. **`compute/city` is U-shaped: the cleanup is super-linear.** It bottoms out at
   **n≈10k** (0.21 ms/city) then rises to 0.6 ms/city at 100k. Going 10k→100k
   (10× cities), compose's wall went 2.1 s → 60 s (**28×, not 10×**). The
   construction (partition + recursive batched solves + stitch) is near-linear;
   the **`neighbor_two_opt` cleanup is not**: pure-Python 2-opt with array
   segment-reversals is super-linear because reversal length grows with n. This is
   an implementation wall (an Or-opt / linked-list 2-opt would restore near-
   linearity), not a fundamental one.

---

## The million-city run

<!-- BEGIN 1M RESULT -->
The n=1,000,000 run completed (single instance, BHH reference). Full pipeline
breakdown:

| pipeline | stage | gap | wall | p |
|---|---|---|---|---|
| `space_filling` | raw Hilbert, no cleanup | 37.6% | 0.65 s | 0.727 |
| `compose:model:k10:none` | **construction only** (model + recursion + batched solves + stitch) | 52.2% | **136 s** | 0.657 |
| `SFC+neighbor2opt` | Hilbert + cleanup | 15.3% | 1058 s | 0.868 |
| `compose:model:k10:neighbor_2opt` | construction + cleanup | **14.0%** | 3159 s | **0.877** |

**The tiny model solves a million cities in 2.3 minutes.** The 2.8 MB
model, recursing ~6 levels deep and never solving more than ~10 cities at once,
produces a complete, valid million-city tour (`compose:…:none`) in **136 s**. The
classical `neighbor_2opt` polish that follows drags it from 52% → 14%, but costs
**~3020 s (50 min), 96% of the wall.** Construction is 4% of the time; cleanup is
96%. **The model was never the bottleneck; the 2-opt is.**

Two sub-findings the 1M point sharpens:

- **Raw composition (52.2%) is worse than a raw Hilbert curve (37.6%).** Un-cleaned,
  the cluster seams are bad. Composition's entire value is post-cleanup,
  where it edges SFC (14.0% vs 15.3%; p 0.877 vs 0.868, the same scale-stable
  ~+0.01 margin seen at every scale ≥ n=1000).
- **The cleanup scales as ≈ n^1.72.** Compose's wall rose **52.6×** for 10× the
  cities (100k → 1M), steeper than the ~n^1.4 the 10k→100k step implied; the
  reversal cost compounds as tours grow. This is the entire case for a reversal-free
  Or-opt / compiled kernel: the fast part (the model) finished in 2 minutes; the
  slow part is a pure-Python array-reversal 2-opt scaling as n^1.72.
<!-- END 1M RESULT -->

---

## Distance from the frontier

On static uniform Euclidean TSP:

- **p ≈ 0.88; the reachable near-linear frontier is ~0.95** (LKH-class ~0.995).
  That's ~2.5–3× too much slack (14% vs ~5% gap): a real engineering gap, and
  not one the model closes (the k-cap sweep showed final quality is set
  by the cleanup, not the local solver or cluster size).
- **f ≈ 150× off the frontier.** Decomposed: **~40× is pure Python→C** (same
  algorithm, terrible constants; recoverable with a numba/C kernel), and **~4× is
  the model + composition overhead failing to buy proportional accuracy** on this
  problem.

**On this problem, a good classical near-linear solver dominates our stack on both
axes, and the learned model is a net negative on `f`.**

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

- **Dynamic re-optimization**: re-solve one affected cluster on a change while
  classical methods recompute from scratch. This is where `f` should invert in our
  favor, and it is the next experiment.
- **Constrained / non-Euclidean variants** (time windows, asymmetric cost matrices)
  where classical heuristics need per-variant re-engineering but a learned solver
  might transfer. Needs the edge-feature model (deferred from Phase 1).

## Immediate next steps

- ~~Reversal-free cleanup~~ **DONE** (see "Kernel rewrite" below).
- **The dynamic experiment**: the frontier that this approach is actually built for.

---

## Kernel rewrite: compiled 2-opt + reversal-free Or-opt

The frontier-distance section named two fixable gaps in the cleanup: a **~40×
constant** (pure Python vs compiled) and a **super-linear scaling** (~n^1.72, from
array-reversal 2-opt). `lnhm/analysis/fast_local_search.py` addresses both:

- **Compiled 2-opt** (numba `@njit`): the same neighbor-list 2-opt with don't-look
  bits and shorter-segment reversal; kills the ~40× interpreter constant.
- **Reversal-free Or-opt** (doubly-linked list): relocates runs of 1–3 cities next
  to a spatial neighbor with O(1) relinking and no segment reversal; near-linear,
  and it finds improving moves plain 2-opt cannot, which lifts quality.

The two alternate to a joint local optimum. Wired into `compose_solve` as
`cleanup="fast_local"`.

### Methodology

Identical protocol to the p/f sweep above: one random instance per scale (seed 0);
reference = exact Held-Karp (n≤12), LKH (n≤2000), BHH `0.7124·√n` (n>2000);
`p = 1/(1+gap)`. The only change is the cleanup local search
(`neighbor_two_opt` → `fast_local_search`, k=8 neighbors, Or-opt segments ≤3). Numba
JIT is warmed before timing. **f is on the corrected CPU-core-seconds basis**
(`time.process_time()`, contention-robust); wall-clock is reported alongside for
latency. Two methods: `cmp+fast` (composition construction + fast cleanup) and
`SFC+fast` (Hilbert + fast cleanup); for compose rows, construction and cleanup are
timed separately.

### Results (fast kernel, cpu-seconds basis)

| n | ref | method | gap% | **p** | cpu_s | wall_s | **f** |
|---|---|---|---|---|---|---|---|
| 10 | exact | cmp+fast | 0.0 | 1.000 | 0.04 | 0.01 | 3699 |
| 10 | exact | SFC+fast | 0.0 | 1.000 | 0.01 | 0.00 | 1285 |
| 100 | LKH | cmp+fast | 2.1 | 0.980 | 0.10 | 0.03 | 1029 |
| 100 | LKH | SFC+fast | 1.4 | 0.986 | 0.00 | 0.00 | 41 |
| 1000 | LKH | cmp+fast | 5.3 | 0.949 | 0.24 | 0.14 | 258 |
| 1000 | LKH | SFC+fast | 6.8 | 0.936 | 0.00 | 0.00 | 4 |
| 100000 | BHH | cmp+fast | 7.2 | 0.933 | 23.22 | 15.25 | 249 |
| 100000 | BHH | SFC+fast | 8.4 | 0.923 | 0.58 | 0.48 | 6 |
| 1000000 | BHH | cmp+fast | 6.9 | 0.936 | 278.76 | 199.08 | 298 |
| 1000000 | BHH | SFC+fast | 7.8 | 0.928 | 27.71 | 25.80 | 30 |

(1M `cmp+fast` split: construction cpu 225 s / wall 147 s; **cleanup cpu 54 s /
wall 52 s**, versus the old pure-Python cleanup's ~3020 s. The million-city polish
went from ~50 min to ~52 s, ~58×.)

### What the rewrite bought

1. **Quality (p) jumped ~0.88 → 0.93–0.94 at scale, and to near-optimal at small n**
   (p 1.000 at n=10, 0.980 at n=100). The gap roughly **halved** at every scale
   (100k: 14.1% → 7.2%). The Or-opt moves are the lift: this is the ~5–8% regime a
   good classical local search reaches, up from the old ~13–15%.
2. **The cleanup stopped being the bottleneck.** At 100k the fast cleanup is
   **0.8 cpu-s vs the old ~60 s** (~75×) at better quality. The super-linear
   50-minute million-city polish is gone (see 1M rows).
3. **The model construction is now the dominant cost.** At 100k, `cmp+fast` spends
   22.4 cpu-s in construction and 0.8 in cleanup; construction is ~96% of compute.

### What the rewrite sharpened

- **On static TSP the verdict got stronger, not weaker.** With a good cleanup,
  `cmp+fast` (p 0.933) beats `SFC+fast` (p 0.923) by the same scale-stable ~+0.01,
  but now costs **~40× more on the cpu basis at 100k (f 249 vs 6), ~10× at 1M
  (298 vs 30)**. The penalty shrinks as construction amortizes, but `SFC+fast`
  dominates at every scale, because the cleanup no longer equalizes them and the
  model construction's parallel cost is now fully charged. The learned model still
  does not earn its keep on static uniform TSP.
- **Residual super-linearity.** The 2-opt half still uses array reversals, so it is
  ~40× cheaper but still super-linear; Or-opt is the near-linear part. Fully
  flattening 2-opt needs a two-level doubly-linked list (O(√n) moves), a further
  step not required at these scales.
- **We approach but do not reach the ~0.95 frontier** (p ~0.93, gap 7–8% at scale).
  Closing the rest needs more neighbors (k) or Or-3opt; diminishing returns.

### Before → after (quality, p)

| n | compose: old → new | SFC: old → new |
|---|---|---|
| 100 | 0.877 → **0.980** | 0.912 → **0.986** |
| 1000 | 0.893 → **0.949** | 0.885 → **0.936** |
| 100000 | 0.877 → **0.933** | 0.865 → **0.923** |
| 1000000 | 0.877 → **0.936** | 0.868 → **0.928** |
