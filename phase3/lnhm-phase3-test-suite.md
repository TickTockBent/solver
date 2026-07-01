# LNHM Phase 3 Test Suite — Adversarial Controls & The Untested Axes

Priority-ordered. T1 and T2 are "am I fooling myself" tests and cost almost
nothing; run them before anything else. T4 is the biggest scientific payoff.
T6 and T7 are where the economic case lives or dies.

Every test pre-registers its verdict rule before running. Same discipline as
Phase 0/2.

---

## T1 — Trivial local solver ablation (the null hypothesis test)

**Question:** Does the model contribute anything to the static composition
pipeline, or was the story always "good clustering + good cleanup"?

**Method:** Identical composition pipeline (same clustering, same stitch, same
`fast_local` cleanup), swapping only the local solver:

| local solver | role |
|---|---|
| random permutation | true null |
| nearest-neighbor within cluster | cheap floor |
| **model (current)** | the thing being tested |
| Held-Karp exact | ceiling (perfect local solves) |

Run at n ∈ {10k, 100k, 1M}, 3 seeds, uniform distribution (repeat on clustered
once T5 data exists). Report final p and f for each row, plus raw pre-cleanup gap.

**Pre-registered verdict:**
- If p(model) − p(NN) < 0.005 after cleanup → **the model is inert in static
  composition.** The static pipeline is clustering + cleanup, full stop. Retire
  the claim that the model earns anything here.
- Where the model lands on the NN↔HK span measures how much of the available
  local-quality headroom it captures — and whether cleanup erases it anyway.
- The k-sweep already hints the answer is "inert." Confirm or refute directly.

**Cost:** ~1–2 hours. Run first.

---

## T2 — Greedy decode table (prediction vs search)

**Question:** How much of the Phase 2 depth extrapolation is the *model* versus
best-of-16 *sampling*? Best-of-16 is search; the headline claim is about
prediction.

**Method:**
1. Rerun the Phase 2 eval with `--mode greedy` on all four cells, all n
   (5→30). Report p side-by-side with the sampled table.
2. For the depth model only, best-of-k curve at n ∈ {25, 30}:
   k ∈ {1, 2, 4, 8, 16}. Shows the marginal value of each doubling of samples.

**Pre-registered verdict:**
- Greedy depth model holds p ≥ 0.94 at n=30 → extrapolation is model-borne;
  the Phase 2 headline stands as written.
- Greedy collapses toward control → the finding becomes "depth training yields
  a better *sampling distribution*." Real, but a different (weaker) claim, and
  the one-shot latency story changes (16 passes, not 1).

**Cost:** Minutes. Pure inference. Run alongside T1.

---

## T3 — Compute-matched classical rows at n=16–30 (pricing the one-shot claim)

**Question:** "Near-frontier one-shot solver up to ~n=30" is a product claim.
Product claims need the f column. Is the model undominated against classical at
matched cost in this range?

**Method:** At n ∈ {16, 20, 25, 30}, 500 instances each, report p and
cpu-seconds for:

- model greedy (1 forward pass)
- model best-of-16
- nearest-neighbor + compiled 2-opt
- SFC + fast_local
- NN + fast_local (Or-opt included)
- LKH (reference, untimed)

Model rows charge amortized load cost honestly (note it separately — it matters
for cold-start phone deployment, not for a resident service).

**Pre-registered verdict:** The one-shot product claim survives only if a model
row occupies a Pareto-undominated point. If NN+fast_local matches p at lower
cost, the claim retires to "competitive but not differentiated."

**Cost:** An evening.

---

## T4 — Extrapolation-ratio law (the provisioning law)

**Question:** Does "train to N → near-frontier to ~1.5N" hold as N grows?
Multiplicative or additive? Where does it break? This is the single most
factory-relevant experiment available: if a stable law exists, it *is* the
provisioning function.

**Method:**
- Train ceilings N ∈ {12, 16, 20, 25, 30}. Ground truth: Held-Karp to 20,
  Concorde 21–30. 2 seeds per ceiling, mixed-distribution training (per T5).
- Evaluate greedy **and** best-of-16 on held-out sizes from N to 2N in steps.
- Define extrapolation reach **R(N) = max n such that p ≥ 0.95** (greedy basis;
  report sampled basis alongside).
- Fit both candidate laws and distinguish:
  - **Multiplicative:** R(N) ≈ c·N — factory implication: training cost to cover
    any target n is log-bounded via bootstrapped curricula.
  - **Additive:** R(N) ≈ N + c — depth is a fixed bonus; coverage of large n
    still requires training near it.

**Pre-registered verdict:** Whichever law fits with lower residuals across ≥4
ceilings wins. If neither fits (reach is erratic), the provisioning-law claim
is dead and depth is characterized empirically per-ceiling instead.

**Cost:** Data gen is the long pole (Concorde instances at 25–30). A weekend.

---

## T5 — Distribution robustness matrix (in flight — formalize it)

**Status:** Already discovered the uniform-only blind spot; mixed retraining in
progress. Formalize so the result is a matrix, not an anecdote.

**Method:**
- Train distributions: {uniform, clustered (Gaussian blobs, varied σ and count),
  mixed 50/25/25 uniform/clustered/bounded-random}.
- Eval distributions: all three **plus** two held-out: grid-perturbed, and small
  TSPLIB instances (free real-world-ish geometry).
- Report the full train×eval p matrix, greedy basis, at n ∈ {10, 20, 30}.

**Two pre-registered questions:**
1. **Interference cost:** does mixed training lose anything on uniform vs the
   uniform specialist? If Δp < 0.005, distribution robustness is free — make
   mixed the default forever.
2. **Fixed-k damage on realistic geometry:** run the *current* composition stack
   (fixed k=10) on clustered instances vs uniform at n ∈ {10k, 100k}. This
   measures what fixed-k actually loses on globby data **before** building
   elastic-k. If the loss is < 1pp of p, elastic-k isn't justified yet; if it's
   large, the elastic-k build is motivated by data instead of intuition.

**Cost:** Piggybacks on the retraining already running + one composition sweep.

---

## T6 — Dynamic re-optimization vs the STRONG baseline (pre-registered)

**The stacked-deck warning, made binding:** classical does not recompute from
scratch. Dynamic VRP's standard move is *incremental repair* — cheapest
insertion of the changed point plus localized Or-opt/2-opt around the change.
With our own compiled kernel that repair is nearly free. Beating a from-scratch
strawman means nothing.

**Protocol (mutation stream):**
- Start from a solved instance at n ∈ {1k, 10k}. Apply a stream of M=500
  mutations: 40% add point, 40% move point, 20% remove point, arrival times
  Poisson.
- After every mutation each method must return a valid updated tour before the
  next mutation arrives.
- Methods:
  - (a) **model cluster re-solve:** re-solve affected cluster(s) + seam repair
  - (b) **classical incremental repair:** cheapest insertion + localized
    fast_local around the change (STRONG baseline)
  - (c) full SFC+fast re-solve per mutation (naive baseline, for scale)
  - (d) hybrid: (b) per mutation + periodic full re-solve every 50 mutations
- Measure: per-update cpu-seconds (distribution, not just mean), steady-state p
  after M mutations (reference: LKH re-solve of the final instance), and quality
  drift curve over the stream.

**Pre-registered verdict:** The model approach must beat **(b)** — not (c) — on
the p-vs-per-update-cost frontier, or match its frontier position with a
meaningfully better drift curve. If it cannot, the dynamic case on Euclidean
instances is dead, stated plainly, and the economic argument moves entirely to
T7.

**Honest prior, stated up front:** (b) is very hard to beat. Design accordingly.

**Cost:** The harness is the work (~a day). Runs are fast.

---

## T7 — Constrained variant transfer (the moat hypothesis) — design sketch

**The actual bet the factory thesis lives or dies on.** Classical local search
dominates Euclidean TSP because fifty years of moves exploit metric geometry.
Break the geometry and every classical baseline needs re-engineering per
variant; a learned model with edge features might transfer with retraining
only.

**Prerequisite build:** edge-feature model — input is a distance *matrix* (or
edge feature tensor), not coordinates. This is the deferred Phase 1 item and is
now the critical path.

**First variant: asymmetric TSP** (d(i,j) ≠ d(j,i), random asymmetry factor).
Chosen because it breaks 2-opt directly — segment reversal changes tour cost
asymmetrically, so the workhorse classical move stops being valid as-is. This is
the cleanest test of "variant robustness is where learning wins."

**Sketch of the experiment (full spec after T1–T6):**
- Train edge-feature model on symmetric Euclidean; fine-tune on asymmetric
  (measure fine-tune cost).
- Classical baselines: NN + asymmetric-safe local search (Or-opt still works;
  2-opt needs the directed variant — implement honestly, this is the
  re-engineering cost being measured).
- Compare: p, f, **and engineering-hours per variant** — the third column is
  the moat metric.

Second variant candidates, in order: time windows, capacitated clusters,
pickup-and-delivery pairing.

---

## T8 — Data floor (cheap, piggyback)

Phase 2 showed 4k instances/level is not data-limited. Find the floor: 2k, 1k,
500 per level at ceiling N=20. If 1k suffices, Held-Karp/Concorde data-gen cost
for deep curricula (T4) drops proportionally. Attach to any retraining run.

---

## Kill criteria summary

| test | if it fails | what dies |
|---|---|---|
| T1 | model ≈ NN after cleanup | model's role in static composition |
| T2 | greedy collapses to control | "prediction" framing of depth extrapolation |
| T3 | model dominated at n≤30 | one-shot product claim on Euclidean |
| T4 | no stable law | provisioning-law framing of the factory |
| T6 | can't beat incremental repair | dynamic case on Euclidean |
| T7 | no transfer advantage | **the factory thesis itself** |

Note what does NOT die in any single failure: the methodology, the p/f ruler,
the depth-extrapolation finding as science, and the compiled-kernel work — those
stand regardless. But if T1, T6, and T7 all fail, the honest conclusion is that
this was a first-rate measurement project about where learned solvers *don't*
help, and that conclusion gets written up with the same rigor as a win.
