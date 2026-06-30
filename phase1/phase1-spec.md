# LNHM Phase 1 Technical Specification — Composition

## Overview

Phase 0 asked "does cross-level reinforcement exist." Phase 1 asks the question
the entire application vision rests on: **can we solve a large instance by
decomposing it into small ones, solving those with a cheap model, and stitching
the pieces back together — with bounded, measurable fidelity loss?**

If yes, model size stops gating *max-n* and starts gating *per-cluster fidelity*:
you can return *some* tour at almost any n, and a single scalar (the cluster cap
`k`) becomes the price/quality dial. If no, the vision collapses to monolithic
solvers and we learn that cheaply.

This is a **reprioritization** of the original spec, which made Phase 1
"REINFORCE beyond the brute-force frontier." Composition demotes that: if you
only ever train on small clusters, you never leave the brute-force-feasible
regime, so the no-ground-truth-at-scale problem REINFORCE was meant to solve
largely evaporates. REINFORCE becomes an optional *comparison baseline* (Phase
1d), not the main event.

## Entry gate

Phase 0's cross-level effect confirmed via the full A/B/C/D matrix (anchors
{5 neg-control, 9, 11}, 5 seeds, constant LR). The n=9 spot-check is a promising
hint (A > D by ~8pp, de-confounded); the full matrix is the gate. If the effect
is absent once confounds are controlled, revisit before investing here.

## Scope

**Euclidean TSP only.** Hold everything else fixed to isolate the composition
question. Explicitly **deferred to Phase 2** (see end): the size↔capacity scaling
law and cross-level effects at higher n with bigger models; the edge-feature
model (consuming real, asymmetric, traffic-weighted distance matrices); and
constrained VRP / time windows / the dynamic last-mile (DoorDash-sibling) class.

## The composition method

A large instance is solved in four stages, applied recursively:

1. **Partition** the points into clusters of size ≤ k.
2. **Local solve** each cluster (the Phase 0 model, or a cheap heuristic).
3. **Stitch** the sub-tours into one tour at the level above — connecting clusters
   by their *ports* (where the tour enters/leaves), not their centroids.
4. **Cleanup** (optional) — a local-search pass to smooth the boundary seams.

Recursion: the stitch step is itself a TSP over cluster representatives, so **the
same Phase 0 model can serve as both the local solver and the stitcher**, applied
at each level. Likely the only genuinely new code is the partitioner, the
port/representative extraction + splice logic, and the cleanup pass.

Cost accounting (target): O(n) total local solves (geometric series over levels),
O(n log n) routing/partition passes, recursion depth O(log_k n). Near-linear, with
the depth parallelizable within each level.

## The experiment: a building-block matrix

The point is not one pipeline but to **map the cost/accuracy frontier across
compositions of cheap building blocks** and find the undominated points.

| Stage | Options to test |
|-------|-----------------|
| Partition | space-filling curve (Hilbert), k-means, grid |
| Local solve | LNHM model @ cap k, nearest-neighbor, (2-opt) |
| Stitch | space-filling-curve order, LNHM model recursively, greedy port-join |
| Cleanup | none, **2-opt on seams only**, 2-opt full |

Key compositions to evaluate (incl. the hybrids raised in review):
- **model-only** (partition → model solve → model/greedy stitch, no cleanup).
- **model + seam-2-opt** — model for fast global structure, a single 2-opt pass
  targeting *only* the cluster boundaries (error concentrates there → most of the
  cleanup at a fraction of full-2-opt cost). Primary hybrid to beat.
- **hilbert + 2-opt** — the cheap classical reference pipeline.
- **hilbert backbone + model refinement** — use the space-filling curve as the
  partition/stitch ordering (cheap, O(n log n)) and let the model optimize within
  clusters and at seams. Isolates "how much does learning add on top of the cheap
  heuristic" and gives a graceful floor (worst case ≈ Hilbert + a little).

Sweep cluster cap `k ∈ {8, 10, 12}` to trace the price/fidelity dial.

## Baselines and the bar

Measure gap against a reference, not just each other:
- **Reference optimum:** Concorde (exact) on a sample where feasible (n ≤ ~1000);
  **LKH-3** as near-optimal reference at larger n.
- **Frontier competitors:** nearest-neighbor (~25% over), **space-filling curve**
  (~25%, O(n log n) — the cheap near-linear competitor, the honest "cheap big
  solve" bar), 2-opt (~5%, pricier).

**The bar is an *undominated point*, not "beat the optimum."** A win is: better
quality than Hilbert at Hilbert-ish cost, OR cheaper than 2-opt at 2-opt-ish
quality. A clean negative (composition ≈ Hilbert at matched cost) is still data —
it relocates the value entirely to dynamics + generality (Phase 2+).

## Metrics

- **Optimality/near-optimality gap** vs Concorde/LKH at n ∈ {50, 100, 500, 1000}.
- **Compute cost** per pipeline (wall + an op-count proxy) — the x-axis of the dial.
- **Per-level compounding:** how much gap is added at each recursion level (the
  thing that decides whether "good enough at any n" holds).
- **Fidelity-vs-k curve:** gap as a function of cluster cap, per pipeline.

## Two gates

1. **Does composition work?** Valid tours; bounded gap vs LKH; loss does not blow
   up with recursion depth. Clear this before racing the frontier.
2. **Does it beat the cheap-heuristic frontier at matched cost?** An undominated
   point on the cost/quality plane.

## Experimental discipline (carried from Phase 0)

- **Anchor the cluster cap in the solver's strong zone** (start k=8, gap <1%; then
  10, 12). If composition fails at k=12 (the wall), you can't tell whether
  *composition* or the *local solver* broke. Isolate one variable.
- **Constant LR** for any controlled training comparison (the cosine schedule
  confounded A vs D in Phase 0).
- A well-trained local solver at the chosen k (a short focused train, or the
  Phase 0 checkpoint) so failures aren't weak-solver artifacts.

## Compute plan — local GPU is sufficient

A decent GPU (4090/24 GB) runs all of this comfortably; **no offload needed.**

| Item | Estimate (local GPU) |
|------|----------------------|
| Full A/B/C/D matrix (entry gate) | ~15–20 min |
| Composition matrix (thousands of tiny solves batched + LKH baselines) | ~1–2 hr |
| One-time test-set generation (n=50–1000) + LKH/Concorde references | hours (CPU, one-time) |

Offload (FlightDeck) becomes relevant only at **Phase 2** — constrained VRP on
large real datasets, multi-seed production sweeps, the edge-feature model.

## New code

- `analysis/baselines.py` — nearest-neighbor, 2-opt (full + seam-restricted),
  space-filling-curve heuristic, LKH-3 wrapper (build the C binary), optional
  Concorde wrapper.
- `analysis/compose.py` — partitioners, recursive solve/stitch, port extraction,
  cleanup; the building-block matrix runner; cost + gap measurement.
- A focused local-solver training at cap k (reuse `training/train.py`).
- Test-set generator for large n (reuse `data/generate.py`; references via LKH).

Everything else (model, training, eval, data, Held-Karp) already exists.

## Decision log

| Outcome | Meaning | Next |
|---------|---------|------|
| Composition works, gap bounded, smooth dial | vision GO | Phase 2: scaling law, edge-features, constrained/dynamic |
| Composition works but only ≈ Hilbert at matched cost | static edge nil | pivot value to dynamics + generality; still build Phase 2 |
| Composition compounds badly with depth | composition fails | fall back to monolithic RL (orig. Phase 1) or rethink |
| model + seam-2-opt is the undominated point | strong result | make it the reference pipeline |

## Deferred to Phase 2 (per the agreed ordering)

1. **Capacity / scaling law** — train model sizes, fit iso-fidelity contours
   `n*(τ, P)`; verify cross-level effects at higher n with bigger models and
   characterize the falloff-sigmoid shape vs size (shift-and-soften vs
   shift-and-sharpen). Sets the cost-optimal `k` per fidelity.
2. **Edge-feature model** — consume distance/cost matrices (asymmetric,
   traffic-weighted) instead of coordinates; the deployable architecture for real
   road networks.
3. **Constrained / dynamic** — time windows, pickup-and-delivery, real-time local
   re-optimization; the DoorDash-sibling NP-hard class.
