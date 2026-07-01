# LNHM: a solver-factory experiment on Euclidean TSP

LNHM ("Large NP-Hard Model") began as a bet: the structural principles that
produce good solutions to small instances of an NP-hard problem are the same
principles at large instances, so you can bootstrap a neural solver from cheap,
exactly-solvable examples toward sizes no exact solver reaches. This repo is
what happened when that bet was tested with pre-registered predictions,
adversarial controls, and a standing rule that every claim gets a kill-test
before it gets believed.

The work ran in four phases over two days on a home lab (one CPU box, one
consumer GPU). Most of the priors died. What survived is more useful than what
was hoped for, because it comes with a mechanism.

## The findings

| # | Claim | Verdict | Where |
|---|-------|---------|-------|
| 1 | Cross-level transfer exists (training on larger n helps smaller n) | Real, modest (+3-5pp), sign-consistent across 30 controlled runs | [phase0/](phase0/) |
| 2 | A 710K-param model can solve million-city instances by recursive composition | Mechanism works: valid 1M-city tours, construction in ~2.5 min | [phase1/](phase1/) |
| 3 | ...and that composition is economically competitive on static TSP | No. Classical near-linear pipelines dominate on both quality and cost | [phase1/](phase1/) |
| 4 | The model contributes to composition quality | Inert. A random-permutation leaf reaches the same post-cleanup tour; so does an exact Held-Karp leaf | [phase3/](phase3/) T1 |
| 5 | Training deeper (n 3-20) beats training on more data | Depth wins decisively; data volume is inert; no interaction | [phase2/](phase2/) |
| 6 | Depth extrapolation is real model skill, not sampling luck | Passes its pre-registered control: one-shot greedy p=0.957 at n=30, 1.5x the training ceiling | [phase3/](phase3/) T2 |
| 7 | Geometric diversity in training data is free | Yes: closes the out-of-distribution geometry penalty from 3.6pp to 0.9pp with zero regression on uniform | [phase3/](phase3/) Test 2 |
| 8 | Best-of-k sampling is a safe evaluation basis | No. It inverts below greedy exactly where the model is OOD, and inflated our own headline 3.5x | [phase3/](phase3/) T2 |

## The mechanism, in three sentences

Strong local search erases constructor quality: on Euclidean TSP, a
fifty-year-old 2-opt/Or-opt pass drives a random starting tour, a
nearest-neighbor tour, a learned model's tour, and an exact solver's tour to
the same local optimum, so nothing upstream of the cleanup can earn its
compute. That is a statement about this problem's improvement landscape, which
is unusually saturated with cheap improving moves, and not about the
composition mechanism, which did exactly what it was designed to do. The
surviving thesis is the contrapositive: learned constructors can only earn
their keep on problems whose landscapes lack cheap improving moves, which is
where this program would go next.

## What positively survived

- **Depth extrapolation.** Train a curriculum to ceiling N and the model
  predicts near-frontier tours out to ~1.5N in a single greedy pass. Survived
  an adversarial decode-mode control at pre-registered thresholds.
- **The diversity free lunch.** Mixing clustered and irregular geometries into
  training closes the OOD-geometry penalty and slightly improves uniform
  extrapolation, at zero quality cost and ~45% extra convergence time.
- **A free OOD detector.** The sign of (best-of-k minus greedy) flags whether
  the model considers an input in-distribution, with no knowledge of the
  training range. See [HIGHLIGHTS.md](HIGHLIGHTS.md).
- **The oracle-row ablation.** Including an exact solver as one arm of the
  leaf-solver ablation answers the "try a bigger model" objection before it is
  raised: a perfect leaf is equally inert.
- **A compiled near-linear local search** (`lnhm/analysis/fast_local_search.py`):
  numba 2-opt plus reversal-free Or-opt, ~58x faster than the naive cleanup at
  1M cities and better quality.

## Repo map

```
phase0/   spec + run log: curriculum learning, cross-level transfer (A/B/C/D controls)
phase1/   spec + results: recursive composition to 1M cities, p/f frontier metrics
phase2/   spec + results: depth x data 2x2, extrapolation discovery
phase3/   results + kill-test suite (T1-T8): the adversarial phase
lnhm/     the code: model, training, data generation, analysis, composition
```

Reproduction: see [lnhm/README.md](lnhm/README.md) for environment and
commands, and [phase3/HARNESS.md](phase3/HARNESS.md) for the eval harness.
Every training instance is deterministic in (base_seed, level, split, index),
so datasets regenerate exactly. Pre-registered predictions are committed before
the results that scored them; the git history is the provenance.

## Status and scope

Tested: Euclidean TSP, uniform and clustered geometries, models up to 2.8M
params (a 4x capacity probe changed no verdict: greedy extrapolation moved by
less than 0.003, so the one-shot results are properties of the training
distribution, not parameter count; extra capacity only sped in-range
convergence and improved OOD sampling calibration), instance sizes to n=30
one-shot and n=1M by composition.
Not tested: the extrapolation-ratio law (does train-to-N cover 1.5N at every
N?), and variant transfer on problems without cheap local-search moves, which
is where the factory thesis now lives or dies.
