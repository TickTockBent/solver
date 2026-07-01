# LNHM Phase 0 Technical Specification

**LNHM — Large NP-Hard Model.** A method (a "solver factory") for producing a bespoke neural solver per NP-hard problem class, trained by curriculum bootstrapping from small instances (cheap exact ground truth) toward large ones. The central per-class bet is **scale-unification**: the structural principles that produce good solutions at small n are the same principles at large n, just more constrained, so training on harder instances retroactively improves performance on easier ones.

Phase 0 is the smallest possible test of that bet, on the friendliest possible class: **Euclidean TSP**. It does not aim to produce a good TSP solver. It aims to make one measurement, in a setting where nothing else confounds it: does cross-level reinforcement (backward transfer across instance sizes) exist?

> **Why TSP is the right warm-up.** Euclidean TSP is rigged in our favor on every axis: Beardwood–Halton–Hammersley gives theoretical grounds for scale-coherence (optimal tour length through n uniform points → β√n), the solution object is a fixed-shape permutation at every n, and exact ground truth is cheap below n≈20. If cross-level reinforcement does not appear here, it appears nowhere. A positive result here is therefore *necessary but not sufficient* evidence for the wider thesis.

---

## System Overview

A training pipeline that generates exact solutions to TSP instances at increasing sizes, trains a neural model against them using curriculum learning, and measures whether training on larger instances improves accuracy on smaller ones.

---

## Data Generation

### Instance Format

Each TSP instance is a set of 2D coordinates normalized to [0, 1] × [0, 1] and a known-optimal tour.

```
{
  "id": "uuid",
  "n": 5,
  "coords": [[0.23, 0.71], [0.85, 0.14], ...],
  "optimal_tour": [0, 3, 1, 4, 2],
  "optimal_distance": 2.847
}
```

The `optimal_tour` is stored in **canonical form** (see Training → Target Canonicalization): rotated to start at city 0 and oriented so the second city index is less than the last.

### Level Semantics — the degeneracy floor

The number of distinct undirected Hamiltonian cycles for n cities is **(n−1)!/2**:

| n | distinct tours | meaning for Phase 0 |
|---|---|---|
| 3 | **1** | degenerate — every output is optimal; accuracy ≡ 100% for any model |
| 4 | 3 | near-degenerate — random tour is optimal ~⅓ of the time |
| 5 | 12 | first level with real combinatorial content (random-tour baseline ≈ 8%) |
| 6 | 60 | random-tour baseline ≈ 1.7% |
| 7 | 360 | random-tour baseline ≈ 0.3% |

n=3 and n=4 are therefore **warm-up levels only**. They seed the model but carry no measurable signal: you cannot detect a 5pp accuracy gap on a metric that saturates at 100%. All backward-transfer measurement is **anchored at n≥5** (see Cross-Level Reinforcement Test). This corrects the original draft, which anchored the headline experiment and the "33% random baseline" success criterion on n=3.

### Exact Solver

**Use Held–Karp dynamic programming (O(n²·2ⁿ)) as the single solver for all of Phase 0 (n ≤ 12).** It is exact and runs in well under 10 ms per instance through n=12 (n=12: ~12·4096 DP states, single-digit milliseconds; memory O(n·2ⁿ) is trivial). Full enumeration of (n−1)!/2 tours is retained only as an **independent cross-check** on a small sample at n ≤ 8, to validate the Held–Karp implementation.

> **Correction from the original draft.** The original drew the enumeration→Held–Karp line at n=12/13 and estimated ~4 min/instance for n=12 enumeration. At 50,000 instances that is ~3,300 CPU-hours for n=12 alone; the data-generation cost would have dwarfed the entire "weekend of GPU time" training budget. Held–Karp from the start collapses this: the whole Phase 0 dataset generates in **minutes to ~an hour on a single multicore machine**. Enumeration is microseconds-vs-milliseconds slower and only worth keeping as a correctness oracle at tiny n.

**Beyond Phase 0 (out of scope here):** n=13–20 still uses Held–Karp (feasible to ~20 nodes with 32GB RAM); n>20 uses Concorde as an exact oracle. The Redis/Kubernetes distributed-generation machinery belongs to **Phase 1+** and is **not required for Phase 0**; Phase 0 needs none of it.

### Instance Generation Targets (Phase 0: levels 3–12)

| n | Training instances | Validation instances | Held–Karp time / instance |
|---|-------------------|---------------------|---------------------------|
| 3 | 2,000 (warm-up) | 500 (warm-up) | <0.01 ms (degenerate) |
| 4 | 2,000 (warm-up) | 500 (warm-up) | <0.01 ms |
| 5 | 10,000 | 2,000 | <0.1 ms |
| 6 | 10,000 | 2,000 | <0.1 ms |
| 7 | 10,000 | 2,000 | <0.1 ms |
| 8 | 50,000 | 5,000 | <0.5 ms |
| 9 | 50,000 | 5,000 | <0.5 ms |
| 10 | 50,000 | 5,000 | ~1 ms |
| 11 | 50,000 | 5,000 | ~2 ms |
| 12 | 50,000 | 5,000 | ~4 ms |

These are starting points. Adjust based on observed generalization — if the model memorizes at a given n, increase instance count. Note that the cross-level control (Model C, below) requires the ability to generate **fresh, unique** instances at the anchor level on demand, so treat the anchor-level generator as a callable rather than a fixed file.

### Generation Infrastructure (Phase 0)

Brute force is embarrassingly parallel; each instance is independent.

- **All of Phase 0 (n ≤ 12):** Python `multiprocessing` across available cores on a single machine. A Beelink/8-core box handles the full dataset in minutes to ~an hour. No queue, no cluster, no Concorde.

### Coordinate Generation

Uniform random in [0, 1]². This is the standard benchmark distribution and the most stationary one, the friendliest case for a scale-unification test.

> **Distribution caveat (load-bearing for the wider thesis).** A solver is only ever rated against an assumed input distribution. Phase 0's uniform-random rating says nothing about robustness to distribution drift (clustered, grid-perturbed, adversarial). Testing non-uniform distributions is a deliberate **Phase 1+** concern; Phase 0 holds distribution fixed so it does not confound the size-transfer measurement.

### Storage

JSONL files per level, gzipped. One file per level. Total storage for Phase 0 is well under 1 GB.

---

## Model Architecture

### Primary Candidate: Transformer

Based on the Attention Model (Kool et al. 2019) pattern, simplified.

**Encoder:**
- Input: n × 2 coordinate matrix
- Linear projection to d_model dimensions
- N_enc transformer encoder layers with multi-head self-attention
- Output: n × d_model node embeddings

**Decoder:**
- Autoregressive. At each step, select the next city to visit.
- Context: embedding of the last selected city, embedding of the first city (for return-trip awareness), mean of remaining unvisited embeddings.
- Attention over unvisited node embeddings to produce selection logits.
- Mask already-visited nodes to −inf before softmax.
- Output: a permutation of node indices.

> **Where the claim lives.** The decoder is a routing-specific head: "select the next city" has no analogue in SAT or graph coloring and will never transfer across problem classes. The transferable, scale-unified structure the LNHM thesis cares about must live in the **encoder's representation**. Phase 0 should therefore interpret a positive cross-level result as encoder-borne structural transfer, and the optional localization probe (see Cross-Level Reinforcement Test) exists to confirm that rather than assume it.

**Starting hyperparameters for Phase 0:**

| Parameter | Value |
|-----------|-------|
| d_model | 128 |
| N_enc (encoder layers) | 3 |
| Attention heads | 8 |
| Feedforward dim | 512 |
| Dropout | 0.1 |

Intentionally small. Phase 0 validates cross-level reinforcement, not solution quality. Scale up later if the hypothesis holds.

### Alternative: Graph Neural Network

If the transformer struggles, try a GNN variant (GatedGCN or GraphSAGE) with edge features encoding pairwise distances. Same autoregressive decoder. Don't build both upfront. Start with the transformer; switch only if Phase 0 results are ambiguous and architecture might be the confound.

---

## Training

### Loss Function

**Supervised (Phase 0):** Cross-entropy between predicted tour probabilities and the (canonicalized) optimal tour at each decoding step. Teacher forcing during training: feed the correct next city at each step, train the model to predict it.

Optimal tours aren't unique; multiple tours can share the same distance. **Evaluation** accepts any tour within 1% of the known optimum (distance is rotation- and reflection-invariant). **Training** uses the canonical target so the loss doesn't fight itself (next item).

### Target Canonicalization

Teacher-forcing cross-entropy is computed against a *specific index sequence*, so it would otherwise penalize the model for emitting a valid rotation or reflection of the optimal cycle. Canonicalize every target before computing loss:

1. **Fix the start:** rotate the tour so it begins at city index 0.
2. **Fix the direction:** of the two orientations, keep the one whose second element is the smaller index.

This yields a unique representative per undirected cycle. The decoder's "first city" context is consistent with always starting at city 0. (If distance ties produce two genuinely distinct optimal cycles for the same instance — rare at small n with continuous coordinates — pick the lexicographically smallest canonical form deterministically.)

### Optimizer

Adam. lr = 1e-4. Cosine annealing with warm restarts, cycle length tuned to curriculum transitions.

### Batch Construction

Each batch contains instances from multiple levels, mixed per the curriculum schedule. Do **not** segregate batches by level. Batch size: 256–512.

Handle mixed n within a batch by **padding** smaller instances to the batch's max n with dummy nodes masked out in attention. (Separate forward passes per n, aggregating gradients, is the fallback if padding distorts results.) Padding is simpler; start there.

### Curriculum Schedule

**Phase 0 curriculum (additive):**

1. **Warm-up (n=3, then n=4):** seed the model. Graduation at these levels is trivial/automatic (n=3 is always 100%); they exist to initialize, not to measure.
2. Add **n=5** — the first level with real combinatorial content. Train until n=5 validation accuracy ≥ 50%.
3. Add **n=6**; new frontier level takes the plurality of the distribution (40%), previous levels split the remainder (60%) equally.
4. Continue level by level through n=12. Each new level gets the plurality; previous levels share the remainder equally.

Record validation accuracy on **all** previous levels after every epoch. This is the primary experimental data.

**Graduation threshold:** 50% accuracy on novel instances at the current frontier level (n≥5) before adding the next. Deliberately low: the goal is to detect cross-level reinforcement, not to perfect each level.

**Accuracy definition:** a solution is "correct" if its total tour distance is within 1% of the known optimum. For small n this is nearly exact match but avoids penalizing optimal tours in a different rotation/direction.

### Training Duration

Phase 0 should not exceed a weekend of GPU time on a single consumer GPU (RTX 3090/4090 class). Target ~100–200 epochs per curriculum level, ~1000–2000 epochs total through n=12. If it takes longer, the model is too large or training is inefficient.

---

## Evaluation

### Metrics (Per Level, Every Epoch)

1. **Accuracy:** % of validation instances within 1% of optimum.
2. **Mean optimality gap:** average (predicted_distance − optimal_distance) / optimal_distance.
3. **Worst-case gap:** maximum optimality gap in the validation set.
4. **Accuracy by level over time:** the key chart. X = epoch, Y = accuracy, one line per level. This is what Phase 0 lives or dies on.

### Cross-Level Reinforcement Test

The experiment, redesigned to (a) anchor on a non-degenerate level and (b) separate genuine size-transfer from data-diversity regularization.

**Anchor levels:** a ∈ {5, 6, 7}. n=5 is the primary anchor; 6 and 7 are robustness checks. (Anchoring on 3 or 4 is invalid — see Level Semantics.)

For a given anchor a and a curriculum range [3, N] with N strictly greater than a (e.g. a=5, N∈{7,10,12}):

- **Model A (treatment):** trained on the full curriculum [3, N]. Record accuracy at level a.
- **Model B (naive control):** trained on level a **only**, same total compute (gradient steps) and same fixed dataset size as A allocated to level a.
- **Model C (regularization control — NEW):** trained on level a **only**, same total compute, but with **fresh, unique instances every step** (effectively unbounded anchor-level data, no repetition).

**Reading the result:**
- **A > B** alone is weak: B sees a fixed dataset many times and will overfit, so part of A's edge could be mere regularization from data diversity rather than transfer from larger instances.
- **A > C** is the load-bearing comparison. C already enjoys unlimited same-size diversity, so if A still beats C, the surplus must come from structure learned at larger n: genuine cross-level reinforcement. If C closes the gap to A, the effect was diversity/regularization, not size transfer; that deflationary finding is worth recording in its own right.

Run for anchors {5, 6, 7} × ranges N ∈ {7, 10, 12}, **≥5 random seeds per configuration**.

**Minimum effect size for "success":** Model A's anchor-level accuracy exceeds **Model C's** by ≥ 5 percentage points, consistent across seeds.

**Optional localization probe (recommended):** to confirm the transfer is *encoder-borne*, take A's trained encoder, freeze it, and train a fresh decoder on anchor level a only; compare against a from-scratch model at a. If the frozen-encoder model wins, the scale-unified structure lives in the representation (the result that matters for the wider thesis) rather than in the routing head.

### Baseline Comparisons (Phase 1+)

At each level with ground truth: nearest-neighbor (greedy), 2-opt, LKH-3 (SOTA heuristic), Concorde (exact). The model needn't beat LKH-3 in Phase 0; it needs to show cross-level reinforcement. Beating baselines comes later.

---

## Infrastructure

### Minimum Viable Setup (Phase 0)

- **Data generation:** one multi-core machine. Held–Karp + `multiprocessing` solves n ≤ 12 in minutes to ~an hour. **No queue, no Kubernetes, no Concorde.**
- **Model training:** single consumer GPU (RTX 3090/4090).
- **Storage:** <1 GB for all Phase 0 data.
- **Orchestration:** a Python script that manages the curriculum, triggers epochs, runs validation, logs metrics.

### Software Stack

- **Language:** Python
- **Framework:** PyTorch
- **Data generation:** NumPy for coordinates; Held–Karp solver (NumPy/Numba or a small C kernel); full-enumeration oracle for n≤8 cross-checks
- **Logging:** Weights & Biases, TensorBoard, or CSV — the key charts are simple
- **Containerization:** optional for Phase 0 (single machine); a CUDA + PyTorch + NumPy image is fine if you want reproducibility

### Repository Structure

```
lnhm/
├── data/
│   ├── generate.py          # Instance generation + Held–Karp solver
│   ├── held_karp.py         # Exact DP solver (+ enumeration oracle for n<=8)
│   └── dataset.py           # PyTorch Dataset class
├── model/
│   ├── encoder.py           # Transformer encoder
│   ├── decoder.py           # Autoregressive decoder
│   └── lnhm.py              # Full model
├── training/
│   ├── train.py             # Training loop
│   ├── curriculum.py        # Curriculum scheduler
│   └── evaluate.py          # Validation + metrics
├── analysis/
│   ├── cross_level.py       # Cross-level reinforcement experiment (A/B/C)
│   └── plot.py              # Metric visualization
├── configs/
│   └── phase0.yaml          # Hyperparameters + curriculum config
└── README.md
```

(`concorde_wrapper.py` and any distributed-generation code are deferred to Phase 1+.)

---

## Phase 0 Config

```yaml
experiment: phase0_cross_level_reinforcement

data:
  levels: [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  warmup_levels: [3, 4]            # degenerate / near-degenerate; seed only, not measured
  train_instances_per_level: 10000 # bump to 50k for n >= 8; 2k for warm-up levels
  val_instances_per_level: 2000    # 500 for warm-up levels
  coordinate_range: [0.0, 1.0]
  distribution: uniform
  solver: held_karp                # exact for all of Phase 0; enumeration only as n<=8 cross-check

model:
  type: transformer
  d_model: 128
  n_encoder_layers: 3
  n_heads: 8
  ff_dim: 512
  dropout: 0.1

training:
  optimizer: adam
  lr: 0.0001
  scheduler: cosine_annealing_warm_restarts
  batch_size: 256
  epochs_per_level: 150
  graduation_threshold: 0.50       # 50% within 1% of optimum, applied at n >= 5
  accuracy_tolerance: 0.01         # 1% optimality gap = "correct"
  target_canonicalization: true    # start at city 0, fix direction (smaller 2nd index)

curriculum:
  strategy: additive
  frontier_weight: 0.4
  prior_levels_weight: equal_share # remaining 0.6 split equally
  measurement_starts_at: 5         # warm-up levels carry no signal

evaluation:
  run_every_n_epochs: 1
  metrics: [accuracy, mean_gap, worst_gap]
  cross_level_seeds: 5
  cross_level_anchors: [5, 6, 7]   # non-degenerate anchor levels
  cross_level_ranges_N: [7, 10, 12] # Model A trains on [3, N], N > anchor
  controls:
    model_b: anchor_only_fixed_data     # naive control (same compute, fixed dataset)
    model_c: anchor_only_fresh_data      # regularization control (unlimited unique data)
  success_metric: A_minus_C_pp          # require A - C >= 5pp, consistent across seeds
  localization_probe: true              # frozen-encoder test: is transfer encoder-borne?

infrastructure:
  gpu: single
  data_gen_parallelism: max_cores
  data_gen_distributed: false      # Phase 0 is single-machine; no queue/cluster/Concorde
  storage_format: jsonl_gzip
```

---

## Decision Log

Decisions to make during or after Phase 0, not before:

| Decision | Trigger | Options |
|----------|---------|---------|
| Switch to GNN | Transformer shows no learning signal after 500 epochs on n=5 | GatedGCN, GraphSAGE |
| Switch to REINFORCE | Phase 0 succeeds, moving to Phase 1 beyond brute-force frontier | Policy gradient with greedy rollout baseline |
| Increase model size | Cross-level reinforcement exists but accuracy plateaus below useful levels | Double d_model and layers |
| Change coordinate distribution | Phase 1 model fails on non-uniform instances | Add clustered, grid-perturbed to training mix |
| Add beam search at inference | Model produces reasonable but not tight solutions | Beam width 5–10 at decode time |
| Kill the project | Phase 0 shows no cross-level reinforcement (A not > C by ≥5pp) across 5 seeds | — |

---

## Success Criteria Summary

**Phase 0 passes if:**
1. The model learns *something* at the anchor level — n=5 accuracy meaningfully exceeds the random-tour baseline (≈ 1 / ((n−1)!/2), i.e. ~8% at n=5).
2. Training on higher levels measurably improves anchor-level accuracy — **Model A beats Model C by ≥ 5pp**, consistent across ≥5 seeds (A > B alone is insufficient; C controls for data-diversity regularization).
3. The effect persists across at least 3 anchor levels / range extensions (e.g. visible for anchors 5, 6, 7), not just one transition.

**Phase 0 fails if:**
1. No learning signal at n=5 after reasonable training.
2. Cross-level reinforcement is absent or inconsistent once the regularization control (C) is accounted for.
3. Lower-level accuracy degrades as higher levels are added despite mixed training.

If Phase 0 passes, build Phase 1. If it fails, stop, and record how it failed; the failure taxonomy across classes is itself a deliverable of the factory program.

---

## Spec Revision Notes (corrections from the original draft)

1. **Anchor moved off the degenerate floor.** n=3 has exactly one tour (accuracy ≡ 100%), so the original headline test and "33% random baseline" success criterion were unmeasurable there. n=3/4 are now warm-up only; all backward-transfer measurement anchors at n≥5.
2. **Held–Karp from the start.** The original's enumerate-to-n=12 plan implied ~3,300 CPU-hours for n=12 alone, making data generation, not GPU training, the project bottleneck. Held–Karp (O(n²·2ⁿ)) generates the entire Phase 0 dataset in minutes-to-an-hour on one machine. Distributed/Concorde infrastructure is deferred to Phase 1+.
3. **A/B confound closed with a third model.** The compute-matched single-level control (Model B) overfits its fixed dataset, so A>B could be mere regularization. Added **Model C** (single-level, unlimited fresh data); the success metric is now **A − C ≥ 5pp**, which isolates genuine size-transfer from data-diversity effects.
4. **Target canonicalization.** Teacher-forcing loss is computed against canonicalized tours (fixed start city 0, fixed direction) so the supervised signal doesn't penalize valid rotations/reflections of the optimal cycle.
5. **Encoder-localization probe added.** Because only encoder-borne structure can generalize toward the wider LNHM thesis, an optional frozen-encoder probe confirms the transfer lives in the representation, not the routing-specific decoder head.
