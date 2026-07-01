# LNHM — the code

**Large NP-Hard Model.** A method ("solver factory") for producing a bespoke
neural solver per NP-hard problem class via curriculum bootstrapping from small,
exactly-solvable instances toward large ones. The central per-class bet is
**scale-unification**: the structural principles that produce good solutions at
small n are the same at large n, so training on harder instances retroactively
improves performance on easier ones (cross-level reinforcement).

This package began as the Phase 0 pipeline and now serves all four phases. For
the story and findings, start at the [root README](../README.md); the specs and
results live in `../phase0/` through `../phase3/`. This file covers setup and
the operational commands. The command reference for held-out evals is
[`../phase3/HARNESS.md`](../phase3/HARNESS.md).

## What's here (all implemented)

| Component | Serves |
|-----------|--------|
| Held-Karp exact solver + enumeration oracle (`data/held_karp.py`) | all phases |
| Instance generators: uniform (`data/generate.py`), clustered (`data/generate_clustered.py`), mixed-diet (`data/generate_mixed.py`) | Phases 0-3 |
| Data pool + mixed-n padded collation (`data/dataset.py`) | training |
| Model: encoder, decoder, wrapper (`model/`) | all phases |
| Curriculum, training loop, evaluation (`training/`) | Phases 0, 2, 3 |
| Cross-level A/B/C/D experiment (`analysis/cross_level.py`) | Phase 0 |
| Classical baselines + LKH wrapper (`analysis/baselines.py`) | Phases 1-3 |
| Recursive composition pipeline (`analysis/compose.py`) | Phase 1 |
| Compiled 2-opt + Or-opt cleanup (`analysis/fast_local_search.py`) | Phases 1, 3 |
| Cost/quality frontier runner (`analysis/frontier.py`) | Phase 1 |
| Held-out eval, greedy + best-of-k (`analysis/eval_heldout.py`) | Phases 2, 3 |
| Kill-suite scripts (`analysis/t1_ablation.py`, `analysis/t5_fixedk_gate.py`) | Phase 3 |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Data generation

Validate the solver against brute force first (Held-Karp must match exact
enumeration):

```bash
.venv/bin/python data/held_karp.py            # self-test, n=3..8
.venv/bin/python data/generate.py --cross-check --cross-check-size 300
```

Generate the full Phase 0 dataset (levels 3–12) into `data/phase0/`:

```bash
.venv/bin/python data/generate.py --output-dir data/phase0
```

Smoke test a single level quickly:

```bash
.venv/bin/python data/generate.py --levels 5 --train-count 100 --val-count 20
```

Output is one gzipped JSONL file per level per split
(`level_05_train.jsonl.gz`, …). Each line is one instance:

```json
{"id": "...", "n": 5, "coords": [[0.23, 0.71], ...],
 "optimal_tour": [0, 3, 1, 4, 2], "optimal_distance": 2.847}
```

Every instance is fully determined by `(base_seed, level, split, index)`, so
generation is reproducible and resumable. Tours are stored **canonicalized**
(start at city 0, direction fixed) so they can be used directly as supervised
targets.

## Training

Smoke test the full loop quickly on CPU (small subset, a few levels):

```bash
.venv/bin/python training/train.py --data-dir data/phase0 \
    --levels 3 4 5 6 --train-limit 600 --val-limit 200 \
    --steps-per-epoch 30 --max-epochs-per-level 4 --device cpu \
    --output-dir runs/smoke
```

Full Phase 0 run (uses `configs/phase0.yaml`; picks up CUDA automatically):

```bash
.venv/bin/python training/train.py --config configs/phase0.yaml --data-dir data/phase0
```

Per-epoch metrics (accuracy / mean gap / worst gap for every active level) are
written to `<output-dir>/metrics.csv`; the final model to `model_final.pt`. Plot
the key chart:

```bash
.venv/bin/python analysis/plot.py --metrics runs/phase0/metrics.csv \
    --out runs/phase0/accuracy_by_level.png
```

Notes:
- Gradient clipping is on by default (`--grad-clip 1.0`). At `lr>1e-4` without it
  the model overshoots on step 1 and collapses to the uniform-policy plateau.
- Evaluation runs per level (uniform n), so greedy decode never sees padding.
- Warm-up levels (n=3,4) graduate automatically; n=3 is always 100% (one tour).

## Layout

```
lnhm/
├── data/        # generators (uniform/clustered/mixed) + exact solvers + dataset
├── model/       # encoder, decoder, full model
├── training/    # train loop, curriculum, evaluate
├── analysis/    # baselines, composition, fast_local, frontier, evals, kill-suite
└── configs/     # phase0.yaml
```
