# LNHM — Phase 0

**Large NP-Hard Model.** A method ("solver factory") for producing a bespoke
neural solver per NP-hard problem class via curriculum bootstrapping from small,
exactly-solvable instances toward large ones. The central per-class bet is
**scale-unification**: the structural principles that produce good solutions at
small n are the same at large n, so training on harder instances retroactively
improves performance on easier ones (cross-level reinforcement).

Phase 0 tests the smallest version of that bet on the friendliest class —
**Euclidean TSP**. The full specification is in
[`../phase0/phase0-spec.md`](../phase0/phase0-spec.md).

## Status

| Component | State |
|-----------|-------|
| Held-Karp exact solver + brute-force oracle (`data/held_karp.py`) | **implemented** |
| Instance generation, parallel + reproducible (`data/generate.py`) | **implemented** |
| Data pool + mixed-n padded collation (`data/dataset.py`) | **implemented** |
| Model — encoder, decoder, wrapper (`model/`) | **implemented** (overfits real tours to 100%) |
| Curriculum, training loop, evaluation (`training/`) | **implemented** |
| Accuracy-by-level plot (`analysis/plot.py`) | **implemented** |
| Cross-level reinforcement experiment (`analysis/cross_level.py`) | stub (next) |

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
├── data/        # generation + solvers + dataset (implemented)
├── model/       # encoder, decoder, full model (stub)
├── training/    # train loop, curriculum, evaluate (stub)
├── analysis/    # cross-level experiment, plotting (stub)
└── configs/     # phase0.yaml
```
