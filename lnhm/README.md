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
| Dataset readers (`data/dataset.py`) | **implemented** (torch wrapper lazy) |
| Model (encoder / decoder) | stub |
| Training loop, curriculum, evaluation | stub |
| Cross-level analysis, plotting | stub |

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

## Layout

```
lnhm/
├── data/        # generation + solvers + dataset (implemented)
├── model/       # encoder, decoder, full model (stub)
├── training/    # train loop, curriculum, evaluate (stub)
├── analysis/    # cross-level experiment, plotting (stub)
└── configs/     # phase0.yaml
```
