# Phase 3 harness notes — read this first (anti-cold-start)

Discovered facts about the LNHM harness needed to run local experiments, so we
don't re-derive them every session. Paths are relative to `lnhm/` unless noted.
All commands below assume `cwd = lnhm/` and the venv at `lnhm/.venv`.

## The shallow model (local CPU probe)

- **`runs/full/model_final.pt`** is the 710K-param shallow Phase-0 model
  (trained on levels 3–12). Arch: `d_model=128, n_encoder_layers=3, n_heads=8,
  ff_dim=512, input_dim=2`. This is the "C00 control-equivalent" and the model we
  use for cheap local CPU tests. `runs/first` and `runs/second` are the same arch.
- **CAVEAT — these are RAW `state_dict`s**, not the self-describing format
  (`{state_dict, model_config}`) that `analysis/eval_heldout.py` requires. To use
  them with the eval harness, wrap them first — see `wrap_checkpoint.py`, which
  writes `runs/full/model_selfdesc.pt`.
- Load directly (bypassing the wrapper):
  ```python
  from model.lnhm import LnhmModel
  m = LnhmModel(d_model=128, n_encoder_layers=3, n_heads=8, ff_dim=512)
  m.load_state_dict(torch.load('runs/full/model_final.pt', map_location='cpu'))
  m.eval()
  tours, log_probs = m.solve(coords, mode='greedy')  # coords: (B, n, 2) float32
  ```
  `mode='sample'` for stochastic rollout (best-of-K). Returns `(tours, log_probs)`.
- The **depth-20 model** (the one that extrapolates well, Phase 2 C10) lives on the
  GPU/Windows box, not here. Local tests use the shallow model; anything needing the
  depth model or larger n runs there.

## Data format (identical for uniform and clustered)

- Gzipped JSONL, one instance/line: `{id, n, coords:[[x,y]...], optimal_tour:[...],
  optimal_distance: float}`. Coords rounded to 6 decimals; the reference solver runs
  on the *rounded* coords so `optimal_distance` is self-consistent.
- File naming: `level_{n:02d}_{split}.jsonl.gz`. The **test sets use split=`train`**
  in the filename (historical); `eval_heldout.py` defaults `--split train`, so keep
  clustered files named `level_XX_train.jsonl.gz` too.
- Load into packed numpy arrays: `data.dataset.load_level_arrays(path, limit)` ->
  `(coords (N,n,2) f32, tours (N,n) i64, distances (N,) f32)`.

## Reference solvers (for the gap denominator)

- **Held-Karp** (`data/held_karp.py:held_karp`) is EXACT but memory-bound above
  n≈13–16. Use for n≤12.
- **LKH-3** (`analysis/baselines.py:lkh_tour`) is near-optimal, needed for n>12.
  Requires the LKH binary. **A built binary already exists at
  `/home/ticktockbent/tools/LKH-3.0.13/LKH`** (not on PATH — pass its path).
  `generate.py`'s `--solver auto` picks Held-Karp for n≤12, LKH for n>12.
- `phase0_test/` is the existing **uniform** held-out set (base_seed=12345, disjoint
  from training), levels {5,8,10,12,16,20,25,30}, 1000/level, references baked in.
  Reuse it directly as the uniform arm of any A/B — no need to regenerate it.

## Running a held-out eval on CPU

```bash
cd lnhm
.venv/bin/python analysis/eval_heldout.py \
  --checkpoint runs/full/model_selfdesc.pt \
  --test-dir <dir> --levels 8 10 12 16 20 25 30 \
  --samples 16 --limit 200 --out <dir>/heldout_full.json --device cpu
```
Reports per-level greedy + sampled-best-of-K gap and `p = 1/(1+mean_gap)`.

## Gotchas

- **Bash cwd persists** across tool calls in a session and an earlier `cd lnhm`
  sticks — check `pwd` if a relative path 404s.
- `.gz` files are forced `binary` in `.gitattributes` (Windows EOL corruption).
  Any new `.jsonl.gz` inherits this — good, leave it.
