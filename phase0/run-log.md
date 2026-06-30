# LNHM Phase 0 — Run Log

A chronological log of training runs and what they showed. Each entry records the
exact command, environment, results, and how much we trust each finding. Run
outputs themselves (`runs/…`) are gitignored and regenerable; this log is the
durable record.

---

## Run 1 — first local validation (single seed, levels 3–8)

- **Date:** 2026-06-30
- **Goal:** sanity-check that a single model genuinely climbs across levels and
  that the curriculum/eval machinery behaves, before building the A/B/C controls.
- **Command:**
  ```bash
  python training/train.py --data-dir data/phase0 \
      --levels 3 4 5 6 7 8 --steps-per-epoch 100 --max-epochs-per-level 60 \
      --device cpu --seed 0 --output-dir runs/first
  ```
- **Config:** d_model=128, 3 enc layers, 8 heads, ff=512, dropout=0.1; Adam lr=1e-4;
  cosine warm restarts; batch=256; grad-clip=1.0; graduation threshold 0.50 (within
  1% of optimum); warm-up levels {3,4} auto-graduate.
- **Environment:** motherbrain, 4 CPU threads. **Wall time: 113 s, 11 epochs total.**

### Per-epoch accuracy trajectory

```
epoch  frontier  n3    n4    n5    n6    n7    n8
   1   n=3     1.00*
   2   n=4     1.00  0.49*
   3   n=5     1.00  0.54  0.24*
   4   n=5     1.00  0.54  0.28*          ← plateau
   5   n=5     1.00  0.53  0.23*
   6   n=5     1.00  0.53  0.29*
   7   n=5     1.00  0.70  0.45*          ← breakthrough begins
   8   n=5     1.00  0.93  0.84*          ← n=5 clicks, graduates
   9   n=6     1.00  0.97  0.91  0.84*    ← n=6 graduates in ONE epoch
  10   n=7     1.00  0.98  0.93  0.87  0.77*
  11   n=8     1.00  0.99  0.94  0.89  0.79  0.69*
(* = frontier / newest level that epoch)
```

### Final metrics (epoch 11)

| n | accuracy | mean gap | worst gap |
|---|----------|----------|-----------|
| 3 | 1.000 | 0.0000 | 0.000 |
| 4 | 0.986 | 0.0005 | 0.066 |
| 5 | 0.945 | 0.0017 | 0.087 |
| 6 | 0.885 | 0.0043 | 0.208 |
| 7 | 0.789 | 0.0087 | 0.198 |
| 8 | 0.689 | 0.0144 | 0.254 |

### Findings (with confidence)

1. **The model learns cleanly. [HIGH]** Accuracy decays smoothly with n
   (1.00→0.69); mean gaps are tiny (n=8 averages 1.4% off even when not within the
   1% bar). No collapse, NaN, or plateau pathology. Architecture + pipeline sound.

2. **A breakthrough at n=5. [HIGH that it happened; mechanism open]** Accuracy sat
   at ~0.25 for four epochs, then jumped to 0.84. Once n=5 was learned, **every
   higher level graduated in a single epoch.** Interpretation: the model learned
   the structural *principle* once, then reused it — consistent with
   "internalize the principle, not the instance," and a small echo of the
   grokking/phase-transition theme. This is also why the run took 2 min, not 50:
   graduation is fast once the principle lands.

3. **Backward-transfer hint. [LOW — deliberately]** Lower levels kept climbing
   after the frontier moved past them (n=5: 0.84→0.945 while training on n=6–8;
   n=4: 0.49→0.99). This is the *shape* the thesis predicts, **but it is not
   evidence.** n=5 remains 12–20% of every batch after graduating, so the gain
   could be plain extra training rather than transfer *from larger instances*.
   Disentangling those is exactly the job of the Model A/B/C controls.

### Caveats / what this run does NOT show

- Single seed, no control group → finding #3 is confounded by design.
- The 0.50 graduation threshold is too low to study anything past n=5: levels
  graduate the first epoch they appear, so n=6–8 numbers reflect *transfer + one
  epoch of training*, not each level's trained ceiling. For the real experiment,
  raise the threshold or fix epochs-per-level so each level actually trains.
- Only reaches n=8; says nothing yet about n=9–12 (slower steps, and the
  breakthrough may or may not extend that far).

### Calibration for later runs

- **Time is a non-issue locally.** Full 3→12 likely < 10 min on this CPU *if*
  n=9–12 graduate as fast as n=6–8 did.
- Next: reproduce with a different seed (Run 2) before changing anything.

---

## Run 2 — reproducibility check (seed 1, identical config)

- **Date:** 2026-06-30
- **Goal:** confirm Run 1's trends are not a single-seed artifact. Same command,
  `--seed 1`, `--output-dir runs/second`. (A same-seed re-run would be a
  deterministic replay, so seed was changed.)
- **Environment:** motherbrain, 4 CPU threads. **Wall time: 84 s, 8 epochs total.**

### Trajectory

```
epoch  frontier  n3    n4    n5    n6    n7    n8
   3   n=5     1.00  0.51  0.28*
   4   n=5     1.00  0.57  0.30*          ← shorter plateau
   5   n=5     1.00  0.78  0.63*          ← breakthrough (epoch 5, vs epoch 8 in Run 1)
   6   n=6     1.00  0.96  0.86  0.79*    ← one-epoch cascade resumes
   7   n=7     1.00  0.98  0.90  0.83  0.74*
   8   n=8     1.00  0.98  0.93  0.86  0.77  0.66*
```

### Final accuracy: Run 1 vs Run 2

| n | Run 1 (seed 0) | Run 2 (seed 1) | Δ |
|---|----------------|----------------|-----|
| 3 | 1.000 | 1.000 | +0.000 |
| 4 | 0.986 | 0.984 | −0.002 |
| 5 | 0.945 | 0.928 | −0.016 |
| 6 | 0.885 | 0.857 | −0.028 |
| 7 | 0.789 | 0.767 | −0.022 |
| 8 | 0.689 | 0.661 | −0.029 |

### Verdict — all four Run 1 findings reproduce

1. **Clean learning:** reproduced; identical monotonic ladder shape.
2. **n=5 breakthrough:** reproduced — plateau-then-jump is real in both runs. The
   **timing is the seed-variable part** (jump at epoch 5 here vs epoch 8 in Run 1);
   *whether* it happens and the post-jump outcome are stable.
3. **One-epoch graduation cascade above n=5:** reproduced.
4. **Backward-transfer hint:** reproduced (n=5: 0.63→0.93 after frontier moved on;
   n=4: 0.48→0.98). Still confounded, still only a hint.

**Caveat on the comparison:** the final-accuracy deltas are mildly confounded by
total epoch count — Run 1 ran 11 epochs vs Run 2's 8 (its later breakthrough
pushed the whole schedule out), so its lower levels got more cumulative training.
The consistent small negative Δ is mostly that, not seed quality. Fixing
epochs-per-level (rather than graduate-and-advance) will remove this for the real
experiment.

**Conclusion:** trends are seed-robust. Cleared to (a) run the full 3→12
curriculum and (b) build the A/B/C controls. Open knob before the real
experiment: raise the graduation threshold or fix epochs-per-level so each level
actually trains and we can measure its ceiling rather than transfer-plus-one-epoch.

---

## Run 3 — full curriculum 3→12 (single seed, scale validation)

- **Date:** 2026-06-30
- **Goal:** confirm the breakthrough/cascade extends past n=8 to the top of the
  brute-force range, and the model doesn't fall apart at n=9–12.
- **Command:** as Run 1 but `--levels 3..12`, `--seed 0`, `--output-dir runs/full`.
- **Environment:** motherbrain, 4 CPU threads. **Wall time: 461 s (~7.7 min), 22 epochs.**

### Outcome: every level graduated. Cascade holds through n=11; n=12 is the wall.

Graduation epochs: n=3@1, 4@2, 5@8, 6@9, 7@10, 8@11, 9@12, 10@13, 11@14, **12@22**.
So n=6–11 each graduated in ~1 epoch as the frontier (the cascade), but **n=12
needed ~8 epochs to crawl to exactly 0.500** — the 710K-param model's capacity
ceiling becoming visible right at the top of the exact-solvable range.

### Final ladder (epoch 22)

| n | accuracy | mean gap | worst gap |
|---|----------|----------|-----------|
| 3 | 1.000 | 0.0000 | 0.000 |
| 4 | 0.960 | 0.0011 | 0.066 |
| 5 | 0.955 | 0.0014 | 0.061 |
| 6 | 0.933 | 0.0021 | 0.101 |
| 7 | 0.888 | 0.0035 | 0.155 |
| 8 | 0.822 | 0.0065 | 0.268 |
| 9 | 0.756 | 0.0091 | 0.259 |
| 10 | 0.666 | 0.0133 | 0.205 |
| 11 | 0.596 | 0.0173 | 0.272 |
| 12 | 0.500 | 0.0231 | 0.377 |

Note: mean gaps stay tiny throughout — even n=12 averages 2.3% off optimal. The
model "mostly gets it" everywhere; the strict 1% bar is what only 50% clear at n=12.

### Backward-transfer view: accuracy at graduation vs. final epoch

| n | @graduation | @final | gain |
|---|-------------|--------|------|
| 4 | 0.486 | 0.960 | +0.474 |
| 5 | 0.836 | 0.955 | +0.119 |
| 6 | 0.843 | 0.933 | +0.091 |
| 7 | 0.767 | 0.888 | +0.121 |
| 8 | 0.689 | 0.822 | +0.132 |
| 9 | 0.631 | 0.756 | +0.125 |
| 10 | 0.562 | 0.666 | +0.104 |
| 11 | 0.519 | 0.596 | +0.078 |

**Every level kept improving after the frontier moved past it** (+0.08 to +0.13
for mid levels), consistently and monotonically — the shape the scale-unification
thesis predicts. Cross-check vs Run 1 (levels 3–8 only): the full run's lower
levels all end *higher* (n=8: 0.689 → 0.822) because they trained longer while the
frontier climbed.

**Still confounded — still not proof.** Those levels remained 7–12% of every batch
after graduating, so the gain mixes "more training on own data" with "transfer
from larger n." Disentangling these is exactly the Model A/B/C job.

### Findings

1. **Scale-unification holds across the full brute-force range. [MEDIUM-HIGH]** No
   collapse at n=9–12; clean monotonic ladder; the learned structure transfers
   upward (one-epoch graduation through n=11).
2. **Capacity wall at n=12 for the 710K model. [HIGH]** 8 epochs to barely clear
   50% vs ~1 epoch elsewhere. Directly feeds the spec's "increase model size"
   decision lever — n≥12 is where this size runs out.
3. **Backward-transfer pattern strengthened but still confounded. [MEDIUM]** See above.

### Status

Local validation **complete and successful**. The approach works end-to-end and
is seed-robust. Next real milestone: `analysis/cross_level.py` (A/B/C controls) to
convert the backward-transfer pattern from a consistent hint into a measured
effect — the compute-heavy part that wants the GPU/offload path.

---
