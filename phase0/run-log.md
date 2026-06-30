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
