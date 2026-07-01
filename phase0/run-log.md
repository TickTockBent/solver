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

## Cross-level spot-check — anchor n=9, N=12, 2 seeds (design validation)

- **Date:** 2026-06-30
- **Goal:** validate the A/B/C/D harness at a HARD anchor with adequate budget,
  after the first attempt (anchor n=5, 600 steps) failed as a budget/saturation
  artifact. Models: A=curriculum[3,12], B=anchor-only small pool (2k),
  C=anchor-only large fresh pool (40k), D=curriculum[3,9]. 2500 steps each,
  compute-matched, same init per seed. CPU.
- **Note:** the run completed cleanly but the harness reported exit code -1 — an
  artifact of a session/container restart mid-run (job dir changed); full results
  were written.

### Results

|        | acc_A | acc_B | acc_C | acc_D | A−C | A−D |
|--------|-------|-------|-------|-------|-----|-----|
| seed 0 | 0.718 | 0.747 | 0.805 | 0.636 | −0.086 | +0.082 |
| seed 1 | 0.704 | 0.760 | 0.796 | 0.625 | −0.092 | +0.079 |
| mean   | 0.711 | 0.753 | 0.800 | 0.631 | **−0.089** | **+0.081** |

### Reading

1. **Harness validated.** A escaped the starvation floor (0.71, healthy), and
   **B < C** (~5pp) — the overfitting control now bites at a hard anchor.
2. **A < C (−0.089):** at equal compute, single-level *focus* beats the diluted
   full curriculum at the anchor. The spec's naive "curriculum > single-level"
   metric **fails** under compute-matching.
3. **A > D (+0.081, consistent both seeds):** A and D are both curricula; the only
   difference is A also trains on levels *larger* than the anchor (10,11,12).
   A wins by ~8pp → **training on larger instances improved the smaller anchor** —
   the backward-transfer / scale-unification signal, isolated. **Adding Model D
   flipped a false negative (A<C) into a real positive** — without it we'd have
   concluded "no effect."

### Caveat (confound surfaced) + fix

The +8pp is confounded by the **LR schedule**: both models used a single cosine
decay over 2500 steps, so D trained its anchor at the low-LR tail while A hit it
mid-schedule. Part of A>D could be LR position, not transfer. **Fix applied:**
`cross_level.py` now uses **constant LR** (removes schedule position as a
variable) and writes the results CSV incrementally with flush (so a killed run
keeps completed rows — directly addressing the detached-death above).

**Verdict:** promising hint, not a result — 2 seeds, one anchor, now de-confounded.
Ready for the full matrix (anchors {5 neg-control, 9, 11}, 5 seeds) on a GPU
(~20 min there vs ~15 h CPU).

---

## Cross-level FULL MATRIX — the Phase 0 cross-level result

- **Date:** 2026-07-01
- **Setup:** anchors {5 (neg control), 9, 11} x ranges {7,10,12} (N>anchor) x 5 seeds,
  2500 steps, **constant LR**, compute-matched, same init per seed. Models A
  (curriculum [3,N]), B (anchor-only small pool), C (anchor-only large fresh pool),
  D (curriculum [3,anchor]). RTX 4090. Raw data: `phase0/cross_level_results.csv`.
- **Key comparison:** A vs D = "do levels LARGER than the anchor help it?" (the
  backward-transfer / scale-unification claim). A vs C = spec's naive metric.

### Results (means over 5 seeds)

| anchor | N | A−D | A−C | abs A / C / D |
|--------|---|-----|-----|---------------|
| 5 | 7  | +0.031 | −0.003 | 0.98 / 0.98 / 0.95 |
| 5 | 10 | +0.025 | −0.008 | 0.97 / 0.98 / 0.95 |
| 5 | 12 | +0.021 | −0.013 | 0.97 / 0.98 / 0.95 |
| 9 | 10 | +0.028 | −0.050 | 0.79 / 0.84 / 0.75 |
| 9 | 12 | +0.049 | −0.029 | 0.81 / 0.84 / 0.75 |
| 11 | 12 | +0.032 | −0.094 | 0.66 / 0.75 / 0.63 |

### Findings

1. **A > D in ALL 30 runs.** Every seed/config positive (range +0.013 to +0.057);
   none negative. Sign-consistency at this level means the effect — training on
   larger instances measurably improves the smaller anchor — is **real and robust**.
2. **The N-slope flips with anchor difficulty (mechanistic result).**
   - n=5 (saturated, C≈0.98): A−D *shrinks* with N (0.031 → 0.025 → 0.021) —
     dilution wins where there's no headroom.
   - n=9 (headroom, C≈0.84): A−D *grows* with N (0.028 → 0.049) — adding levels
     11,12 above the anchor helps *more*, despite extra dilution — transfer wins.
   This sign flip is strong evidence for genuine larger-instance transfer, not just
   "curriculum helps." (By levels-above-anchor: 1 level ≈ +0.03, 3 levels ≈ +0.05,
   but only where the anchor has room to use them.)
3. **Magnitude is modest — below the ≥5pp bar.** Best config (n=9, N=12) means
   **+0.049**, right at the line; 3/5 seeds clear +0.05, one outlier at +0.033. No
   config cleanly PASSES "≥5pp across all seeds." The effect is real but small
   (~2–5pp) at this model scale.
4. **A−C < 0 everywhere** — single-level focus beats the diluted curriculum at
   equal compute. Expected; confirms D (not C) is the right thesis control.

### Verdict

**Cross-level reinforcement is CONFIRMED — real, robustly-signed, mechanistically
coherent — but MODEST (~3–5pp), not the ≥5pp "strong effect" we pre-registered.**

Reconciliation: Run 3's confounded backward-transfer hint was +8 to +13pp; the
clean A/D controls shrink it to ~3–5pp. That gap is exactly why we ran the
controls — the confounded number was inflated by anchor levels remaining in the
batch mix. This is the de-confounded truth.

### Implications

- The program is alive; the open question is now the effect's *size*, which is
  what **Phase 2's capacity sweep** must answer: **does A−D grow with model size?**
  (+3pp → +10pp with a bigger model would make the thesis strong; staying ~3pp
  means real-but-small, and the composition path carries the weight.)
- **Phase 1 (composition) does not depend on this being large** — the model is a
  local solver regardless — so it proceeds unblocked.
- n=11's slope is untestable in Phase 0 (no levels above 12); confirming the flip
  at higher anchors needs a curriculum extended past n=12 (Phase 2).

---
