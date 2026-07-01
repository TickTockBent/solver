#!/usr/bin/env bash
# Phase 2: depth x data 2x2 overtraining experiment.
# Trains the 4 cells x N seeds to convergence, then runs the held-out eval on each.
# Run this on the GPU box after the data pools are present (data/phase0 + data/phase0_test).
#
#   PYTHON=./.venv/bin/python  bash phase2/run_phase2.sh          # 2 seeds (default)
#   SEEDS="0 1 2"  bash phase2/run_phase2.sh                      # add a 3rd seed
#
# Order: full 2x2 at seed 0 first, so a preliminary interaction is available early.
set -euo pipefail
cd "$(dirname "$0")/../lnhm"

PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-data/phase0}"
TEST_DIR="${TEST_DIR:-data/phase0_test}"
OUT="${OUT:-runs/phase2}"
SEEDS="${SEEDS:-0 1}"
MAX_EPOCHS="${MAX_EPOCHS:-150}"      # per-level cap (convergence via graduation @ 50%)
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EVAL_LEVELS="${EVAL_LEVELS:-5 8 10 12 16 20 25 30}"
SAMPLES="${SAMPLES:-16}"
EVAL_LIMIT="${EVAL_LIMIT:-1000}"

LEVELS_LOW="3 4 5 6 7 8 9 10 11 12"
LEVELS_HIGH="3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

run_cell () {
  local name="$1" levels="$2" limit="$3" seed="$4"
  local run="${name}_s${seed}"
  echo ""; echo "############ TRAIN ${run}  (levels: ${levels%% *}..${levels##* }, limit ${limit}, seed ${seed}) ############"
  $PYTHON training/train.py --data-dir "$DATA_DIR" --output-dir "$OUT" \
    --levels $levels --train-limit "$limit" \
    --max-epochs-per-level "$MAX_EPOCHS" --steps-per-epoch "$STEPS_PER_EPOCH" \
    --run-name "$run" --seed "$seed"
  echo "############ EVAL ${run} ############"
  $PYTHON analysis/eval_heldout.py --checkpoint "$OUT/$run/model_final.pt" \
    --test-dir "$TEST_DIR" --levels $EVAL_LEVELS --samples "$SAMPLES" \
    --limit "$EVAL_LIMIT" --out "$OUT/$run/heldout.json"
}

for seed in $SEEDS; do
  run_cell C00_control "$LEVELS_LOW"  4000  "$seed"
  run_cell C10_depth   "$LEVELS_HIGH" 4000  "$seed"
  run_cell C01_data    "$LEVELS_LOW"  20000 "$seed"
  run_cell C11_both    "$LEVELS_HIGH" 20000 "$seed"
done

echo ""; echo "ALL RUNS DONE -> $OUT"
echo "Assemble tables: $PYTHON ../phase2/assemble_results.py --runs $OUT"
