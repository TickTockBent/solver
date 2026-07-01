#!/usr/bin/env bash
# LNHM container entrypoint.
#
# Works in both deployment paths:
#   - Docker image (this repo's Dockerfile): WORKDIR /workspace/lnhm, run as ENTRYPOINT.
#   - FlightDeck bundle: code extracted to /workspace, invoked as
#     `bash docker/entrypoint.sh`. Repo root is derived from this script's location.
#
# Env-driven so a job platform can parametrize without rebuilding. Regenerates the
# (deterministic) dataset if absent, checks the GPU, then runs the selected task,
# writing artifacts to LNHM_OUTPUT_DIR and optionally self-uploading them.
#
# LNHM_TASK selects what to run:
#   train      (default) -- curriculum training (training/train.py) + accuracy plot
#   crosslevel           -- A/B/C/D cross-level matrix (analysis/cross_level.py)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

LNHM_TASK="${LNHM_TASK:-train}"
LNHM_LEVELS="${LNHM_LEVELS:-3 4 5 6 7 8 9 10 11 12}"
LNHM_STEPS_PER_EPOCH="${LNHM_STEPS_PER_EPOCH:-100}"
LNHM_MAX_EPOCHS="${LNHM_MAX_EPOCHS:-150}"
LNHM_SEED="${LNHM_SEED:-0}"
LNHM_DEVICE="${LNHM_DEVICE:-cuda}"
LNHM_DATA_DIR="${LNHM_DATA_DIR:-$REPO_ROOT/data/phase0}"
LNHM_OUTPUT_DIR="${LNHM_OUTPUT_DIR:-$REPO_ROOT/outputs}"
LNHM_GENERATE_DATA="${LNHM_GENERATE_DATA:-auto}"        # auto | always | never
LNHM_LKH_BINARY="${LNHM_LKH_BINARY:-LKH}"              # LKH-3 binary (labels n>12)
LNHM_XLEVEL_ARGS="${LNHM_XLEVEL_ARGS:-}"                # extra args for cross_level.py
LNHM_RESULT_UPLOAD_URL="${LNHM_RESULT_UPLOAD_URL:-}"    # optional: PUT target for results
# Training run identity + model-size overrides (for level/capacity sweeps):
LNHM_RUN_NAME="${LNHM_RUN_NAME:-}"
LNHM_D_MODEL="${LNHM_D_MODEL:-}"
LNHM_N_LAYERS="${LNHM_N_LAYERS:-}"
LNHM_N_HEADS="${LNHM_N_HEADS:-}"
LNHM_FF_DIM="${LNHM_FF_DIM:-}"

mkdir -p "$LNHM_DATA_DIR" "$LNHM_OUTPUT_DIR"

echo "=== LNHM entrypoint (task=$LNHM_TASK, repo=$REPO_ROOT) ==="
echo "  device=$LNHM_DEVICE data=$LNHM_DATA_DIR output=$LNHM_OUTPUT_DIR generate=$LNHM_GENERATE_DATA"

# --- Dataset: regenerate deterministically (seed-fixed) when needed (both tasks) ---
needs_generation=false
case "$LNHM_GENERATE_DATA" in
  always) needs_generation=true ;;
  never)  needs_generation=false ;;
  *)      [ -z "$(ls -A "$LNHM_DATA_DIR" 2>/dev/null || true)" ] && needs_generation=true ;;
esac
if [ "$needs_generation" = true ]; then
  echo "--- generating dataset (levels: $LNHM_LEVELS; LKH for n>12) ---"
  # shellcheck disable=SC2086
  python data/generate.py --output-dir "$LNHM_DATA_DIR" --levels $LNHM_LEVELS --lkh-binary "$LNHM_LKH_BINARY"
else
  echo "--- using existing dataset ---"
fi

# --- GPU sanity check (non-fatal) ---
python - <<'PY'
import torch
ok = torch.cuda.is_available()
print(f"--- torch {torch.__version__} | cuda: {ok} | "
      f"{torch.cuda.get_device_name(0) if ok else 'CPU only'} ---")
PY

# --- Run the selected task ---
case "$LNHM_TASK" in
  train)
    echo "--- training (curriculum) ---"
    train_extra=()
    [ -n "$LNHM_RUN_NAME" ] && train_extra+=(--run-name "$LNHM_RUN_NAME")
    [ -n "$LNHM_D_MODEL" ] && train_extra+=(--d-model "$LNHM_D_MODEL")
    [ -n "$LNHM_N_LAYERS" ] && train_extra+=(--n-encoder-layers "$LNHM_N_LAYERS")
    [ -n "$LNHM_N_HEADS" ] && train_extra+=(--n-heads "$LNHM_N_HEADS")
    [ -n "$LNHM_FF_DIM" ] && train_extra+=(--ff-dim "$LNHM_FF_DIM")
    # shellcheck disable=SC2086
    python training/train.py \
      --data-dir "$LNHM_DATA_DIR" \
      --levels $LNHM_LEVELS \
      --steps-per-epoch "$LNHM_STEPS_PER_EPOCH" \
      --max-epochs-per-level "$LNHM_MAX_EPOCHS" \
      --seed "$LNHM_SEED" \
      --device "$LNHM_DEVICE" \
      --output-dir "$LNHM_OUTPUT_DIR" \
      ${train_extra[@]+"${train_extra[@]}"}
    # train.py writes to a per-run subdir; plot from the newest one.
    latest_run="$(ls -td "$LNHM_OUTPUT_DIR"/*/ 2>/dev/null | head -1)"
    echo "--- plotting ($latest_run) ---"
    python analysis/plot.py \
      --metrics "${latest_run}metrics.csv" \
      --out "${latest_run}accuracy_by_level.png" || echo "  (plot failed, continuing)"
    ;;
  crosslevel)
    echo "--- cross-level A/B/C/D matrix ---"
    # shellcheck disable=SC2086
    python analysis/cross_level.py \
      --data-dir "$LNHM_DATA_DIR" \
      --device "$LNHM_DEVICE" \
      --output-dir "$LNHM_OUTPUT_DIR" \
      $LNHM_XLEVEL_ARGS
    ;;
  *)
    echo "ERROR: unknown LNHM_TASK='$LNHM_TASK' (expected: train | crosslevel)" >&2
    exit 2
    ;;
esac

# --- Self-exfiltrate results (FlightDeck has no artifact retrieval) ---
if [ -n "$LNHM_RESULT_UPLOAD_URL" ]; then
  echo "--- uploading results ---"
  results_tarball="/tmp/lnhm-results-${LNHM_TASK}-seed${LNHM_SEED}.tar.gz"
  tar -czf "$results_tarball" -C "$LNHM_OUTPUT_DIR" .
  if curl -fsS -T "$results_tarball" "$LNHM_RESULT_UPLOAD_URL"; then
    echo "  uploaded -> $LNHM_RESULT_UPLOAD_URL"
  else
    echo "  UPLOAD FAILED (exit $?)"
  fi
fi

echo "=== done; artifacts in $LNHM_OUTPUT_DIR ==="
ls -la "$LNHM_OUTPUT_DIR"
