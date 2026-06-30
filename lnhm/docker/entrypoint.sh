#!/usr/bin/env bash
# LNHM Phase 0 run script.
#
# Works in both deployment paths:
#   - Docker image (this repo's Dockerfile): WORKDIR /workspace/lnhm, run as ENTRYPOINT.
#   - FlightDeck bundle: code extracted to /workspace, invoked as
#     `bash docker/entrypoint.sh`. Repo root is derived from this script's location,
#     so it does not matter which.
#
# Driven entirely by environment variables so a job platform can parametrize the
# run without rebuilding. Regenerates the (deterministic) dataset if absent,
# checks the GPU, trains, plots, and — because FlightDeck does NOT retrieve
# artifacts itself — optionally self-uploads the results tarball.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

LNHM_LEVELS="${LNHM_LEVELS:-3 4 5 6 7 8 9 10 11 12}"
LNHM_STEPS_PER_EPOCH="${LNHM_STEPS_PER_EPOCH:-100}"
LNHM_MAX_EPOCHS="${LNHM_MAX_EPOCHS:-150}"
LNHM_SEED="${LNHM_SEED:-0}"
LNHM_DEVICE="${LNHM_DEVICE:-cuda}"
LNHM_DATA_DIR="${LNHM_DATA_DIR:-$REPO_ROOT/data/phase0}"
LNHM_OUTPUT_DIR="${LNHM_OUTPUT_DIR:-$REPO_ROOT/outputs}"
LNHM_GENERATE_DATA="${LNHM_GENERATE_DATA:-auto}"        # auto | always | never
LNHM_RESULT_UPLOAD_URL="${LNHM_RESULT_UPLOAD_URL:-}"     # optional: PUT target for results

mkdir -p "$LNHM_DATA_DIR" "$LNHM_OUTPUT_DIR"

echo "=== LNHM entrypoint (repo: $REPO_ROOT) ==="
echo "  levels=[$LNHM_LEVELS] steps/epoch=$LNHM_STEPS_PER_EPOCH max_epochs=$LNHM_MAX_EPOCHS"
echo "  seed=$LNHM_SEED device=$LNHM_DEVICE generate=$LNHM_GENERATE_DATA"
echo "  data=$LNHM_DATA_DIR output=$LNHM_OUTPUT_DIR"

# --- Dataset: regenerate deterministically (seed-fixed) when needed ---
needs_generation=false
case "$LNHM_GENERATE_DATA" in
  always) needs_generation=true ;;
  never)  needs_generation=false ;;
  *)      [ -z "$(ls -A "$LNHM_DATA_DIR" 2>/dev/null || true)" ] && needs_generation=true ;;
esac
if [ "$needs_generation" = true ]; then
  echo "--- generating dataset ---"
  # shellcheck disable=SC2086
  python data/generate.py --output-dir "$LNHM_DATA_DIR" --levels $LNHM_LEVELS
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

# --- Train ---
echo "--- training ---"
# shellcheck disable=SC2086
python training/train.py \
  --data-dir "$LNHM_DATA_DIR" \
  --levels $LNHM_LEVELS \
  --steps-per-epoch "$LNHM_STEPS_PER_EPOCH" \
  --max-epochs-per-level "$LNHM_MAX_EPOCHS" \
  --seed "$LNHM_SEED" \
  --device "$LNHM_DEVICE" \
  --output-dir "$LNHM_OUTPUT_DIR"

# --- Plot ---
echo "--- plotting ---"
python analysis/plot.py \
  --metrics "$LNHM_OUTPUT_DIR/metrics.csv" \
  --out "$LNHM_OUTPUT_DIR/accuracy_by_level.png" || echo "  (plot failed, continuing)"

# --- Self-exfiltrate results (FlightDeck has no artifact retrieval yet) ---
if [ -n "$LNHM_RESULT_UPLOAD_URL" ]; then
  echo "--- uploading results ---"
  results_tarball="/tmp/lnhm-results-seed${LNHM_SEED}.tar.gz"
  tar -czf "$results_tarball" -C "$LNHM_OUTPUT_DIR" .
  if curl -fsS -T "$results_tarball" "$LNHM_RESULT_UPLOAD_URL"; then
    echo "  uploaded -> $LNHM_RESULT_UPLOAD_URL"
  else
    echo "  UPLOAD FAILED (exit $?)"
  fi
fi

echo "=== done; artifacts in $LNHM_OUTPUT_DIR ==="
ls -la "$LNHM_OUTPUT_DIR"
