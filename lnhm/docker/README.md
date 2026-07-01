# Offloading LNHM training to a GPU provider

Two deployment paths. They share `docker/entrypoint.sh` (env-driven: regenerate
data → train → plot → optional self-upload of results).

## Path A — FlightDeck (ThermAI_FlightDeck CLI)

FlightDeck does **not** build images. It runs a stock `base_image`, extracts your
code bundle into `/workspace`, `pip install`s a requirements file, and runs an
entrypoint command. So for FlightDeck we ship **code + requirements + command**,
not a Dockerfile.

```bash
flightdeck submit "bash docker/entrypoint.sh" \
  --name lnhm-phase0-full \
  --files . \
  --requirements docker/requirements-flightdeck.txt \
  --base-image pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
  --gpu RTX4090 --gpu-count 1 --time 1h \
  --env LNHM_LEVELS="3 4 5 6 7 8 9 10 11 12" \
  --env LNHM_MAX_EPOCHS=150 \
  --env LNHM_DEVICE=cuda \
  --env LNHM_GENERATE_DATA=auto \
  --env LNHM_RESULT_UPLOAD_URL="https://<your-presigned-PUT-target>"
```

### Three things to know before this works end-to-end

1. **vast.ai is not implemented in FlightDeck yet.** The codebase currently routes
   to Lambda Cloud and ECS only (`ACTIVE_PROVIDERS=lambda`); the vast.ai provider
   is specced but not built. So today this runs on Lambda/ECS, not vast.ai.
2. **FlightDeck retrieves no artifacts and captures no stdout.** Only sidecar
   telemetry (GPU util/temp/power) comes back. That is why the entrypoint
   self-uploads its results tarball when `LNHM_RESULT_UPLOAD_URL` is set — without
   it, metrics.csv / checkpoint / plot stay on the (ephemeral) instance and are
   lost. Set that URL to an S3/GCS presigned PUT, transfer.sh, or any HTTP PUT
   sink you control.
3. **Bundle hygiene.** `--files .` tarballs the directory (500 MB limit). Make sure
   `.venv/` (~1 GB with torch), `runs/`, and large `data/` are excluded from the
   bundle, or submit from a clean tree (e.g. `git archive`). Verify whether the
   CLI honors `.gitignore`/an ignore file before the first real submit.
   - The deterministic dataset (36 MB) MAY be bundled to skip in-instance
     regeneration — set `--env LNHM_GENERATE_DATA=never` if you bundle it. Default
     `auto` regenerates on the instance (costs a few GPU-minutes of CPU work).

## Path B — local GPU box (Windows/Linux + Docker + NVIDIA GPU)

Self-contained: build the image and run it on the machine with the GPU.

### 1. Get the project onto the box

Cleanest (no GitHub) — clone directly from the source host over SSH:

```powershell
git clone ssh://ticktockbent@10.0.20.72/home/ticktockbent/projects/experiments/solver
cd solver/lnhm
```

Offline fallback: on the source host run `git bundle create solver.bundle --all`,
copy the single file across, then `git clone solver.bundle solver`.

### 2. Build the image

```powershell
docker build -t lnhm-phase0 -f Dockerfile .
```

The dataset is regenerated inside the container on first run (deterministic). To
skip ~10–15 min of in-container CPU data-gen, mount a prebuilt dataset at run time
(`-v <host-data>:/workspace/lnhm/data/phase0 -e LNHM_GENERATE_DATA=never`) or
uncomment the data-gen `RUN` in the Dockerfile.

### 3. Run

GPU prerequisites on Windows: Docker Desktop on the **WSL2 backend** + a recent
NVIDIA driver (the NVIDIA Container Toolkit ships via WSL2); then `--gpus all` works.

**A/B/C/D cross-level matrix** (the Phase 0 gate, ~20 min on a 4090):

```powershell
docker run --gpus all `
  -e LNHM_TASK=crosslevel `
  -v ${PWD}/outputs:/workspace/outputs `
  lnhm-phase0
# -> outputs/cross_level_results.csv
```

**Full curriculum training** (metrics.csv + checkpoint + accuracy plot):

```powershell
docker run --gpus all `
  -e LNHM_TASK=train -e LNHM_MAX_EPOCHS=150 `
  -v ${PWD}/outputs:/workspace/outputs `
  lnhm-phase0
```

Override the matrix with e.g.
`-e LNHM_XLEVEL_ARGS="--anchors 5 9 11 --seeds 0 1 2 3 4 --total-steps 2500"`.

**Training a range of models (level / capacity sweep).** Each run writes to its own
subdir under `outputs/<run-name>/` (auto-named `L<min>-<max>_d<d_model>_s<seed>` if
`LNHM_RUN_NAME` is unset), so runs never clobber. Set the level range and/or model
size per run:

```powershell
# extend the curriculum to n=20 (image builds LKH, so it labels n>12 itself)
docker run --gpus all -e LNHM_TASK=train `
  -e LNHM_LEVELS="3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" `
  -e LNHM_RUN_NAME=L3-20_d128 `
  -v ${PWD}/outputs:/workspace/outputs lnhm-phase0

# a bigger model (capacity sweep): d_model=256, 4 layers
docker run --gpus all -e LNHM_TASK=train -e LNHM_D_MODEL=256 -e LNHM_N_LAYERS=4 `
  -e LNHM_RUN_NAME=L3-12_d256 `
  -v ${PWD}/outputs:/workspace/outputs lnhm-phase0
```

Loop those `docker run`s (varying `LNHM_LEVELS` / `LNHM_D_MODEL` / `LNHM_RUN_NAME`)
to produce a family of models.

**Exporting the trained model.** Each run's checkpoint is
`outputs/<run-name>/model_final.pt` on the host (via the mounted volume) — a
self-describing dict (`state_dict` + `model_config` + `levels`), so the loader
rebuilds the right architecture automatically. Copy it back to the source host
(it's ~3–11 MB), or set `-e LNHM_RESULT_UPLOAD_URL=<PUT-target>` to have the run
upload its results tarball itself.

(PowerShell: backtick line-continuation, `${PWD}`. cmd.exe: `%cd%`. bash/WSL:
`\` and `$PWD`.)

For providers that pull a prebuilt image (vast.ai-native, RunPod), additionally
`docker push <registry>/lnhm-phase0:latest` so the instance can pull it.

## Environment variables (both paths)

| Var | Default | Meaning |
|-----|---------|---------|
| `LNHM_TASK` | `train` | `train` (curriculum) or `crosslevel` (A/B/C/D matrix) |
| `LNHM_XLEVEL_ARGS` | — | extra args passed to `cross_level.py` (anchors/seeds/steps) |
| `LNHM_RUN_NAME` | auto | names the `outputs/<run-name>/` subdir (sweeps don't clobber) |
| `LNHM_D_MODEL` / `LNHM_N_LAYERS` / `LNHM_N_HEADS` / `LNHM_FF_DIM` | config | model-size overrides (capacity sweep) |
| `LNHM_LKH_BINARY` | `LKH` | LKH-3 binary (labels n>12 during data-gen) |
| `LNHM_LEVELS` | `3 4 … 12` | curriculum levels |
| `LNHM_STEPS_PER_EPOCH` | `100` | optimizer steps per epoch |
| `LNHM_MAX_EPOCHS` | `150` | max epochs per level before forced advance |
| `LNHM_SEED` | `0` | RNG seed (vary for multi-seed runs) |
| `LNHM_DEVICE` | `cuda` | `cuda` / `cpu` |
| `LNHM_GENERATE_DATA` | `auto` | `auto` (gen if missing) / `always` / `never` |
| `LNHM_RESULT_UPLOAD_URL` | — | optional HTTP PUT target for the results tarball |
