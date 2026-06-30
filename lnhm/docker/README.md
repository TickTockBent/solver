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

## Path B — direct image (vast.ai-native API, RunPod, local GPU)

For providers that pull a prebuilt image, use the repo `Dockerfile` (CUDA +
PyTorch base, deps baked in, `entrypoint.sh` as ENTRYPOINT):

```bash
docker build -t <registry>/lnhm-phase0:latest -f Dockerfile .
docker push  <registry>/lnhm-phase0:latest        # must be pull-able by the instance
docker run --gpus all \
  -e LNHM_LEVELS="3 4 5 6 7 8 9 10 11 12" -e LNHM_MAX_EPOCHS=150 \
  -v "$PWD/outputs:/workspace/outputs" <registry>/lnhm-phase0:latest
```

Local GPU smoke test (no provider): same `docker run` on the box with the GPU.

## Environment variables (both paths)

| Var | Default | Meaning |
|-----|---------|---------|
| `LNHM_LEVELS` | `3 4 … 12` | curriculum levels |
| `LNHM_STEPS_PER_EPOCH` | `100` | optimizer steps per epoch |
| `LNHM_MAX_EPOCHS` | `150` | max epochs per level before forced advance |
| `LNHM_SEED` | `0` | RNG seed (vary for multi-seed runs) |
| `LNHM_DEVICE` | `cuda` | `cuda` / `cpu` |
| `LNHM_GENERATE_DATA` | `auto` | `auto` (gen if missing) / `always` / `never` |
| `LNHM_RESULT_UPLOAD_URL` | — | optional HTTP PUT target for the results tarball |
