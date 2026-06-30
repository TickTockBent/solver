# LNHM Phase 0 training image.
#
# This image is the DIRECT-IMAGE path (vast.ai-native API, RunPod, local GPU box):
# providers that pull a prebuilt image and run it. NOTE: FlightDeck does NOT use a
# custom image — it runs a stock base_image + your code bundle + a pip-installed
# requirements file (see docker/README.md, "Path A"). Use this Dockerfile for
# everything except FlightDeck.
#
# Items marked [FLIGHTDECK] noted only for cross-reference.
#
# Build:   docker build -t <registry>/lnhm-phase0:<tag> -f Dockerfile .
# Run:     docker run --gpus all -e LNHM_LEVELS="3 4 5 6 7 8 9 10 11 12" \
#                 -v "$PWD/outputs:/workspace/outputs" <image>

# [FLIGHTDECK] Pin the CUDA tag to match the GPUs the platform schedules on
# vast.ai (driver/CUDA compatibility). Verify this tag exists before relying on it.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /workspace/lnhm

# Runtime dependencies beyond torch (the CUDA torch build ships in the base image,
# so we deliberately do NOT pip install torch here).
RUN pip install --no-cache-dir "numpy>=1.24" "pyyaml>=6.0" "matplotlib>=3.7"

# Project source (data/, model/, training/, analysis/, configs/, docker/).
COPY . /workspace/lnhm

# Default data + artifact locations. [FLIGHTDECK] Confirm where the sidecar
# collects artifacts from and point LNHM_OUTPUT_DIR there (or mount it).
ENV LNHM_DATA_DIR=/workspace/data/phase0 \
    LNHM_OUTPUT_DIR=/workspace/outputs \
    LNHM_DEVICE=cuda \
    LNHM_GENERATE_DATA=auto

# Strip any CR (Windows checkouts via git autocrlf give the entrypoint CRLF line
# endings, which breaks the shebang: "/usr/bin/env: 'bash\r': No such file").
RUN sed -i 's/\r$//' /workspace/lnhm/docker/entrypoint.sh \
    && chmod +x /workspace/lnhm/docker/entrypoint.sh \
    && mkdir -p "$LNHM_DATA_DIR" "$LNHM_OUTPUT_DIR"

# OPTIONAL: bake the dataset into the image at build time so the (expensive) GPU
# instance starts training immediately instead of spending minutes generating
# data on CPU. Uncomment to trade a slower build for faster job startup:
# RUN python data/generate.py --output-dir "$LNHM_DATA_DIR"

ENTRYPOINT ["/workspace/lnhm/docker/entrypoint.sh"]
