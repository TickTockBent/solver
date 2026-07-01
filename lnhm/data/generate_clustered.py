"""Clustered (non-uniform) Euclidean TSP instances for the Phase 3 elastic-k probe.

Same schema, determinism, and labeling pipeline as ``generate.py`` -- the ONLY
difference is the coordinate sampler: instead of uniform-random points, each
instance is a Gaussian mixture of a few tight blobs. This is the geometry that
fixed-k composition cuts artificial seams through, and the question is whether the
model's quality holds on it (dense blobs are OFF the uniform training distribution).

Reference labels: Held-Karp (exact) for n<=12, LKH-3 (near-optimal) for n>12, via
``--solver auto`` -- identical to how ``data/phase0_test`` (the uniform arm) was
labeled, so gaps are directly comparable. Instances are fully determined by
(base_seed, level, split, index, clusters, sigma); the default base_seed (777) is
disjoint from training (0) and the uniform test set (12345).

Examples
--------
    # Clustered test set at the phase0_test levels, 200/level, LKH for n>12:
    python data/generate_clustered.py --output-dir data/phase3_clustered \
        --levels 8 10 12 16 20 25 30 --count 200 \
        --clusters 3 --sigma 0.05 \
        --lkh-binary /home/ticktockbent/tools/LKH-3.0.13/LKH
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import uuid
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from held_karp import canonicalize_tour, held_karp, tour_distance  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COORDINATE_DECIMALS = 6


def instance_seed(level: int, split: str, index: int, base_seed: int) -> int:
    """Deterministic per-instance RNG seed. Distinct namespace from generate.py
    via the ':clustered' tag so a clustered instance never collides with a uniform
    one at the same (base_seed, level, split, index)."""
    key = f"{base_seed}:{level}:{split}:{index}:clustered"
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_instance_id(level: int, split: str, index: int, base_seed: int) -> str:
    digest = hashlib.md5(f"{base_seed}:{level}:{split}:{index}:clustered".encode()).digest()
    return str(uuid.UUID(bytes=digest))


def sample_clustered_coordinates(
    rng: np.random.Generator, n: int, clusters: int, sigma: float
) -> np.ndarray:
    """Gaussian-mixture blobs in the unit square.

    `clusters` centers are drawn uniformly (kept away from the border by `sigma` so
    a full blob fits), each of `n` points is assigned to a random center and jittered
    by an isotropic Gaussian of std `sigma`, then clipped to [0,1]. Small `sigma` +
    few `clusters` => tight, well-separated clouds (the elastic-k target geometry)."""
    effective_clusters = max(1, min(clusters, n))
    margin = min(sigma * 2.0, 0.25)
    centers = rng.uniform(margin, 1.0 - margin, size=(effective_clusters, 2))
    assignment = rng.integers(0, effective_clusters, size=n)
    points = centers[assignment] + rng.normal(0.0, sigma, size=(n, 2))
    return np.clip(points, 0.0, 1.0)


def generate_single_instance(task: Tuple) -> Dict:
    level, split, index, base_seed, clusters, sigma, solver, lkh_binary = task
    rng = np.random.default_rng(instance_seed(level, split, index, base_seed))
    coordinates = np.round(
        sample_clustered_coordinates(rng, level, clusters, sigma), COORDINATE_DECIMALS
    )
    use_lkh = solver == "lkh" or (solver == "auto" and level > 12)
    if use_lkh:
        from analysis.baselines import lkh_tour
        optimal_tour = lkh_tour(coordinates, lkh_binary=lkh_binary)
        optimal_distance = tour_distance(coordinates, optimal_tour)
    else:
        optimal_tour, optimal_distance = held_karp(coordinates)
    canonical_tour = canonicalize_tour(optimal_tour)
    return {
        "id": deterministic_instance_id(level, split, index, base_seed),
        "n": level,
        "coords": coordinates.tolist(),
        "optimal_tour": canonical_tour,
        "optimal_distance": round(optimal_distance, COORDINATE_DECIMALS),
    }


def generate_level(
    level, split, count, base_seed, clusters, sigma, output_dir, num_workers, solver, lkh_binary
) -> Tuple[str, int, float]:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"level_{level:02d}_{split}.jsonl.gz")
    tasks = (
        (level, split, index, base_seed, clusters, sigma, solver, lkh_binary)
        for index in range(count)
    )
    start_time = time.monotonic()
    written = 0
    with gzip.open(output_path, "wt", encoding="utf-8") as output_file:
        with Pool(processes=num_workers) as pool:
            for instance in pool.imap_unordered(generate_single_instance, tasks, chunksize=16):
                output_file.write(json.dumps(instance) + "\n")
                written += 1
    return output_path, written, time.monotonic() - start_time


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clustered (blob) TSP instances.")
    parser.add_argument("--levels", type=int, nargs="+", default=[8, 10, 12, 16, 20, 25, 30])
    parser.add_argument("--output-dir", default="data/phase3_clustered")
    parser.add_argument("--count", type=int, default=200, help="Instances per level.")
    parser.add_argument("--clusters", type=int, default=3, help="Number of blobs per instance.")
    parser.add_argument("--sigma", type=float, default=0.05, help="Blob std-dev in unit square.")
    parser.add_argument("--base-seed", type=int, default=777,
                        help="Disjoint from training (0) and uniform test (12345).")
    parser.add_argument("--split", default="train",
                        help="Filename split tag (default 'train' to match eval_heldout).")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--solver", choices=["auto", "held_karp", "lkh"], default="auto")
    parser.add_argument("--lkh-binary", default="LKH", help="Path to LKH-3 binary (for n>12).")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    arguments = parse_arguments(argv)
    print(
        f"Clustered gen: levels {arguments.levels} -> '{arguments.output_dir}' "
        f"({arguments.count}/level, {arguments.clusters} blobs, sigma={arguments.sigma}, "
        f"base_seed={arguments.base_seed})"
    )
    total_instances = 0
    total_seconds = 0.0
    for level in arguments.levels:
        path, written, seconds = generate_level(
            level, arguments.split, arguments.count, arguments.base_seed,
            arguments.clusters, arguments.sigma, arguments.output_dir,
            arguments.workers, arguments.solver, arguments.lkh_binary,
        )
        rate = written / seconds if seconds > 0 else float("inf")
        print(f"  n={level:>2} {written:>5} instances in {seconds:6.1f}s ({rate:7.1f}/s) -> {path}")
        total_instances += written
        total_seconds += seconds
    print(f"Done: {total_instances} instances in {total_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
