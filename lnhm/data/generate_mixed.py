"""Mixed-distribution TSP instances for the Phase 3 data-diversity retrain.

Produces a training diet that spans THREE geometries so the model learns to handle
arbitrary bounded point sets, not just uniform ones:

  - uniform   : i.i.d. uniform in the unit square (even density; the old diet).
  - clustered : Gaussian-mixture blobs with RANDOMIZED cluster count and spread per
                instance (k in [2,6], sigma in [0.03,0.15]) -- deliberately NOT the
                fixed k=3/sigma=0.05 of the held-out test set, so the test geometry is
                inside the training span but never memorized.
  - random    : "truly random, spatially bounded" grab-bag -- random component count,
                anisotropic (stretched, rotated) blobs, and a random uniform-background
                fraction. Spans the whole uniform<->clumpy continuum.

`--distribution mixed` (default) picks one of the three per instance, so a single
level file is a balanced blend. Same schema/labeling as generate.py (Held-Karp n<=12,
LKH n>12) so the output is drop-in for train.py and eval_heldout.py. Determinism keyed
on (base_seed, level, split, index) in a ':mixed' namespace, disjoint from the uniform
(generate.py) and clustered-test (generate_clustered.py, base_seed=777) instances.

Example (levels 3-12 training pool, train+val, Held-Karp only):
    python data/generate_mixed.py --output-dir data/phase3_mixed \
        --levels 3 4 5 6 7 8 9 10 11 12 --train-count 4000 --val-count 500
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
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
DISTRIBUTIONS = ("uniform", "clustered", "random")


def instance_seed(level: int, split: str, index: int, base_seed: int) -> int:
    key = f"{base_seed}:{level}:{split}:{index}:mixed"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def deterministic_instance_id(level: int, split: str, index: int, base_seed: int) -> str:
    digest = hashlib.md5(f"{base_seed}:{level}:{split}:{index}:mixed".encode()).digest()
    return str(uuid.UUID(bytes=digest))


def sample_uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.random((n, 2))


def sample_clustered(rng: np.random.Generator, n: int, clusters: int, sigma: float) -> np.ndarray:
    """Isotropic Gaussian-mixture blobs (same shape as generate_clustered)."""
    effective_clusters = max(1, min(clusters, n))
    margin = min(sigma * 2.0, 0.25)
    centers = rng.uniform(margin, 1.0 - margin, size=(effective_clusters, 2))
    assignment = rng.integers(0, effective_clusters, size=n)
    points = centers[assignment] + rng.normal(0.0, sigma, size=(n, 2))
    return np.clip(points, 0.0, 1.0)


def sample_random_bounded(rng: np.random.Generator, n: int) -> np.ndarray:
    """Grab-bag: random # of anisotropic (stretched/rotated) blobs + uniform background.

    No fixed structure -- a random fraction of the points is pure-uniform noise and the
    rest belongs to randomly-shaped, randomly-oriented components. Clipped to the box."""
    background_fraction = rng.uniform(0.0, 0.5)
    num_background = int(round(background_fraction * n))
    num_structured = n - num_background

    points = []
    if num_structured > 0:
        num_components = int(rng.integers(1, max(2, num_structured // 2 + 1)))
        centers = rng.uniform(0.0, 1.0, size=(num_components, 2))
        assignment = rng.integers(0, num_components, size=num_structured)
        for point_index in range(num_structured):
            component = assignment[point_index]
            scale_x = rng.uniform(0.02, 0.25)
            scale_y = rng.uniform(0.02, 0.25)
            theta = rng.uniform(0.0, math.pi)
            local = np.array([rng.normal(0.0, scale_x), rng.normal(0.0, scale_y)])
            rotation = np.array([[math.cos(theta), -math.sin(theta)],
                                 [math.sin(theta), math.cos(theta)]])
            points.append(centers[component] + rotation @ local)
    if num_background > 0:
        points.extend(rng.uniform(0.0, 1.0, size=(num_background, 2)))

    coordinates = np.asarray(points, dtype=np.float64)
    rng.shuffle(coordinates)  # don't leave structured points before background ones
    return np.clip(coordinates, 0.0, 1.0)


def sample_mixed(rng: np.random.Generator, n: int, distribution: str) -> Tuple[np.ndarray, str]:
    """Return (coordinates, chosen_distribution). 'mixed' picks one uniformly per call."""
    chosen = distribution
    if distribution == "mixed":
        chosen = DISTRIBUTIONS[int(rng.integers(0, len(DISTRIBUTIONS)))]
    if chosen == "uniform":
        return sample_uniform(rng, n), chosen
    if chosen == "clustered":
        clusters = int(rng.integers(2, 7))
        sigma = float(rng.uniform(0.03, 0.15))
        return sample_clustered(rng, n, clusters, sigma), chosen
    if chosen == "random":
        return sample_random_bounded(rng, n), chosen
    raise ValueError(f"unknown distribution: {distribution}")


def generate_single_instance(task: Tuple) -> Dict:
    level, split, index, base_seed, distribution, solver, lkh_binary = task
    rng = np.random.default_rng(instance_seed(level, split, index, base_seed))
    raw_coordinates, chosen = sample_mixed(rng, level, distribution)
    coordinates = np.round(raw_coordinates, COORDINATE_DECIMALS)
    use_lkh = solver == "lkh" or (solver == "auto" and level > 12)
    if use_lkh:
        from analysis.baselines import lkh_tour
        optimal_tour = lkh_tour(coordinates, lkh_binary=lkh_binary)
        optimal_distance = tour_distance(coordinates, optimal_tour)
    else:
        optimal_tour, optimal_distance = held_karp(coordinates)
    return {
        "id": deterministic_instance_id(level, split, index, base_seed),
        "n": level,
        "dist": chosen,
        "coords": coordinates.tolist(),
        "optimal_tour": canonicalize_tour(optimal_tour),
        "optimal_distance": round(optimal_distance, COORDINATE_DECIMALS),
    }


def generate_level(level, split, count, base_seed, distribution, output_dir,
                   num_workers, solver, lkh_binary) -> Tuple[str, int, float, Dict[str, int]]:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"level_{level:02d}_{split}.jsonl.gz")
    tasks = ((level, split, index, base_seed, distribution, solver, lkh_binary)
             for index in range(count))
    start_time = time.monotonic()
    written = 0
    distribution_counts: Dict[str, int] = {}
    with gzip.open(output_path, "wt", encoding="utf-8") as output_file:
        with Pool(processes=num_workers) as pool:
            for instance in pool.imap_unordered(generate_single_instance, tasks, chunksize=16):
                output_file.write(json.dumps(instance) + "\n")
                written += 1
                distribution_counts[instance["dist"]] = distribution_counts.get(instance["dist"], 0) + 1
    return output_path, written, time.monotonic() - start_time, distribution_counts


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mixed-distribution TSP instances.")
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    parser.add_argument("--output-dir", default="data/phase3_mixed")
    parser.add_argument("--train-count", type=int, default=4000)
    parser.add_argument("--val-count", type=int, default=500)
    parser.add_argument("--distribution", choices=["mixed", *DISTRIBUTIONS], default="mixed")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--solver", choices=["auto", "held_karp", "lkh"], default="auto")
    parser.add_argument("--lkh-binary", default="LKH")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    arguments = parse_arguments(argv)
    print(f"Mixed gen ({arguments.distribution}): levels {arguments.levels} -> "
          f"'{arguments.output_dir}' (base_seed={arguments.base_seed})")
    grand_counts: Dict[str, int] = {}
    total_seconds = 0.0
    for level in arguments.levels:
        for split, count in (("train", arguments.train_count), ("val", arguments.val_count)):
            path, written, seconds, counts = generate_level(
                level, split, count, arguments.base_seed, arguments.distribution,
                arguments.output_dir, arguments.workers, arguments.solver, arguments.lkh_binary,
            )
            for key, value in counts.items():
                grand_counts[key] = grand_counts.get(key, 0) + value
            total_seconds += seconds
            mix = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            print(f"  n={level:>2} {split:<5} {written:>5} in {seconds:6.1f}s  [{mix}] -> {path}")
    print(f"Done in {total_seconds:.1f}s. Distribution totals: {dict(sorted(grand_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
