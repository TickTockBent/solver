"""Instance generation for LNHM Phase 0.

Generates uniform-random Euclidean TSP instances, solves each exactly with
Held-Karp, canonicalizes the optimal tour, and writes gzipped JSONL (one file
per level per split). Generation is parallelized across CPU cores; every
instance is fully determined by (base_seed, level, split, index), so a run is
reproducible and resumable.

Examples
--------
Generate the full Phase 0 dataset into ./data/phase0 :

    python data/generate.py --output-dir data/phase0

Validate the Held-Karp solver against brute force before generating :

    python data/generate.py --cross-check --cross-check-size 300

Generate a single small level for a smoke test :

    python data/generate.py --levels 5 --train-count 100 --val-count 20
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
from typing import Dict, Iterator, List, Tuple

import numpy as np

# Allow `python data/generate.py` to find the sibling solver module regardless
# of the current working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from held_karp import brute_force, canonicalize_tour, held_karp  # noqa: E402

# Coordinates are rounded so stored instances are compact and exactly
# reproducible; the solver runs on the rounded coordinates so the stored optimal
# distance is self-consistent with the stored coordinates.
COORDINATE_DECIMALS = 6

DEFAULT_LEVELS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
WARMUP_LEVELS = {3, 4}  # degenerate / near-degenerate; seed the model only


def default_split_counts(level: int) -> Tuple[int, int]:
    """(train, val) instance counts per level, matching the Phase 0 spec table."""
    if level in WARMUP_LEVELS:
        return 2_000, 500
    if level <= 7:
        return 10_000, 2_000
    return 50_000, 5_000


def instance_seed(level: int, split: str, index: int, base_seed: int) -> int:
    """Deterministic per-instance RNG seed derived from its coordinates."""
    digest = hashlib.sha256(f"{base_seed}:{level}:{split}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def deterministic_instance_id(level: int, split: str, index: int, base_seed: int) -> str:
    digest = hashlib.md5(f"{base_seed}:{level}:{split}:{index}".encode()).digest()
    return str(uuid.UUID(bytes=digest))


def generate_single_instance(task: Tuple[int, str, int, int]) -> Dict:
    """Worker function: build one solved, canonicalized instance."""
    level, split, index, base_seed = task
    rng = np.random.default_rng(instance_seed(level, split, index, base_seed))
    coordinates = np.round(rng.random((level, 2)), COORDINATE_DECIMALS)
    optimal_tour, optimal_distance = held_karp(coordinates)
    canonical_tour = canonicalize_tour(optimal_tour)
    return {
        "id": deterministic_instance_id(level, split, index, base_seed),
        "n": level,
        "coords": coordinates.tolist(),
        "optimal_tour": canonical_tour,
        "optimal_distance": round(optimal_distance, COORDINATE_DECIMALS),
    }


def generate_split(
    level: int,
    split: str,
    count: int,
    base_seed: int,
    output_dir: str,
    num_workers: int,
) -> Tuple[str, int, float]:
    """Generate one level/split to gzipped JSONL. Returns (path, count, seconds)."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"level_{level:02d}_{split}.jsonl.gz")
    tasks = ((level, split, index, base_seed) for index in range(count))

    start_time = time.monotonic()
    instances_written = 0
    with gzip.open(output_path, "wt", encoding="utf-8") as output_file:
        with Pool(processes=num_workers) as pool:
            for instance in pool.imap_unordered(
                generate_single_instance, tasks, chunksize=64
            ):
                output_file.write(json.dumps(instance) + "\n")
                instances_written += 1
    elapsed_seconds = time.monotonic() - start_time
    return output_path, instances_written, elapsed_seconds


def cross_check(level: int, sample_size: int, base_seed: int) -> int:
    """Compare Held-Karp against the brute-force oracle. Returns mismatch count."""
    DISTANCE_TOLERANCE = 1e-6
    mismatches = 0
    for index in range(sample_size):
        rng = np.random.default_rng(instance_seed(level, "crosscheck", index, base_seed))
        coordinates = np.round(rng.random((level, 2)), COORDINATE_DECIMALS)
        _, held_karp_distance = held_karp(coordinates)
        _, brute_force_distance = brute_force(coordinates)
        if abs(held_karp_distance - brute_force_distance) > DISTANCE_TOLERANCE:
            mismatches += 1
            print(
                f"  MISMATCH n={level} index={index}: "
                f"held_karp={held_karp_distance:.6f} brute_force={brute_force_distance:.6f}"
            )
    return mismatches


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LNHM Phase 0 TSP instances.")
    parser.add_argument(
        "--levels", type=int, nargs="+", default=DEFAULT_LEVELS,
        help="Instance sizes (n) to generate. Default: 3..12.",
    )
    parser.add_argument(
        "--output-dir", default="data/phase0",
        help="Directory for the gzipped JSONL output files.",
    )
    parser.add_argument(
        "--base-seed", type=int, default=0,
        help="Base seed; instances are reproducible given this value.",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of worker processes. Default: all cores.",
    )
    parser.add_argument(
        "--train-count", type=int, default=None,
        help="Override the per-level training instance count for all levels.",
    )
    parser.add_argument(
        "--val-count", type=int, default=None,
        help="Override the per-level validation instance count for all levels.",
    )
    parser.add_argument(
        "--cross-check", action="store_true",
        help="Validate Held-Karp against brute force, then exit without generating.",
    )
    parser.add_argument(
        "--cross-check-size", type=int, default=200,
        help="Instances per level to cross-check (levels <= 10 only).",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    arguments = parse_arguments(argv)

    if arguments.cross_check:
        print("Cross-checking Held-Karp against brute force...")
        total_mismatches = 0
        for level in arguments.levels:
            if level > 10:
                print(f"  n={level}: skipped (brute force infeasible above n=10)")
                continue
            mismatches = cross_check(level, arguments.cross_check_size, arguments.base_seed)
            status = "OK" if mismatches == 0 else f"{mismatches} MISMATCHES"
            print(f"  n={level}: {arguments.cross_check_size} instances -> {status}")
            total_mismatches += mismatches
        if total_mismatches:
            print(f"FAILED: {total_mismatches} total mismatches")
            return 1
        print("PASSED: Held-Karp matches brute force on all sampled instances")
        return 0

    print(
        f"Generating levels {arguments.levels} into '{arguments.output_dir}' "
        f"with {arguments.workers} workers (base_seed={arguments.base_seed})"
    )
    grand_total_instances = 0
    grand_total_seconds = 0.0
    for level in arguments.levels:
        default_train, default_val = default_split_counts(level)
        train_count = arguments.train_count if arguments.train_count is not None else default_train
        val_count = arguments.val_count if arguments.val_count is not None else default_val
        for split, count in (("train", train_count), ("val", val_count)):
            path, written, seconds = generate_split(
                level, split, count, arguments.base_seed,
                arguments.output_dir, arguments.workers,
            )
            rate = written / seconds if seconds > 0 else float("inf")
            print(
                f"  n={level:>2} {split:<5} {written:>6} instances "
                f"in {seconds:6.1f}s ({rate:8.0f}/s) -> {path}"
            )
            grand_total_instances += written
            grand_total_seconds += seconds

    print(
        f"Done: {grand_total_instances} instances in {grand_total_seconds:.1f}s total"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
