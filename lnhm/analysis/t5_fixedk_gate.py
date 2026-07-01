"""Phase 3 T5 sub-question 2 — the fixed-k damage gate (elastic-k go/no-go).

Question: what does the CURRENT fixed-k composition stack actually lose on globby data,
before we build elastic-k? If the loss is < ~1pp of p, elastic-k isn't justified by data
yet; if it's large, the build is motivated by measurement instead of intuition.

T1 established that the leaf solver is irrelevant after fast_local cleanup, so any excess
loss of fixed-k composition on clustered data must come from the PARTITION (seams cut
through dense blobs), not the leaf. To isolate that, we hold leaf = Held-Karp (exact) and
cleanup = fast_local fixed, and compare on uniform vs clustered globals:

    fixed-k composition (k=10)     -- imposes uniform-size cluster seams
    SFC + fast_local               -- partition-free reference
    NN  + fast_local               -- partition-free reference (n<=10k; O(n^2))

Damage = how much longer fixed-k composition's tour is than the best partition-free
baseline, on each distribution. On uniform this should be ~0 (T1/Phase1: cmp+fast approx
SFC+fast). Excess on clustered = the fixed-k seam penalty. Optional LKH gives absolute p.

    python analysis/t5_fixedk_gate.py --sizes 10000 --seeds 5 [--lkh /path/to/LKH]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from analysis.baselines import nearest_neighbor, space_filling_curve, tour_distance  # noqa: E402
from analysis.compose import compose_solve, held_karp_solver, partition_space_filling  # noqa: E402
from analysis.fast_local_search import fast_local_search  # noqa: E402


def sample_uniform(rng, n):
    return rng.random((n, 2))


def sample_clustered_scale(rng, n, points_per_blob=40, sigma=0.012):
    """Tight blobs of ~points_per_blob each -- larger than k=10, so fixed-k must cut
    each blob into several clusters (the seam-damage scenario)."""
    num_blobs = max(1, n // points_per_blob)
    centers = rng.uniform(0.0, 1.0, size=(num_blobs, 2))
    assignment = rng.integers(0, num_blobs, size=n)
    points = centers[assignment] + rng.normal(0.0, sigma, size=(n, 2))
    return np.clip(points, 0.0, 1.0)


def timed(function, *args):
    start = time.process_time()
    result = function(*args)
    return result, time.process_time() - start


def composition_pipeline(coordinates, k_cap):
    return compose_solve(coordinates, k_cap, local_solver=held_karp_solver,
                         partitioner=partition_space_filling, cleanup="fast_local")


def sfc_fast_pipeline(coordinates):
    return fast_local_search(coordinates, space_filling_curve(coordinates))


def nn_fast_pipeline(coordinates):
    return fast_local_search(coordinates, nearest_neighbor(coordinates))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--k-cap", type=int, default=10)
    parser.add_argument("--points-per-blob", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=0.012)
    parser.add_argument("--nn-max", type=int, default=20000, help="Skip NN+fast above this n.")
    parser.add_argument("--lkh", default=None, help="LKH binary path for absolute p (n<=10k).")
    parser.add_argument("--lkh-max-trials", type=int, default=1000)
    arguments = parser.parse_args()

    distributions = {"uniform": sample_uniform,
                     "clustered": lambda rng, n: sample_clustered_scale(
                         rng, n, arguments.points_per_blob, arguments.sigma)}

    print(f"# T5 fixed-k damage gate (k_cap={arguments.k_cap}, leaf=Held-Karp, "
          f"cleanup=fast_local, blob~{arguments.points_per_blob}pts, sigma={arguments.sigma})\n")

    for size in arguments.sizes:
        run_nn = size <= arguments.nn_max
        print(f"### n={size}  ({arguments.seeds} seeds){'  [NN skipped: too large]' if not run_nn else ''}")
        header = f"{'dist':>10} {'pipeline':>16} {'len':>10} {'cpu_s':>8} {'vs_best%':>9}"
        if arguments.lkh:
            header += f" {'gap_lkh%':>9} {'p':>7}"
        print(header)
        print("-" * len(header))

        for dist_name, sampler in distributions.items():
            per_pipeline = {"composition": [], "sfc_fast": []}
            if run_nn:
                per_pipeline["nn_fast"] = []
            lkh_lengths = []
            cpu_accumulator = {name: [] for name in per_pipeline}

            for seed in range(arguments.seeds):
                rng = np.random.default_rng(4242 + 101 * seed)
                coordinates = sampler(rng, size).astype(np.float64)

                tour, cpu = timed(composition_pipeline, coordinates, arguments.k_cap)
                per_pipeline["composition"].append(tour_distance(coordinates, tour))
                cpu_accumulator["composition"].append(cpu)

                tour, cpu = timed(sfc_fast_pipeline, coordinates)
                per_pipeline["sfc_fast"].append(tour_distance(coordinates, tour))
                cpu_accumulator["sfc_fast"].append(cpu)

                if run_nn:
                    tour, cpu = timed(nn_fast_pipeline, coordinates)
                    per_pipeline["nn_fast"].append(tour_distance(coordinates, tour))
                    cpu_accumulator["nn_fast"].append(cpu)

                if arguments.lkh:
                    from analysis.baselines import lkh_tour
                    lkh_t = lkh_tour(coordinates, lkh_binary=arguments.lkh,
                                     max_trials=arguments.lkh_max_trials)
                    lkh_lengths.append(tour_distance(coordinates, lkh_t))

            mean_len = {name: float(np.mean(vals)) for name, vals in per_pipeline.items()}
            best_partition_free = min(
                mean_len[name] for name in ("sfc_fast", "nn_fast") if name in mean_len)
            lkh_mean = float(np.mean(lkh_lengths)) if lkh_lengths else None

            for name in per_pipeline:
                length = mean_len[name]
                cpu_seconds = float(np.mean(cpu_accumulator[name]))
                vs_best = (length / best_partition_free - 1.0) * 100.0
                row = f"{dist_name:>10} {name:>16} {length:>10.3f} {cpu_seconds:>8.2f} {vs_best:>+8.2f}%"
                if arguments.lkh and lkh_mean:
                    gap = (length / lkh_mean - 1.0) * 100.0
                    row += f" {gap:>+8.2f}% {1.0/(1.0+gap/100.0):>7.3f}"
                print(row, flush=True)
            print()

    print("# Gate: 'vs_best%' for composition is the fixed-k damage. Read the CLUSTERED")
    print("# composition row: <~1.1% => elastic-k not justified; large => motivated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
