"""Phase 3 T1 — trivial local-solver ablation (the null-hypothesis test).

Holds the composition pipeline FIXED (same partitioner, same k_cap, same fast_local
cleanup) and swaps ONLY the local (leaf) solver:

    random permutation | nearest-neighbor | model (greedy) | Held-Karp exact

Question: does the model contribute anything to STATIC composition, or is the story
just "good clustering + good cleanup"? Reports, per (solver, n): pre-cleanup p (how
much the leaf solver matters raw), post-cleanup p (whether cleanup erases the
difference), and f = cpu-core-seconds per 1e6 cities / p. Reference is BHH
(0.7124*sqrt(n)) in the unit square, consistent with the Phase 1 scale rows.

Pre-registered verdict: if p(model) - p(NN) < 0.005 AFTER cleanup, the model is inert
in static composition.

    python analysis/t1_ablation.py --sizes 10000 100000 --seeds 3 \
        --model-checkpoint runs/full/model_selfdesc.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from analysis.baselines import nearest_neighbor, tour_distance  # noqa: E402
from analysis.compose import (  # noqa: E402
    compose_solve, held_karp_solver, make_model_batch_solver, partition_space_filling,
)
from analysis.fast_local_search import fast_local_search  # noqa: E402

BHH_CONSTANT = 0.7124  # expected optimal tour length ~ BHH_CONSTANT * sqrt(n) in [0,1]^2


def make_random_solver(seed: int):
    """A deterministic random-permutation leaf solver (the true null)."""
    generator = np.random.default_rng(seed)
    def solve(cluster_coordinates: np.ndarray):
        return list(generator.permutation(len(cluster_coordinates)))
    return solve


def nearest_neighbor_solver(cluster_coordinates: np.ndarray):
    return nearest_neighbor(np.asarray(cluster_coordinates, dtype=np.float64))


def proximity(tour_length: float, reference_length: float) -> float:
    """p = L*/L_ours, clamped to (0,1]; reference is the (approx-optimal) BHH length."""
    return reference_length / tour_length if tour_length > 0 else 0.0


def run_one(coordinates, k_cap, solver_kind, model_batch_solver, leaf_seed):
    """Construct (timed) then clean up (timed). Returns dict of gaps + cpu-seconds."""
    local_solver = None
    batch_solver = None
    if solver_kind == "random":
        local_solver = make_random_solver(leaf_seed)
    elif solver_kind == "nn":
        local_solver = nearest_neighbor_solver
    elif solver_kind == "held_karp":
        local_solver = held_karp_solver
    elif solver_kind == "model":
        batch_solver = model_batch_solver
    else:
        raise ValueError(solver_kind)

    construct_start = time.process_time()
    raw_tour = compose_solve(
        coordinates, k_cap, local_solver=local_solver or held_karp_solver,
        partitioner=partition_space_filling, cleanup="none",
        batch_local_solver=batch_solver,
    )
    construct_cpu = time.process_time() - construct_start
    raw_length = tour_distance(coordinates, raw_tour)

    cleanup_start = time.process_time()
    clean_tour = fast_local_search(coordinates, raw_tour)
    cleanup_cpu = time.process_time() - cleanup_start
    clean_length = tour_distance(coordinates, clean_tour)

    assert sorted(clean_tour) == list(range(len(coordinates))), "invalid tour"
    return {
        "raw_length": raw_length, "clean_length": clean_length,
        "construct_cpu": construct_cpu, "cleanup_cpu": cleanup_cpu,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[10000, 100000])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--k-cap", type=int, default=10)
    parser.add_argument("--model-checkpoint", default="runs/full/model_selfdesc.pt")
    parser.add_argument("--solvers", nargs="+",
                        default=["random", "nn", "model", "held_karp"])
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    model_batch_solver = make_model_batch_solver(arguments.model_checkpoint, device=arguments.device)

    print(f"# T1 local-solver ablation (k_cap={arguments.k_cap}, cleanup=fast_local, "
          f"ref=BHH, {arguments.seeds} seeds)\n")
    header = f"{'n':>9} {'solver':>10} {'p_raw':>7} {'p_clean':>8} {'cpu_s':>8} {'f':>9}"
    print(header)
    print("-" * len(header))

    results = {}
    for size in arguments.sizes:
        reference_length = BHH_CONSTANT * (size ** 0.5)
        for solver_kind in arguments.solvers:
            raw_ps, clean_ps, cpu_totals = [], [], []
            for seed in range(arguments.seeds):
                coordinates = np.random.default_rng(1000 * seed + 7).random((size, 2))
                metrics = run_one(coordinates, arguments.k_cap, solver_kind,
                                  model_batch_solver, leaf_seed=seed)
                raw_ps.append(proximity(metrics["raw_length"], reference_length))
                clean_ps.append(proximity(metrics["clean_length"], reference_length))
                cpu_totals.append(metrics["construct_cpu"] + metrics["cleanup_cpu"])
            p_raw = float(np.mean(raw_ps))
            p_clean = float(np.mean(clean_ps))
            cpu_seconds = float(np.mean(cpu_totals))
            f_value = (cpu_seconds * (1e6 / size)) / p_clean if p_clean > 0 else float("inf")
            results[(size, solver_kind)] = (p_raw, p_clean, cpu_seconds, f_value)
            print(f"{size:>9} {solver_kind:>10} {p_raw:>7.3f} {p_clean:>8.3f} "
                  f"{cpu_seconds:>8.2f} {f_value:>9.1f}", flush=True)

    print("\n# Verdict check (post-cleanup p(model) - p(NN); <0.005 => model inert):")
    for size in arguments.sizes:
        if (size, "model") in results and (size, "nn") in results:
            delta = results[(size, "model")][1] - results[(size, "nn")][1]
            flag = "INERT" if delta < 0.005 else "contributes"
            print(f"  n={size:>9}: Δp = {delta:+.4f}  -> {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
