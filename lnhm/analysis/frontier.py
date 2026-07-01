"""Phase 1 cost/quality frontier runner for LNHM composition.

Evaluates baselines and composition pipelines against an LKH near-optimal
reference, measuring optimality gap AND wall-cost per instance, then reports the
Pareto-undominated points. This is the Phase 1 experiment (see phase1/phase1-spec.md).

Pipelines:
  - baselines: nearest_neighbor, NN+2opt, space_filling_curve (Hilbert), SFC+2opt
  - composition: {model, held_karp} local solver x {none, seam_2opt, full_2opt}
    cleanup, at cluster cap k. (Hilbert is both a baseline and the partition/stitch
    ordering inside composition.)
  - LKH itself is shown as the near-optimal / high-cost anchor.

Example:
  python analysis/frontier.py --lkh-binary /home/ticktockbent/tools/LKH-3.0.13/LKH \
      --checkpoint runs/full/model_final.pt --sizes 50 100 200 --k-caps 10 --instances 5
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from data.held_karp import held_karp, tour_distance  # noqa: E402
from analysis.baselines import (lkh_tour, nearest_neighbor, neighbor_two_opt,  # noqa: E402
                                space_filling_curve, two_opt)
from analysis.compose import compose_solve, held_karp_solver, make_model_batch_solver  # noqa: E402

Pipeline = Tuple[str, Callable[[np.ndarray], List[int]]]


def build_pipelines(k_caps: List[int], model_batch_solver, scale: bool = False) -> List[Pipeline]:
    """The pipelines to race. Each maps coordinates -> a tour (cycle).

    scale=True drops the O(n^2) pipelines (full 2-opt, Held-Karp local solves) and
    keeps only the near-linear ones, for large-n runs. Model composition uses the
    batched local solver (one padded forward pass per level)."""
    pipelines: List[Pipeline] = [
        ("space_filling", lambda c: space_filling_curve(c)),
        ("SFC+neighbor2opt", lambda c: neighbor_two_opt(c, space_filling_curve(c))),
    ]
    if not scale:  # these are O(n^2) (nearest_neighbor, full 2-opt) -- not for huge n
        pipelines += [
            ("nearest_neighbor", lambda c: nearest_neighbor(c)),
            ("NN+2opt", lambda c: two_opt(c, nearest_neighbor(c))),
            ("SFC+2opt", lambda c: two_opt(c, space_filling_curve(c))),
            ("NN+neighbor2opt", lambda c: neighbor_two_opt(c, nearest_neighbor(c))),
        ]
    cleanups = ("none", "neighbor_2opt") if scale else ("none", "seam_2opt", "full_2opt", "neighbor_2opt")
    for k in k_caps:
        for cleanup in cleanups:
            pipelines.append((f"compose:model:k{k}:{cleanup}", lambda c, k=k, cl=cleanup:
                compose_solve(c, k_cap=k, batch_local_solver=model_batch_solver, cleanup=cl)))
        if not scale:
            for cleanup in cleanups:
                pipelines.append((f"compose:hk:k{k}:{cleanup}", lambda c, k=k, cl=cleanup:
                    compose_solve(c, k_cap=k, local_solver=held_karp_solver, cleanup=cl)))
    return pipelines


def pareto_undominated(rows: List[Dict]) -> None:
    """Mark rows that are not dominated (lower gap AND lower cost) by another."""
    for row in rows:
        dominated = any(
            other is not row
            and other["mean_gap_pct"] <= row["mean_gap_pct"]
            and other["mean_wall_ms"] <= row["mean_wall_ms"]
            and (other["mean_gap_pct"] < row["mean_gap_pct"]
                 or other["mean_wall_ms"] < row["mean_wall_ms"])
            for other in rows
        )
        row["pareto"] = "" if dominated else "*"


def run(arguments) -> List[Dict]:
    rng = np.random.default_rng(arguments.seed)
    model_batch_solver = make_model_batch_solver(arguments.checkpoint, device=arguments.device)
    pipelines = build_pipelines(arguments.k_caps, model_batch_solver, scale=arguments.scale)

    all_rows: List[Dict] = []
    for size in arguments.sizes:
        print(f"\n### n={size} ({arguments.instances} instances, ref={arguments.reference}) ###", flush=True)
        instances = [rng.random((size, 2)) for _ in range(arguments.instances)]

        if arguments.reference == "bhh":
            # Beardwood-Halton-Hammersley: expected optimal ~ 0.7124*sqrt(n) in the unit square.
            bhh_distance = 0.7124 * (size ** 0.5)
            reference_distances = [bhh_distance] * arguments.instances
            reference_label, reference_wall = "BHH (0.7124*sqrt(n))", 0.0
        else:
            reference_distances, reference_walls = [], []
            for instance in instances:
                start = time.perf_counter()
                reference_tour = lkh_tour(instance, lkh_binary=arguments.lkh_binary)
                reference_walls.append((time.perf_counter() - start) * 1000)
                reference_distances.append(tour_distance(instance, reference_tour))
            reference_label, reference_wall = "LKH (reference)", statistics.mean(reference_walls)

        size_rows: List[Dict] = [{
            "pipeline": reference_label, "n": size,
            "mean_gap_pct": 0.0, "mean_wall_ms": reference_wall,
        }]

        for name, solve in pipelines:
            gaps, walls = [], []
            for instance, reference in zip(instances, reference_distances):
                start = time.perf_counter()
                tour = solve(instance)
                walls.append((time.perf_counter() - start) * 1000)
                assert sorted(tour) == list(range(size)), f"{name} produced an invalid tour"
                gaps.append((tour_distance(instance, tour) / reference - 1) * 100)
            size_rows.append({
                "pipeline": name, "n": size,
                "mean_gap_pct": statistics.mean(gaps), "mean_wall_ms": statistics.mean(walls),
            })

        pareto_undominated(size_rows)
        for row in sorted(size_rows, key=lambda r: r["mean_gap_pct"]):
            print(f"  {row.get('pareto',''):1} {row['pipeline']:26} "
                  f"gap={row['mean_gap_pct']:6.2f}%  cost={row['mean_wall_ms']:8.1f} ms", flush=True)
        all_rows.extend(size_rows)
    return all_rows


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LNHM Phase 1 cost/quality frontier.")
    parser.add_argument("--lkh-binary", default="LKH")
    parser.add_argument("--checkpoint", default=os.path.join(PROJECT_ROOT, "runs/full/model_final.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--k-caps", type=int, nargs="+", default=[10])
    parser.add_argument("--instances", type=int, default=5)
    parser.add_argument("--reference", choices=["lkh", "bhh"], default="lkh",
                        help="lkh = near-optimal per instance; bhh = 0.7124*sqrt(n) (for huge n).")
    parser.add_argument("--scale", action="store_true", help="Drop O(n^2) pipelines; keep near-linear only.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "runs/phase1"))
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    arguments = parse_arguments(argv)
    os.makedirs(arguments.output_dir, exist_ok=True)
    rows = run(arguments)
    results_path = os.path.join(arguments.output_dir, "frontier_results.csv")
    with open(results_path, "w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=["pipeline", "n", "mean_gap_pct", "mean_wall_ms", "pareto"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})
    print(f"\nResults -> {results_path}  (* = Pareto-undominated within its n)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
