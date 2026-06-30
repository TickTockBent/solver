"""Cross-level reinforcement experiment for LNHM Phase 0.

Tests whether training on the full curriculum improves anchor-level accuracy
beyond single-level training at equal compute. All models train for the SAME
number of gradient steps (compute-matched), start from the same init per seed,
and are evaluated on the same held-out anchor validation set:

  - Model A (treatment): additive curriculum over levels [3, N], N > anchor.
  - Model B (overfitting control): anchor only, SMALL fixed pool (overfits).
  - Model C (regularization control): anchor only, LARGE fresh pool (no overfit).
  - Model D (curriculum-depth control): curriculum [3, anchor] -- climbs up to the
    anchor but trains on nothing larger.

Two comparisons:
  - A vs C: "does the full curriculum beat single-level training?" (spec metric).
  - A vs D: "do levels LARGER than the anchor specifically help it?" -- the
    backward-transfer / scale-unification claim, the actual thesis test.

Design notes (learned from spot-check runs):
  - Effect is only visible where the single-level control does NOT saturate, i.e.
    HARD anchors near the capacity wall (n=5 is a negative control).
  - Budget must be large enough for A to traverse the curriculum and let the
    anchor break through.
  - Constant LR (no decay schedule): a decaying schedule trains each level at a
    different LR depending on WHEN it appears, which confounds A vs D (D trains
    its anchor at the schedule tail). Constant LR removes that variable.

The default matrix is a GPU/offload job. For a local spot-check pass a subset.
See phase0/phase0-spec.md, "Cross-Level Reinforcement Test".
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
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import CurriculumDataPool, LevelArrays, collate_padded  # noqa: E402
from data.held_karp import canonicalize_tour, held_karp  # noqa: E402
from model.lnhm import LnhmModel  # noqa: E402
from training.curriculum import AdditiveCurriculum  # noqa: E402
from training.evaluate import evaluate_level  # noqa: E402
from training.train import masked_teacher_forcing_loss, resolve_device  # noqa: E402

COORDINATE_DECIMALS = 6
WARMUP_LEVELS = [3, 4]
RESULT_FIELDS = ["anchor", "range_N", "seed", "acc_A", "acc_B", "acc_C", "acc_D",
                 "A_minus_C", "A_minus_D", "A_minus_B"]


# --------------------------------------------------------------------------- #
# Batch sources                                                               #
# --------------------------------------------------------------------------- #
def generate_anchor_pool(anchor_n: int, count: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate `count` solved anchor-size instances (coords, canonical tours)."""
    coordinates = np.empty((count, anchor_n, 2), dtype=np.float32)
    tours = np.empty((count, anchor_n), dtype=np.int64)
    for index in range(count):
        instance_coordinates = np.round(rng.random((anchor_n, 2)), COORDINATE_DECIMALS)
        optimal_tour, _ = held_karp(instance_coordinates)
        coordinates[index] = instance_coordinates
        tours[index] = canonicalize_tour(optimal_tour)
    return coordinates, tours


def make_fixed_pool_source(coordinates: np.ndarray, tours: np.ndarray, batch_size: int) -> Callable:
    """Batch source that samples (with replacement) from a fixed in-memory pool."""
    pool_size = coordinates.shape[0]

    def sample(_step: int, rng: np.random.Generator):
        rows = rng.integers(0, pool_size, size=batch_size)
        return collate_padded([coordinates[r] for r in rows], [tours[r] for r in rows])

    return sample


class CurriculumSource:
    """Additive curriculum with a fixed step budget per stage, so the frontier
    advances deterministically from level 3 up to the top level over total_steps."""

    def __init__(self, data_pool: CurriculumDataPool, levels: List[int], total_steps: int,
                 batch_size: int, frontier_weight: float):
        self.data_pool = data_pool
        self.levels = sorted(levels)
        self.batch_size = batch_size
        self.curriculum = AdditiveCurriculum(self.levels, WARMUP_LEVELS, frontier_weight=frontier_weight)
        self.num_stages = len(self.levels)
        self.steps_per_stage = max(1, total_steps // self.num_stages)

    def sample(self, step: int, rng: np.random.Generator):
        stage = min(step // self.steps_per_stage, self.num_stages - 1)
        self.curriculum.frontier_index = stage
        weights = self.curriculum.level_weights()
        return self.data_pool.sample_mixed_batch(weights, self.batch_size, rng)


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def train_for_steps(model, sample_fn: Callable, total_steps: int, device, lr: float,
                    sampling_rng: np.random.Generator, grad_clip: float = 1.0):
    # Constant LR (no scheduler) -- see module docstring: a decay schedule would
    # train each curriculum level at a different LR by position, confounding A vs D.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(total_steps):
        coordinates, target_tours, node_padding_mask, step_valid_mask = (
            tensor.to(device) for tensor in sample_fn(step, sampling_rng)
        )
        optimizer.zero_grad()
        per_step_log_probs = model(coordinates, target_tours, node_padding_mask)
        loss = masked_teacher_forcing_loss(per_step_log_probs, step_valid_mask)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


def train_one_model(model_config: dict, sample_fn: Callable, total_steps: int, device,
                    lr: float, seed: int) -> LnhmModel:
    """Build a model from a fixed init (per seed) and train it for total_steps."""
    torch.manual_seed(seed)  # identical init across A/B/C/D for this seed
    model = LnhmModel.from_config(model_config).to(device)
    sampling_rng = np.random.default_rng(10_000 + seed)
    train_for_steps(model, sample_fn, total_steps, device, lr, sampling_rng)
    return model


# --------------------------------------------------------------------------- #
# Experiment driver                                                            #
# --------------------------------------------------------------------------- #
def run_experiment(arguments, config, results_path: str) -> List[Dict]:
    device = resolve_device(arguments.device)
    model_config = config["model"]
    lr = config["training"]["lr"]
    tolerance = config["training"]["accuracy_tolerance"]
    batch_size = arguments.batch_size or config["training"]["batch_size"]
    frontier_weight = config["curriculum"]["frontier_weight"]
    total_steps = arguments.total_steps

    max_range_n = max(arguments.ranges)
    all_levels = list(range(3, max_range_n + 1))
    print(f"Loading curriculum data (levels {all_levels}) ... device={device}", flush=True)
    data_pool = CurriculumDataPool(arguments.data_dir, all_levels,
                                   train_limit=arguments.train_limit, val_limit=arguments.val_limit)

    results: List[Dict] = []
    # Incremental, flushed CSV: completed rows survive even if the run is killed.
    with open(results_path, "w", newline="") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        results_file.flush()

        for anchor in arguments.anchors:
            print(f"\n# anchor n={anchor}: generating control pools "
                  f"(B={arguments.b_pool}, C={arguments.c_pool}) ...", flush=True)
            pool_rng = np.random.default_rng(777 + anchor)
            b_coords, b_tours = generate_anchor_pool(anchor, arguments.b_pool, pool_rng)
            c_coords, c_tours = generate_anchor_pool(anchor, arguments.c_pool, pool_rng)
            b_source = make_fixed_pool_source(b_coords, b_tours, batch_size)
            c_source = make_fixed_pool_source(c_coords, c_tours, batch_size)
            anchor_val: LevelArrays = data_pool.val[anchor]
            # B, C, D depend only on (anchor, seed), NOT on range_n -> compute once per
            # seed and reuse across ranges. Only A varies with the curriculum range.
            b_accuracy_cache: Dict[int, float] = {}
            c_accuracy_cache: Dict[int, float] = {}
            d_accuracy_cache: Dict[int, float] = {}

            for range_n in arguments.ranges:
                if range_n <= anchor:
                    print(f"  skip range N={range_n} <= anchor {anchor}", flush=True)
                    continue
                for seed in arguments.seeds:
                    started = time.monotonic()
                    accuracies: Dict[str, float] = {}

                    # Model A: full curriculum [3, range_n] -- depends on range, trained every time.
                    curriculum_source = CurriculumSource(
                        data_pool, list(range(3, range_n + 1)), total_steps, batch_size, frontier_weight)
                    a_model = train_one_model(model_config, curriculum_source.sample, total_steps, device, lr, seed)
                    accuracies["A"] = evaluate_level(a_model, anchor_val, device, tolerance)["accuracy"]

                    # Models B/C/D: independent of range_n -> compute once per (anchor, seed).
                    if seed not in d_accuracy_cache:
                        b_model = train_one_model(model_config, b_source, total_steps, device, lr, seed)
                        b_accuracy_cache[seed] = evaluate_level(b_model, anchor_val, device, tolerance)["accuracy"]
                        c_model = train_one_model(model_config, c_source, total_steps, device, lr, seed)
                        c_accuracy_cache[seed] = evaluate_level(c_model, anchor_val, device, tolerance)["accuracy"]
                        d_source = CurriculumSource(
                            data_pool, list(range(3, anchor + 1)), total_steps, batch_size, frontier_weight)
                        d_model = train_one_model(model_config, d_source.sample, total_steps, device, lr, seed)
                        d_accuracy_cache[seed] = evaluate_level(d_model, anchor_val, device, tolerance)["accuracy"]
                    accuracies["B"] = b_accuracy_cache[seed]
                    accuracies["C"] = c_accuracy_cache[seed]
                    accuracies["D"] = d_accuracy_cache[seed]

                    row = {
                        "anchor": anchor, "range_N": range_n, "seed": seed,
                        "acc_A": accuracies["A"], "acc_B": accuracies["B"],
                        "acc_C": accuracies["C"], "acc_D": accuracies["D"],
                        "A_minus_C": accuracies["A"] - accuracies["C"],
                        "A_minus_D": accuracies["A"] - accuracies["D"],
                        "A_minus_B": accuracies["A"] - accuracies["B"],
                    }
                    results.append(row)
                    writer.writerow(row)
                    results_file.flush()  # persist this row immediately
                    print(f"  anchor={anchor} N={range_n} seed={seed}: "
                          f"A={row['acc_A']:.3f} B={row['acc_B']:.3f} C={row['acc_C']:.3f} D={row['acc_D']:.3f} "
                          f"| A-C={row['A_minus_C']:+.3f} A-D={row['A_minus_D']:+.3f} "
                          f"({time.monotonic()-started:.0f}s)", flush=True)
    return results


def summarize(results: List[Dict]) -> None:
    print("\n=== Summary: mean over seeds (PASS = mean diff >= +0.05) ===", flush=True)
    print("  A vs C = full curriculum beats single-level (spec metric)")
    print("  A vs D = levels LARGER than anchor help it (thesis test); D = curriculum [3, anchor]\n")
    configs = sorted({(r["anchor"], r["range_N"]) for r in results})
    print(f"{'anchor':>6} {'N':>3} {'acc_A':>7} {'acc_C':>7} {'acc_D':>7} "
          f"{'A-C':>7} {'A-D':>7} {'A>C':>5} {'A>D':>5}")
    for anchor, range_n in configs:
        subset = [r for r in results if r["anchor"] == anchor and r["range_N"] == range_n]
        mean = lambda key: statistics.mean(r[key] for r in subset)
        a_c, a_d = mean("A_minus_C"), mean("A_minus_D")
        print(f"{anchor:>6} {range_n:>3} {mean('acc_A'):>7.3f} {mean('acc_C'):>7.3f} {mean('acc_D'):>7.3f} "
              f"{a_c:>+7.3f} {a_d:>+7.3f} {('PASS' if a_c >= 0.05 else 'fail'):>5} "
              f"{('PASS' if a_d >= 0.05 else 'fail'):>5}")


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LNHM Phase 0 cross-level reinforcement experiment.")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/phase0.yaml"))
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data/phase0"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "runs/xlevel"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--anchors", type=int, nargs="+", default=[5, 9, 11])
    parser.add_argument("--ranges", type=int, nargs="+", default=[7, 10, 12], help="N in curriculum [3, N].")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--total-steps", type=int, default=2500, help="Compute budget (gradient steps) per model.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--b-pool", type=int, default=2000, help="Model B fixed-pool size (small => overfits).")
    parser.add_argument("--c-pool", type=int, default=50000, help="Model C fresh-pool size (large => no overfit).")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    arguments = parse_arguments(argv)
    with open(arguments.config) as config_file:
        config = yaml.safe_load(config_file)
    os.makedirs(arguments.output_dir, exist_ok=True)
    results_path = os.path.join(arguments.output_dir, "cross_level_results.csv")

    results = run_experiment(arguments, config, results_path)
    summarize(results)
    print(f"\nResults -> {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
