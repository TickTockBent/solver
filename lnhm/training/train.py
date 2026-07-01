"""Training loop for LNHM Phase 0.

Supervised, teacher-forced cross-entropy against canonicalized optimal tours,
driven by the additive curriculum. Each "epoch" is a fixed number of optimizer
steps over curriculum-mixed batches; after every epoch all active levels are
evaluated on validation and the metrics are appended to a CSV (the cross-level
chart is built from this). The frontier advances when it graduates.

Gradient clipping is on by default: without it (and at lr>1e-4) the model
overshoots on the first step and collapses onto the uniform-policy plateau.

Run:
    python training/train.py --config configs/phase0.yaml --data-dir data/phase0
Smoke test (small, fast, CPU):
    python training/train.py --data-dir data/phase0 --levels 3 4 5 6 \
        --train-limit 500 --val-limit 200 --steps-per-epoch 30 --max-epochs-per-level 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch
import yaml

# Put the project root (parent of training/) on the path for cross-package imports.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.dataset import CurriculumDataPool  # noqa: E402
from model.lnhm import LnhmModel  # noqa: E402
from training.curriculum import AdditiveCurriculum  # noqa: E402
from training.evaluate import evaluate_levels  # noqa: E402


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_teacher_forcing_loss(per_step_log_probs: torch.Tensor, step_valid_mask: torch.Tensor) -> torch.Tensor:
    """Negative mean log-prob over the valid (non-padded) decode steps."""
    step_valid = step_valid_mask.to(per_step_log_probs.dtype)
    total_log_prob = (per_step_log_probs * step_valid).sum()
    return -total_log_prob / step_valid.sum().clamp(min=1.0)


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LNHM Phase 0 model.")
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/phase0.yaml"))
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data/phase0"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "runs/phase0"))
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", type=int, nargs="+", default=None, help="Override curriculum levels.")
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--max-epochs-per-level", type=int, default=None, help="Default: config epochs_per_level.")
    parser.add_argument("--batch-size", type=int, default=None, help="Default: config training.batch_size.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm; 0 disables.")
    parser.add_argument("--train-limit", type=int, default=None, help="Cap instances/level (smoke tests).")
    parser.add_argument("--val-limit", type=int, default=None, help="Cap val instances/level (smoke tests).")
    # Range/sweep support: name each run and override model size from the CLI.
    parser.add_argument("--run-name", default=None, help="Output subdir name (auto if omitted).")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-encoder-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--ff-dim", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    arguments = parse_arguments(argv)
    with open(arguments.config) as config_file:
        config = yaml.safe_load(config_file)

    levels = arguments.levels if arguments.levels is not None else config["data"]["levels"]
    warmup_levels = config["data"].get("warmup_levels", [])
    batch_size = arguments.batch_size or config["training"]["batch_size"]
    max_epochs_per_level = arguments.max_epochs_per_level or config["training"]["epochs_per_level"]
    graduation_threshold = config["training"]["graduation_threshold"]
    accuracy_tolerance = config["training"]["accuracy_tolerance"]
    learning_rate = config["training"]["lr"]
    frontier_weight = config["curriculum"]["frontier_weight"]

    # Model config with optional CLI size overrides (for capacity / level sweeps).
    model_config = dict(config["model"])
    for key, value in (("d_model", arguments.d_model), ("n_encoder_layers", arguments.n_encoder_layers),
                       ("n_heads", arguments.n_heads), ("ff_dim", arguments.ff_dim)):
        if value is not None:
            model_config[key] = value

    # Named output subdir so a range of runs (different levels/sizes/seeds) never clobber.
    run_name = arguments.run_name or f"L{min(levels)}-{max(levels)}_d{model_config['d_model']}_s{arguments.seed}"
    run_output_dir = os.path.join(arguments.output_dir, run_name)

    torch.manual_seed(arguments.seed)
    sampling_rng = np.random.default_rng(arguments.seed)
    device = resolve_device(arguments.device)
    os.makedirs(run_output_dir, exist_ok=True)

    print(f"Loading data for levels {levels} from {arguments.data_dir} ...")
    data_pool = CurriculumDataPool(
        arguments.data_dir, levels, train_limit=arguments.train_limit, val_limit=arguments.val_limit
    )
    curriculum = AdditiveCurriculum(levels, warmup_levels, frontier_weight=frontier_weight)

    model = LnhmModel.from_config(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=arguments.steps_per_epoch, T_mult=1
    )
    print(f"Model: {model.num_parameters()} params on {device}")

    print(f"Run: {run_name}  ->  {run_output_dir}")
    metrics_path = os.path.join(run_output_dir, "metrics.csv")
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["global_epoch", "frontier_level", "level", "accuracy", "mean_gap", "worst_gap"])

    global_epoch = 0
    run_start = time.monotonic()
    while True:
        frontier_level = curriculum.frontier_level
        level_weights = curriculum.level_weights()
        weight_summary = ", ".join(f"{lvl}:{wgt:.2f}" for lvl, wgt in sorted(level_weights.items()))
        print(f"\n=== Frontier n={frontier_level}  mix[{weight_summary}] ===")

        graduated = False
        for _ in range(max_epochs_per_level):
            model.train()
            epoch_loss_total = 0.0
            for _ in range(arguments.steps_per_epoch):
                coordinates, target_tours, node_padding_mask, step_valid_mask = (
                    tensor.to(device)
                    for tensor in data_pool.sample_mixed_batch(level_weights, batch_size, sampling_rng)
                )
                optimizer.zero_grad()
                per_step_log_probs = model(coordinates, target_tours, node_padding_mask)
                loss = masked_teacher_forcing_loss(per_step_log_probs, step_valid_mask)
                loss.backward()
                if arguments.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), arguments.grad_clip)
                optimizer.step()
                scheduler.step()
                epoch_loss_total += loss.item()

            global_epoch += 1
            metrics_by_level = evaluate_levels(
                model, data_pool.val, curriculum.active_levels, device, accuracy_tolerance
            )
            for level in curriculum.active_levels:
                level_metrics = metrics_by_level[level]
                metrics_writer.writerow([
                    global_epoch, frontier_level, level,
                    f"{level_metrics['accuracy']:.4f}",
                    f"{level_metrics['mean_gap']:.4f}",
                    f"{level_metrics['worst_gap']:.4f}",
                ])
            metrics_file.flush()

            mean_epoch_loss = epoch_loss_total / arguments.steps_per_epoch
            accuracy_summary = " ".join(
                f"n{lvl}={metrics_by_level[lvl]['accuracy']:.2f}" for lvl in curriculum.active_levels
            )
            print(f"  epoch {global_epoch:>4} loss={mean_epoch_loss:.4f}  acc[{accuracy_summary}]")

            frontier_accuracy = metrics_by_level[frontier_level]["accuracy"]
            if curriculum.should_graduate(frontier_accuracy, graduation_threshold):
                graduated = True
                break

        status = "graduated" if graduated else f"reached max epochs ({max_epochs_per_level})"
        print(f"  n={frontier_level} {status}")

        if curriculum.is_complete:
            break
        curriculum.advance()

    metrics_file.close()
    checkpoint_path = os.path.join(run_output_dir, "model_final.pt")
    torch.save({"state_dict": model.state_dict(), "model_config": model_config,
                "levels": levels, "run_name": run_name}, checkpoint_path)
    elapsed = time.monotonic() - run_start

    # Compute-tracking summary: convergence is data-dependent, so record exactly how
    # much training each run took (steps, instances seen, wall). Lets the 2x2
    # overtraining experiment separate quality-compounding from compute-compounding.
    total_steps = global_epoch * arguments.steps_per_epoch
    summary = {
        "run_name": run_name,
        "levels": levels,
        "seed": arguments.seed,
        "train_limit": arguments.train_limit,
        "batch_size": batch_size,
        "steps_per_epoch": arguments.steps_per_epoch,
        "total_epochs": global_epoch,
        "total_steps": total_steps,
        "instances_seen": total_steps * batch_size,
        "wall_seconds": round(elapsed, 1),
    }
    with open(os.path.join(run_output_dir, "train_summary.json"), "w") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(f"\nDone in {elapsed:.1f}s ({total_steps} steps, {summary['instances_seen']} instances). "
          f"Metrics -> {metrics_path}  Checkpoint -> {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
