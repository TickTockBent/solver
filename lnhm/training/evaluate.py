"""Validation and metrics for LNHM Phase 0.

Per level: accuracy (fraction of instances whose greedily-decoded tour is within
``tolerance`` of the optimum), mean optimality gap, and worst-case gap.
Evaluation runs per level, so every batch has a uniform n and the decoder's
greedy rollout never sees padding.

See phase0/phase0-spec.md, "Evaluation".
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from data.dataset import LevelArrays


def tour_distance_torch(coordinates: torch.Tensor, tours: torch.Tensor) -> torch.Tensor:
    """Closed-tour length for a batch. coords (B, n, 2), tours (B, n) -> (B,)."""
    batch_indices = torch.arange(coordinates.shape[0], device=coordinates.device).unsqueeze(1)
    ordered_points = coordinates[batch_indices, tours]  # (B, n, 2)
    next_points = torch.roll(ordered_points, shifts=-1, dims=1)
    leg_lengths = torch.linalg.norm(ordered_points - next_points, dim=-1)
    return leg_lengths.sum(dim=1)


@torch.no_grad()
def evaluate_level(
    model,
    level_arrays: LevelArrays,
    device: torch.device,
    tolerance: float = 0.01,
    batch_size: int = 1024,
) -> Dict[str, float]:
    """Greedy-decode a level's validation set and return accuracy / gap metrics."""
    model.eval()
    optimality_gaps = []
    num_instances = len(level_arrays)
    for start in range(0, num_instances, batch_size):
        end = min(start + batch_size, num_instances)
        coordinates = torch.from_numpy(level_arrays.coordinates[start:end]).to(device)
        optimal_distances = torch.from_numpy(level_arrays.distances[start:end]).to(device)

        predicted_tours, _ = model.solve(coordinates, mode="greedy")
        predicted_distances = tour_distance_torch(coordinates, predicted_tours)
        # Guard against a zero optimum (only the degenerate single-point case).
        gap = (predicted_distances - optimal_distances) / optimal_distances.clamp(min=1e-9)
        optimality_gaps.append(gap.cpu().numpy())

    all_gaps = np.concatenate(optimality_gaps)
    return {
        "accuracy": float((all_gaps <= tolerance).mean()),
        "mean_gap": float(all_gaps.mean()),
        "worst_gap": float(all_gaps.max()),
    }


@torch.no_grad()
def evaluate_levels(
    model,
    val_pools: Dict[int, LevelArrays],
    levels,
    device: torch.device,
    tolerance: float = 0.01,
) -> Dict[int, Dict[str, float]]:
    """Evaluate several levels; returns {level: {accuracy, mean_gap, worst_gap}}."""
    return {level: evaluate_level(model, val_pools[level], device, tolerance) for level in levels}
