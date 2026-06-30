"""Dataset utilities for LNHM Phase 0.

- ``read_instances`` / ``load_instances``: stream/collect instances from a
  gzipped JSONL level file (no torch required).
- ``load_level_arrays``: load one level into packed numpy arrays (every instance
  in a level shares the same n, so coordinates pack into a single array).
- ``CurriculumDataPool``: holds train/val arrays per level and samples mixed-n
  batches weighted by the curriculum schedule.
- ``collate_padded``: pad a list of mixed-n instances into a batch with the
  node-padding and per-step validity masks the model/training loop need.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import torch
except ImportError:  # readers below still work without torch
    torch = None


def read_instances(path: str) -> Iterator[Dict]:
    """Stream instances one at a time from a gzipped JSONL level file."""
    with gzip.open(path, "rt", encoding="utf-8") as instance_file:
        for line in instance_file:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_instances(path: str) -> List[Dict]:
    """Read all instances from a gzipped JSONL level file into memory."""
    return list(read_instances(path))


def load_level_arrays(
    path: str, limit: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one level into packed arrays.

    Returns (coordinates, optimal_tours, optimal_distances) with shapes
    (count, n, 2) float32, (count, n) int64, (count,) float32.
    """
    coordinates_list: List[List[List[float]]] = []
    tours_list: List[List[int]] = []
    distances_list: List[float] = []
    for instance in read_instances(path):
        coordinates_list.append(instance["coords"])
        tours_list.append(instance["optimal_tour"])
        distances_list.append(instance["optimal_distance"])
        if limit is not None and len(distances_list) >= limit:
            break
    return (
        np.asarray(coordinates_list, dtype=np.float32),
        np.asarray(tours_list, dtype=np.int64),
        np.asarray(distances_list, dtype=np.float32),
    )


class LevelArrays:
    """Packed arrays for a single level (uniform n)."""

    def __init__(self, level: int, coordinates: np.ndarray, tours: np.ndarray, distances: np.ndarray):
        self.level = level
        self.coordinates = coordinates
        self.tours = tours
        self.distances = distances

    def __len__(self) -> int:
        return self.distances.shape[0]


def collate_padded(coordinates_list: List[np.ndarray], tours_list: List[np.ndarray]):
    """Pad a list of mixed-n instances into a single batch.

    Returns (coordinates, target_tours, node_padding_mask, step_valid_mask):
      coordinates      : (B, max_n, 2) float32
      target_tours     : (B, max_n) int64, padded with city 0
      node_padding_mask: (B, max_n) bool, True where a node is padding
      step_valid_mask  : (B, max_n - 1) bool, True where decode step is real
    """
    if torch is None:
        raise RuntimeError("torch is required for collate_padded")
    batch_size = len(coordinates_list)
    instance_lengths = [coordinates.shape[0] for coordinates in coordinates_list]
    max_num_nodes = max(instance_lengths)

    coordinates = torch.zeros(batch_size, max_num_nodes, 2, dtype=torch.float32)
    target_tours = torch.zeros(batch_size, max_num_nodes, dtype=torch.long)
    node_padding_mask = torch.ones(batch_size, max_num_nodes, dtype=torch.bool)
    for row, (instance_coordinates, instance_tour, length) in enumerate(
        zip(coordinates_list, tours_list, instance_lengths)
    ):
        coordinates[row, :length] = torch.from_numpy(np.ascontiguousarray(instance_coordinates))
        target_tours[row, :length] = torch.from_numpy(np.ascontiguousarray(instance_tour))
        node_padding_mask[row, :length] = False

    lengths_tensor = torch.tensor(instance_lengths, dtype=torch.long)
    decode_step_indices = torch.arange(1, max_num_nodes)  # predicting positions 1..max_n-1
    step_valid_mask = decode_step_indices.unsqueeze(0) < lengths_tensor.unsqueeze(1)
    return coordinates, target_tours, node_padding_mask, step_valid_mask


class CurriculumDataPool:
    """Per-level train/val arrays plus weighted mixed-batch sampling."""

    def __init__(self, data_dir: str, levels: List[int], train_limit=None, val_limit=None):
        self.train: Dict[int, LevelArrays] = {}
        self.val: Dict[int, LevelArrays] = {}
        for level in levels:
            train_path = os.path.join(data_dir, f"level_{level:02d}_train.jsonl.gz")
            val_path = os.path.join(data_dir, f"level_{level:02d}_val.jsonl.gz")
            self.train[level] = LevelArrays(level, *load_level_arrays(train_path, train_limit))
            self.val[level] = LevelArrays(level, *load_level_arrays(val_path, val_limit))

    def sample_mixed_batch(self, level_weights: Dict[int, float], batch_size: int, rng: np.random.Generator):
        """Sample a padded batch with per-level counts drawn from the weights."""
        levels = [level for level, weight in level_weights.items() if weight > 0]
        weights = np.asarray([level_weights[level] for level in levels], dtype=np.float64)
        weights = weights / weights.sum()
        per_level_counts = rng.multinomial(batch_size, weights)

        chosen_coordinates: List[np.ndarray] = []
        chosen_tours: List[np.ndarray] = []
        for level, count in zip(levels, per_level_counts):
            if count == 0:
                continue
            pool = self.train[level]
            row_indices = rng.integers(0, len(pool), size=int(count))
            for row in row_indices:
                chosen_coordinates.append(pool.coordinates[row])
                chosen_tours.append(pool.tours[row])
        return collate_padded(chosen_coordinates, chosen_tours)
