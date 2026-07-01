"""Composition for LNHM Phase 1: decompose -> local-solve -> stitch -> cleanup.

Partition points into clusters of size <= k_cap, solve each cluster locally,
stitch the sub-tours into one global cycle via greedy port-joining, then
optionally clean up the boundary seams with 2-opt.

The local solver is pluggable: Held-Karp gives exact small-cluster solves (used
to isolate stitching quality from model quality), and the LNHM model slots in for
the full experiment. Single-level for now; recursive cluster-ordering (solving the
inter-cluster problem with the same machinery) is a marked extension.

See phase1/phase1-spec.md.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Sequence, Set, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from data.held_karp import held_karp, tour_distance  # noqa: E402
from analysis.baselines import nearest_neighbor, neighbor_two_opt, space_filling_curve, two_opt  # noqa: E402

# A local solver maps cluster coordinates (m, 2) -> a visiting order of 0..m-1.
LocalSolver = Callable[[np.ndarray], List[int]]


def held_karp_solver(cluster_coordinates: np.ndarray) -> List[int]:
    """Exact local solver (for clusters within Held-Karp's reach)."""
    tour, _ = held_karp(cluster_coordinates)
    return tour


def make_model_solver(checkpoint_path: str, model_config: dict = None, device: str = "cpu") -> LocalSolver:
    """Build a LocalSolver backed by a trained LNHM model (greedy decode).

    Each cluster is normalized into [0,1]^2 by a UNIFORM scale + translate before
    the model sees it: the model was trained on [0,1]^2 instances, and uniform
    similarity transforms preserve the optimal tour order (anisotropic stretch
    would NOT -- it distorts distances and can change the tour).
    """
    import torch  # lazy: keeps the Held-Karp path torch-free
    import yaml
    from model.lnhm import LnhmModel

    torch_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # Self-describing checkpoint (train.py): config travels with the weights.
        state_dict = checkpoint["state_dict"]
        if model_config is None:
            model_config = checkpoint.get("model_config")
    else:
        state_dict = checkpoint  # legacy: a bare state_dict
    if model_config is None:
        with open(os.path.join(PROJECT_ROOT, "configs/phase0.yaml")) as config_file:
            model_config = yaml.safe_load(config_file)["model"]

    model = LnhmModel.from_config(model_config)
    model.load_state_dict(state_dict)
    model.to(torch_device).eval()

    def solve(cluster_coordinates: np.ndarray) -> List[int]:
        points = np.asarray(cluster_coordinates, dtype=np.float64)
        num_points = len(points)
        if num_points <= 3:
            return list(range(num_points))  # any order is optimal for n<=3
        lower = points.min(axis=0)
        span = float((points.max(axis=0) - lower).max())
        if span < 1e-12:
            return list(range(num_points))
        normalized = (points - lower) / span  # uniform scale + translate
        batch = torch.tensor(normalized, dtype=torch.float32, device=torch_device).unsqueeze(0)
        with torch.no_grad():
            tour, _ = model.solve(batch, mode="greedy")
        return tour[0].tolist()

    return solve


def make_model_batch_solver(checkpoint_path: str, model_config: dict = None,
                            device: str = "cpu", chunk: int = 8192):
    """Like make_model_solver, but solves MANY clusters per call in one padded
    forward pass. Clusters within a composition level are independent, so this is
    the 'batch dimension' -- far cheaper than a Python loop of single solves."""
    import torch
    import yaml
    from model.lnhm import LnhmModel

    torch_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if model_config is None:
            model_config = checkpoint.get("model_config")
    else:
        state_dict = checkpoint
    if model_config is None:
        with open(os.path.join(PROJECT_ROOT, "configs/phase0.yaml")) as config_file:
            model_config = yaml.safe_load(config_file)["model"]
    model = LnhmModel.from_config(model_config)
    model.load_state_dict(state_dict)
    model.to(torch_device).eval()

    def batch_solve(clusters: List[np.ndarray]) -> List[List[int]]:
        orders: List[List[int]] = []
        for start in range(0, len(clusters), chunk):
            group = clusters[start:start + chunk]
            lengths = [len(c) for c in group]
            max_n = max(lengths)
            coords = torch.zeros(len(group), max_n, 2, dtype=torch.float32, device=torch_device)
            padding = torch.ones(len(group), max_n, dtype=torch.bool, device=torch_device)
            for row, cluster in enumerate(group):
                points = np.asarray(cluster, dtype=np.float64)
                length = len(points)
                lower = points.min(axis=0)
                span = float((points.max(axis=0) - lower).max())
                if span < 1e-12:
                    span = 1.0
                coords[row, :length] = torch.from_numpy(((points - lower) / span).astype(np.float32))
                padding[row, :length] = False
            with torch.no_grad():
                tours, _ = model.solve(coords, node_padding_mask=padding, mode="greedy")
            tours = tours.cpu().numpy()
            for row, length in enumerate(lengths):
                # First `length` picks are the real nodes (padding is masked out until
                # they're all visited); trivial clusters get the identity order.
                orders.append(list(range(length)) if length <= 3 else tours[row, :length].tolist())
        return orders

    return batch_solve


# --------------------------------------------------------------------------- #
# Partitioners: return a list of clusters, each a list of global point indices  #
# (together a partition of range(n)).                                           #
# --------------------------------------------------------------------------- #
def partition_space_filling(coordinates: np.ndarray, k_cap: int) -> List[List[int]]:
    """Chunk the Hilbert order into contiguous groups of size <= k_cap."""
    order = space_filling_curve(coordinates)
    return [order[i:i + k_cap] for i in range(0, len(order), k_cap)]


def partition_median(coordinates: np.ndarray, k_cap: int) -> List[List[int]]:
    """Recursively split along the longer axis at the median until clusters <= k_cap."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    clusters: List[List[int]] = []

    def split(indices: List[int]) -> None:
        if len(indices) <= k_cap:
            clusters.append(list(indices))
            return
        points = coordinates[indices]
        axis = int(np.argmax(points.max(axis=0) - points.min(axis=0)))
        sorted_order = np.argsort(points[:, axis], kind="stable")
        midpoint = len(indices) // 2
        split([indices[o] for o in sorted_order[:midpoint]])
        split([indices[o] for o in sorted_order[midpoint:]])

    split(list(range(len(coordinates))))
    return clusters


# --------------------------------------------------------------------------- #
# Stitching                                                                    #
# --------------------------------------------------------------------------- #
def _quick_solver(points: np.ndarray) -> List[int]:
    """Fast near-optimal solver for the small groups at the base of the ordering
    recursion (NN + 2-opt; both O(m^2) but m is tiny here)."""
    if len(points) <= 3:
        return list(range(len(points)))
    return two_opt(points, nearest_neighbor(points))


def _cut_options(cycle: List[int]) -> List[Tuple[int, int, List[int]]]:
    """All ways to open a cluster's cyclic sub-tour into a path, as (entry, exit, path).

    Cutting one edge of the cycle yields a Hamiltonian path through the cluster whose
    endpoints were joined by that edge; both traversal directions are offered so the
    stitcher can pick entry/exit freely."""
    length = len(cycle)
    if length == 1:
        return [(cycle[0], cycle[0], [cycle[0]])]
    options: List[Tuple[int, int, List[int]]] = []
    for cut in range(length):
        path = cycle[cut + 1:] + cycle[:cut + 1]
        options.append((path[0], path[-1], path))
        options.append((path[-1], path[0], path[::-1]))
    return options


def stitch_dp(coordinates: np.ndarray, cluster_cycles: List[List[int]],
              cluster_order: List[int]) -> Tuple[List[int], Set[int]]:
    """Stitch cluster sub-tours into one global tour, choosing where to open each
    cluster's cycle to MINIMIZE total seam length, given the cluster order.

    DP over the ordered clusters: each cluster contributes O(2k) open-options
    (entry, exit); the transition cost from cluster i to i+1 is dist(exit_i, entry_{i+1}).
    Returns (global_tour, seam_edge_positions)."""
    ordered_cycles = [cluster_cycles[index] for index in cluster_order]
    options = [_cut_options(cycle) for cycle in ordered_cycles]
    num_clusters = len(options)

    def gap(a: int, b: int) -> float:
        return float(np.linalg.norm(coordinates[a] - coordinates[b]))

    cost = [0.0] * len(options[0])
    backpointer = [[-1] * len(option) for option in options]
    for i in range(1, num_clusters):
        next_cost = [float("inf")] * len(options[i])
        for s2, (entry2, _exit2, _path2) in enumerate(options[i]):
            best, best_previous = float("inf"), -1
            for s1, (_entry1, exit1, _path1) in enumerate(options[i - 1]):
                candidate = cost[s1] + gap(exit1, entry2)
                if candidate < best:
                    best, best_previous = candidate, s1
            next_cost[s2] = best
            backpointer[i][s2] = best_previous
        cost = next_cost

    chosen = [0] * num_clusters
    chosen[-1] = min(range(len(cost)), key=lambda s: cost[s])
    for i in range(num_clusters - 1, 0, -1):
        chosen[i - 1] = backpointer[i][chosen[i]]

    global_tour: List[int] = []
    seam_positions: Set[int] = set()
    for i in range(num_clusters):
        if i > 0:
            seam_positions.add(len(global_tour) - 1)  # first-edge position of this junction
        global_tour.extend(options[i][chosen[i]][2])
    return global_tour, seam_positions


def _expand_positions(positions: Set[int], window: int, num_cities: int) -> Set[int]:
    expanded: Set[int] = set()
    for position in positions:
        for offset in range(-window, window + 1):
            candidate = position + offset
            if 0 <= candidate < num_cities - 1:
                expanded.add(candidate)
    return expanded


# --------------------------------------------------------------------------- #
# Full pipeline                                                                #
# --------------------------------------------------------------------------- #
def compose_solve(
    coordinates: Sequence[Sequence[float]],
    k_cap: int,
    local_solver: LocalSolver = held_karp_solver,
    partitioner: Callable[[np.ndarray, int], List[List[int]]] = partition_space_filling,
    cleanup: str = "none",   # "none" | "seam_2opt" | "full_2opt" | "neighbor_2opt"
    seam_window: int = 2,
    batch_local_solver: Callable[[List[np.ndarray]], List[List[int]]] = None,
) -> List[int]:
    """Solve a large instance by composition. Returns a global tour (valid cycle).

    `batch_local_solver` (optional) solves all clusters of a level in one call
    (the parallel 'batch dimension'); without it, clusters are solved one at a time.
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    num_cities = len(coordinates)
    # Solve a list of clusters at once if a batch solver is given, else one by one.
    solve_clusters = batch_local_solver or (lambda group: [local_solver(c) for c in group])

    # Base case (and the bottom of the recursion below): small enough to solve directly.
    if num_cities <= k_cap:
        return solve_clusters([coordinates])[0]

    clusters = partitioner(coordinates, k_cap)
    cluster_orders = solve_clusters([coordinates[cluster] for cluster in clusters])
    cluster_cycles: List[List[int]] = [
        [cluster[i] for i in order] for cluster, order in zip(clusters, cluster_orders)
    ]

    # Order the clusters by RECURSIVELY composing their centroids: the centroid TSP
    # is the same problem one level up, so the solver calls itself -- log_k(n) depth,
    # O(n) total, no O(m^2) blow-up. The recursion uses the fast base solver and no
    # cleanup; we only need the cluster visiting order here.
    centroids = np.asarray([coordinates[cluster].mean(axis=0) for cluster in clusters])
    cluster_order = compose_solve(centroids, k_cap, local_solver=_quick_solver,
                                  partitioner=partition_space_filling, cleanup="none")
    global_tour, seam_positions = stitch_dp(coordinates, cluster_cycles, cluster_order)

    if cleanup == "none":
        pass
    elif cleanup == "seam_2opt":
        allowed = _expand_positions(seam_positions, seam_window, num_cities)
        global_tour = two_opt(coordinates, global_tour, allowed_first=allowed)
    elif cleanup == "full_2opt":
        global_tour = two_opt(coordinates, global_tour)
    elif cleanup == "neighbor_2opt":
        global_tour = neighbor_two_opt(coordinates, global_tour)
    elif cleanup != "none":
        raise ValueError(f"unknown cleanup mode: {cleanup!r}")
    return global_tour


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 1) Stitching quality vs exact optimum, on instances Held-Karp can fully solve.
    small_gaps = []
    for _ in range(50):
        sample = rng.random((12, 2))
        _, optimal_distance = held_karp(sample)
        composed = compose_solve(sample, k_cap=6, cleanup="seam_2opt")
        assert sorted(composed) == list(range(12)), "composed tour invalid"
        small_gaps.append(tour_distance(sample, composed) / optimal_distance - 1)
    print(f"n=12, k=6 composed (seam-2opt) gap vs exact optimum: {np.mean(small_gaps)*100:.1f}%")

    # 2) Larger instance: composition vs the cheap space-filling-curve baseline.
    big = rng.random((300, 2))
    sfc_tour = space_filling_curve(big)
    sfc_distance = tour_distance(big, sfc_tour)
    for cleanup_mode in ("none", "seam_2opt", "full_2opt"):
        composed = compose_solve(big, k_cap=10, cleanup=cleanup_mode)
        assert sorted(composed) == list(range(300)), f"invalid tour ({cleanup_mode})"
        composed_distance = tour_distance(big, composed)
        print(f"n=300, k=10 composed[{cleanup_mode:9}]: {composed_distance:.3f} "
              f"({(composed_distance/sfc_distance-1)*100:+.1f}% vs space-filling-curve {sfc_distance:.3f})")
    print("OK: composition produces valid tours")
