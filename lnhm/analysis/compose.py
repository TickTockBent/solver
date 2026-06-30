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
from analysis.baselines import space_filling_curve, two_opt  # noqa: E402

# A local solver maps cluster coordinates (m, 2) -> a visiting order of 0..m-1.
LocalSolver = Callable[[np.ndarray], List[int]]


def held_karp_solver(cluster_coordinates: np.ndarray) -> List[int]:
    """Exact local solver (for clusters within Held-Karp's reach)."""
    tour, _ = held_karp(cluster_coordinates)
    return tour


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
def _open_cycle_at_longest_edge(global_indices: List[int], local_order: List[int],
                                coordinates: np.ndarray) -> List[int]:
    """Turn a cluster's cyclic sub-tour into a path by cutting its longest edge.

    The two path endpoints become the cluster's natural ports for stitching.
    """
    cyclic = [global_indices[local] for local in local_order]
    length = len(cyclic)
    if length <= 2:
        return cyclic
    worst_distance, worst_position = -1.0, 0
    for k in range(length):
        a = cyclic[k]
        b = cyclic[(k + 1) % length]
        edge = float(np.linalg.norm(coordinates[a] - coordinates[b]))
        if edge > worst_distance:
            worst_distance, worst_position = edge, k
    return cyclic[worst_position + 1:] + cyclic[:worst_position + 1]


def stitch_greedy_ports(coordinates: np.ndarray, cluster_paths: List[List[int]],
                        cluster_order: List[int]) -> Tuple[List[int], Set[int]]:
    """Concatenate cluster paths in order, flipping each to connect at its nearer port.

    Returns (global_tour, seam_edge_positions) where seam positions index the first
    edge of each junction in the global tour (for seam-restricted cleanup).
    """
    coordinates = np.asarray(coordinates, dtype=np.float64)
    global_tour: List[int] = []
    seam_positions: Set[int] = set()
    current_end = None
    for cluster_index in cluster_order:
        path = list(cluster_paths[cluster_index])
        if current_end is not None:
            distance_to_start = float(np.linalg.norm(coordinates[current_end] - coordinates[path[0]]))
            distance_to_end = float(np.linalg.norm(coordinates[current_end] - coordinates[path[-1]]))
            if distance_to_end < distance_to_start:
                path = path[::-1]
            seam_positions.add(len(global_tour) - 1)  # first-edge position of the junction
        global_tour.extend(path)
        current_end = global_tour[-1]
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
    cleanup: str = "none",   # "none" | "seam_2opt" | "full_2opt"
    seam_window: int = 2,
) -> List[int]:
    """Solve a large instance by composition. Returns a global tour (valid cycle)."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    num_cities = len(coordinates)

    clusters = partitioner(coordinates, k_cap)
    cluster_paths: List[List[int]] = []
    centroids: List[np.ndarray] = []
    for cluster in clusters:
        local_order = local_solver(coordinates[cluster])
        cluster_paths.append(_open_cycle_at_longest_edge(cluster, local_order, coordinates))
        centroids.append(coordinates[cluster].mean(axis=0))

    if len(clusters) > 1:
        cluster_order = space_filling_curve(np.asarray(centroids))  # TODO: recurse for huge n
    else:
        cluster_order = [0]

    global_tour, seam_positions = stitch_greedy_ports(coordinates, cluster_paths, cluster_order)

    if cleanup == "seam_2opt":
        allowed = _expand_positions(seam_positions, seam_window, num_cities)
        global_tour = two_opt(coordinates, global_tour, allowed_first=allowed)
    elif cleanup == "full_2opt":
        global_tour = two_opt(coordinates, global_tour)
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
