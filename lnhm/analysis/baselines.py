"""Classical TSP baselines for LNHM Phase 1.

Reference points for the composition cost/quality frontier:
  - nearest_neighbor    : greedy construction (~25% over optimal).
  - two_opt             : local search to a 2-opt optimum (~5%); supports
                          seam-restricted moves via `allowed_first`.
  - space_filling_curve : Hilbert-curve ordering, O(n log n) (~25%) -- the cheap
                          near-linear competitor and a natural partition/stitch order.
  - lkh_tour            : LKH-3 wrapper (near-optimal reference). Needs the LKH binary.

Tours are lists of city indices forming a cycle (implicit return to start).
Distances use data.held_karp.tour_distance for consistency.
"""
from __future__ import annotations

import os
import sys
from typing import Iterable, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from data.held_karp import euclidean_distance_matrix, tour_distance  # noqa: E402


def nearest_neighbor(coordinates: Sequence[Sequence[float]], start: int = 0) -> List[int]:
    """Greedy nearest-neighbor construction from `start`."""
    distance_matrix = euclidean_distance_matrix(coordinates)
    num_cities = distance_matrix.shape[0]
    visited = np.zeros(num_cities, dtype=bool)
    visited[start] = True
    tour = [start]
    current_city = start
    for _ in range(num_cities - 1):
        distances_from_current = distance_matrix[current_city].copy()
        distances_from_current[visited] = np.inf
        next_city = int(distances_from_current.argmin())
        visited[next_city] = True
        tour.append(next_city)
        current_city = next_city
    return tour


def two_opt(
    coordinates: Sequence[Sequence[float]],
    tour: Sequence[int],
    allowed_first: Optional[Iterable[int]] = None,
    max_passes: int = 1000,
    eps: float = 1e-9,
) -> List[int]:
    """2-opt local search to a local optimum (first-improvement).

    `allowed_first` restricts the first edge's position to a given set (e.g. seam
    positions), so a cleanup pass can target only cluster boundaries.
    """
    distance_matrix = euclidean_distance_matrix(coordinates)
    tour = list(tour)
    num_cities = len(tour)
    if num_cities < 4:
        return tour
    first_positions = list(allowed_first) if allowed_first is not None else list(range(num_cities - 1))

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in first_positions:
            if i >= num_cities - 1:
                continue
            city_i = tour[i]
            city_i_next = tour[i + 1]
            for j in range(i + 2, num_cities):
                j_next = (j + 1) % num_cities
                if j_next == i:
                    continue
                city_j = tour[j]
                city_j_next = tour[j_next]
                delta = (distance_matrix[city_i, city_j] + distance_matrix[city_i_next, city_j_next]
                         - distance_matrix[city_i, city_i_next] - distance_matrix[city_j, city_j_next])
                if delta < -eps:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    city_i_next = tour[i + 1]  # refresh after the reversal
                    improved = True
    return tour


def space_filling_curve(coordinates: Sequence[Sequence[float]], order: int = 16) -> List[int]:
    """Order points along a Hilbert curve. Returns the visiting order (a cycle).

    Vectorized over all points (numpy loop over the `order` bit levels), so it
    scales to millions of points in well under a second."""
    points = np.asarray(coordinates, dtype=np.float64)
    side = 1 << order
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    span = np.where(upper > lower, upper - lower, 1.0)
    scaled = np.clip(((points - lower) / span * (side - 1)).round().astype(np.int64), 0, side - 1)

    x = scaled[:, 0].copy()
    y = scaled[:, 1].copy()
    hilbert = np.zeros(len(points), dtype=np.int64)
    s = side // 2
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        hilbert += s * s * ((3 * rx) ^ ry)
        # rotate quadrant: flip where (ry==0 & rx==1), then swap x/y where ry==0
        flip = (ry == 0) & (rx == 1)
        x = np.where(flip, side - 1 - x, x)
        y = np.where(flip, side - 1 - y, y)
        swap = ry == 0
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        s //= 2
    return list(np.argsort(hilbert, kind="stable"))


def _reverse_cyclic(tour: List[int], position: np.ndarray, low: int, high: int, n: int) -> None:
    """Reverse the cyclic tour segment [low..high] (inclusive, may wrap), updating position."""
    segment_length = (high - low) % n + 1
    for step in range(segment_length // 2):
        p = (low + step) % n
        q = (high - step) % n
        tour[p], tour[q] = tour[q], tour[p]
        position[tour[p]] = p
        position[tour[q]] = q


def neighbor_two_opt(
    coordinates: Sequence[Sequence[float]],
    tour: Sequence[int],
    k_neighbors: int = 8,
    max_passes: int = 40,
    time_limit: float = None,
    neighbor_lists: np.ndarray = None,
) -> List[int]:
    """Near-linear 2-opt: each city only considers 2-opt partners among its k spatial
    nearest neighbors (k-d tree), with don't-look bits and shorter-segment reversal.

    This is the honest cheap-and-good baseline for large TSP (what real solvers use
    under the hood), and the near-linear cleanup for composition at scale.
    """
    from scipy.spatial import cKDTree
    import time as _time

    coords = np.asarray(coordinates, dtype=np.float64)
    n = len(tour)
    tour = list(tour)
    if n < 4:
        return tour
    if neighbor_lists is None:
        _, neighbor_index = cKDTree(coords).query(coords, k=min(k_neighbors + 1, n), workers=-1)
        neighbor_lists = np.atleast_2d(neighbor_index)[:, 1:]  # drop self column

    position = np.empty(n, dtype=np.int64)
    for index, city in enumerate(tour):
        position[city] = index

    def dist(a: int, b: int) -> float:
        dx = coords[a, 0] - coords[b, 0]
        dy = coords[a, 1] - coords[b, 1]
        return (dx * dx + dy * dy) ** 0.5

    dont_look = np.zeros(n, dtype=bool)
    started = _time.perf_counter()
    for _pass in range(max_passes):
        improved_any = False
        for a in range(n):
            if dont_look[a]:
                continue
            if time_limit is not None and _time.perf_counter() - started > time_limit:
                return tour
            position_a = position[a]
            improved = False
            for step in (1, -1):  # break a's successor edge, then its predecessor edge
                b = tour[(position_a + step) % n]
                dist_ab = dist(a, b)
                for c in neighbor_lists[a]:
                    dist_ac = dist(a, c)
                    if dist_ac >= dist_ab:
                        break  # neighbors sorted ascending -> no closer partner remains
                    position_c = position[c]
                    d = tour[(position_c + step) % n]
                    if d == a or c == b:
                        continue
                    if dist_ac + dist(b, d) - dist_ab - dist(c, d) < -1e-10:
                        if step == 1:
                            i, j = (position_a, position_c) if position_a < position_c else (position_c, position_a)
                        else:
                            i, j = (position_a - 1) % n, (position_c - 1) % n
                            if i > j:
                                i, j = j, i
                        _reverse_cyclic(tour, position, i + 1, j, n)
                        for touched in (a, b, c, d):
                            dont_look[touched] = False
                        improved = improved_any = True
                        break
                if improved:
                    break
            if not improved:
                dont_look[a] = True
        if not improved_any:
            break
    return tour


def lkh_tour(
    coordinates: Sequence[Sequence[float]],
    lkh_binary: str = "LKH",
    runs: int = 1,
    coordinate_scale: int = 1_000_000,
    workdir: Optional[str] = None,
) -> List[int]:
    """Near-optimal tour via LKH-3. Requires the LKH binary on PATH (or `lkh_binary`).

    Coordinates are scaled to integers (TSPLIB EUC_2D rounds distances). The
    returned tour's true length should be recomputed with `tour_distance` on the
    original float coordinates, not trusted from LKH's integer objective.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which(lkh_binary) is None:
        raise RuntimeError(
            f"LKH binary '{lkh_binary}' not found on PATH. Build LKH-3 "
            f"(http://akira.ruc.dk/~keld/research/LKH-3/) and pass lkh_binary=..."
        )
    points = np.asarray(coordinates, dtype=np.float64)
    num_cities = len(points)
    scaled = (points * coordinate_scale).round().astype(np.int64)

    directory = tempfile.mkdtemp(prefix="lkh_", dir=workdir)
    problem_path = os.path.join(directory, "problem.tsp")
    parameter_path = os.path.join(directory, "problem.par")
    tour_path = os.path.join(directory, "problem.tour")

    with open(problem_path, "w") as problem_file:
        problem_file.write(
            f"NAME: problem\nTYPE: TSP\nDIMENSION: {num_cities}\n"
            f"EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n"
        )
        for index, (x, y) in enumerate(scaled, start=1):
            problem_file.write(f"{index} {int(x)} {int(y)}\n")
        problem_file.write("EOF\n")
    with open(parameter_path, "w") as parameter_file:
        parameter_file.write(f"PROBLEM_FILE = {problem_path}\nRUNS = {runs}\nTOUR_FILE = {tour_path}\n")

    subprocess.run([lkh_binary, parameter_path], check=True, capture_output=True)

    tour: List[int] = []
    with open(tour_path) as tour_file:
        in_tour_section = False
        for line in tour_file:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_tour_section = True
                continue
            if in_tour_section:
                if line in ("-1", "EOF", ""):
                    break
                tour.append(int(line) - 1)
    return tour


if __name__ == "__main__":
    # Self-test on small instances where the exact optimum is known via Held-Karp.
    sys.path.insert(0, PROJECT_ROOT)
    from data.held_karp import held_karp

    rng = np.random.default_rng(0)
    nn_gaps, two_opt_gaps, sfc_gaps = [], [], []
    for cities in range(6, 11):
        for _ in range(100):
            sample = rng.random((cities, 2))
            _, optimal_distance = held_karp(sample)

            nn = nearest_neighbor(sample)
            assert sorted(nn) == list(range(cities)), "NN produced invalid tour"
            nn_gaps.append(tour_distance(sample, nn) / optimal_distance - 1)

            polished = two_opt(sample, nn)
            assert sorted(polished) == list(range(cities)), "2-opt produced invalid tour"
            two_opt_gaps.append(tour_distance(sample, polished) / optimal_distance - 1)

            sfc = space_filling_curve(sample)
            assert sorted(sfc) == list(range(cities)), "SFC produced invalid tour"
            sfc_gaps.append(tour_distance(sample, sfc) / optimal_distance - 1)

    print(f"mean optimality gap over n=6..10 (vs exact):")
    print(f"  nearest_neighbor   : {np.mean(nn_gaps)*100:5.1f}%")
    print(f"  2-opt (from NN)    : {np.mean(two_opt_gaps)*100:5.1f}%")
    print(f"  space_filling_curve: {np.mean(sfc_gaps)*100:5.1f}%")
    print("OK: all baselines produce valid tours")
