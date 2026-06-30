"""Exact TSP solvers for LNHM Phase 0.

Held-Karp dynamic programming is the primary solver for every Phase 0 instance
size (n <= 12). It is exact and runs in well under 10 ms per instance at the top
of that range, so the whole Phase 0 dataset generates in minutes on one machine.
Full-permutation enumeration (``brute_force``) is retained only as an independent
correctness oracle for small n.

Tours are returned as a list of city indices beginning at city 0. Call
``canonicalize_tour`` to additionally fix the traversal direction before storing
a tour as a supervised training target (see the Phase 0 spec, "Target
Canonicalization").
"""
from __future__ import annotations

import itertools
import math
from typing import List, Sequence, Tuple

import numpy as np

Coordinates = Sequence[Sequence[float]]


def euclidean_distance_matrix(coordinates: Coordinates) -> np.ndarray:
    """Full pairwise Euclidean distance matrix for a set of 2D points."""
    points = np.asarray(coordinates, dtype=np.float64)
    pairwise_differences = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    return np.sqrt((pairwise_differences ** 2).sum(axis=-1))


def tour_distance(coordinates: Coordinates, tour: Sequence[int]) -> float:
    """Total length of a closed tour visiting ``tour`` in order and returning."""
    ordered_points = np.asarray(coordinates, dtype=np.float64)[list(tour)]
    next_points = np.roll(ordered_points, -1, axis=0)
    leg_lengths = np.sqrt(((ordered_points - next_points) ** 2).sum(axis=1))
    return float(leg_lengths.sum())


def canonicalize_tour(tour: Sequence[int]) -> List[int]:
    """Return the unique representative of an undirected cycle.

    Rotates the tour to start at city 0, then orients it so the second city
    index is smaller than the last. This removes the rotation/reflection
    symmetry that would otherwise make teacher-forcing cross-entropy penalize
    valid equivalent tours.
    """
    city_sequence = list(tour)
    start_position = city_sequence.index(0)
    rotated = city_sequence[start_position:] + city_sequence[:start_position]
    if len(rotated) > 2 and rotated[-1] < rotated[1]:
        rotated = [rotated[0]] + rotated[1:][::-1]
    return rotated


def held_karp(coordinates: Coordinates) -> Tuple[List[int], float]:
    """Exact shortest tour via Held-Karp dynamic programming. O(n^2 * 2^n)."""
    distance_matrix = euclidean_distance_matrix(coordinates)
    num_cities = distance_matrix.shape[0]

    if num_cities <= 1:
        return [0], 0.0
    if num_cities == 2:
        return [0, 1], float(distance_matrix[0, 1] + distance_matrix[1, 0])

    # best_path[(visited_bits, end_city)] = (min_cost, predecessor_city)
    #
    # visited_bits is a bitmask over cities 1..num_cities-1; city 0 is the fixed
    # start and is implicit. Bit (city - 1) is set when `city` has been visited.
    # The entry is the minimum cost of a path that starts at city 0, visits
    # exactly the cities in visited_bits, and ends at end_city.
    best_path = {}

    # Base case: paths from the start directly to a single other city.
    for end_city in range(1, num_cities):
        single_city_bits = 1 << (end_city - 1)
        best_path[(single_city_bits, end_city)] = (
            float(distance_matrix[0, end_city]),
            0,
        )

    # Grow the visited set one city at a time.
    for subset_size in range(2, num_cities):
        for visited_subset in itertools.combinations(range(1, num_cities), subset_size):
            subset_bits = 0
            for city in visited_subset:
                subset_bits |= 1 << (city - 1)
            for end_city in visited_subset:
                bits_without_end = subset_bits & ~(1 << (end_city - 1))
                best_cost = math.inf
                best_predecessor = 0
                for predecessor in visited_subset:
                    if predecessor == end_city:
                        continue
                    candidate_cost = (
                        best_path[(bits_without_end, predecessor)][0]
                        + distance_matrix[predecessor, end_city]
                    )
                    if candidate_cost < best_cost:
                        best_cost = candidate_cost
                        best_predecessor = predecessor
                best_path[(subset_bits, end_city)] = (best_cost, best_predecessor)

    # Close the tour: every city visited, then return to the start.
    all_visited_bits = (1 << (num_cities - 1)) - 1
    best_total_cost = math.inf
    best_final_city = 0
    for end_city in range(1, num_cities):
        candidate_cost = (
            best_path[(all_visited_bits, end_city)][0]
            + distance_matrix[end_city, 0]
        )
        if candidate_cost < best_total_cost:
            best_total_cost = candidate_cost
            best_final_city = end_city

    # Walk predecessor pointers back to the start to recover the tour.
    optimal_tour: List[int] = []
    remaining_bits = all_visited_bits
    current_city = best_final_city
    while current_city != 0:
        optimal_tour.append(current_city)
        predecessor = best_path[(remaining_bits, current_city)][1]
        remaining_bits &= ~(1 << (current_city - 1))
        current_city = predecessor
    optimal_tour.append(0)
    optimal_tour.reverse()

    return optimal_tour, float(best_total_cost)


def brute_force(coordinates: Coordinates) -> Tuple[List[int], float]:
    """Exact shortest tour by full enumeration. Correctness oracle for small n."""
    points = np.asarray(coordinates, dtype=np.float64)
    num_cities = points.shape[0]
    if num_cities <= 2:
        return held_karp(points)

    best_distance = math.inf
    best_tour: List[int] = []
    # Fix city 0 as the start; enumerate orderings of the remaining cities.
    for tail_ordering in itertools.permutations(range(1, num_cities)):
        candidate_tour = [0, *tail_ordering]
        candidate_distance = tour_distance(points, candidate_tour)
        if candidate_distance < best_distance:
            best_distance = candidate_distance
            best_tour = candidate_tour
    return best_tour, float(best_distance)


if __name__ == "__main__":
    # Self-test: Held-Karp must match the brute-force oracle on random instances.
    rng = np.random.default_rng(0)
    DISTANCE_TOLERANCE = 1e-9
    total_checked = 0
    for cities in range(3, 9):
        for _ in range(200):
            sample_coordinates = rng.random((cities, 2))
            _, held_karp_distance = held_karp(sample_coordinates)
            _, brute_force_distance = brute_force(sample_coordinates)
            assert abs(held_karp_distance - brute_force_distance) < DISTANCE_TOLERANCE, (
                f"mismatch at n={cities}: "
                f"held_karp={held_karp_distance} brute_force={brute_force_distance}"
            )
            total_checked += 1
    print(f"OK: Held-Karp matched brute force on {total_checked} random instances (n=3..8)")
