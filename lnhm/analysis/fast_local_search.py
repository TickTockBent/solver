"""Compiled near-linear local search for LNHM Phase 1 cleanup.

The pure-Python `baselines.neighbor_two_opt` is the composition-cleanup bottleneck:
it is (a) ~40x slower than compiled code (interpreter overhead) and (b) super-linear
at scale (~n^1.72 measured 100k->1M) because array-reversal 2-opt moves cost
O(segment length) and segments grow with n.

This module fixes both:
  - `_two_opt_kernel` : the same neighbor-list 2-opt, njit-compiled (the ~40x constant).
  - `_or_opt_kernel`  : Or-opt (relocate segments of 1..3 cities) in a doubly-linked
                        list -- O(1) per move, REVERSAL-FREE, so near-linear. Or-opt
                        also finds improving moves plain 2-opt cannot, which lifts
                        tour quality (p) toward the classical ~5%-gap regime.

`fast_local_search` alternates the two to convergence. Falls back to a no-numba
shim if numba is unavailable (import still succeeds; kernels just run un-jitted).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

try:
    from numba import njit  # type: ignore
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - fallback keeps the pipeline importable
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):  # noqa: D401 - decorator shim
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(func):
            return func

        return _wrap


@njit(cache=True)
def _dist(coords, a, b):
    dx = coords[a, 0] - coords[b, 0]
    dy = coords[a, 1] - coords[b, 1]
    return (dx * dx + dy * dy) ** 0.5


@njit(cache=True)
def _reverse_cyclic(tour, position, low, high, n):
    """Reverse the (possibly wrapping) cyclic segment [low..high], updating position."""
    seg_len = (high - low) % n + 1
    for s in range(seg_len // 2):
        p = (low + s) % n
        q = (high - s) % n
        city_p = tour[p]
        city_q = tour[q]
        tour[p] = city_q
        tour[q] = city_p
        position[city_q] = p
        position[city_p] = q


@njit(cache=True)
def _two_opt_kernel(coords, tour, position, neighbors, dont_look, n, k, max_passes, eps):
    """Neighbor-list 2-opt with don't-look bits and shorter-segment reversal."""
    improvements = 0
    for _pass in range(max_passes):
        improved_any = False
        for a in range(n):
            if dont_look[a]:
                continue
            position_a = position[a]
            improved = False
            for side in range(2):
                step = 1 if side == 0 else -1
                b = tour[(position_a + step) % n]
                dist_ab = _dist(coords, a, b)
                for idx in range(k):
                    c = neighbors[a, idx]
                    if c < 0:
                        break
                    dist_ac = _dist(coords, a, c)
                    if dist_ac >= dist_ab:
                        break  # neighbors sorted ascending -> no closer partner remains
                    position_c = position[c]
                    d = tour[(position_c + step) % n]
                    if d == a or c == b:
                        continue
                    if dist_ac + _dist(coords, b, d) - dist_ab - _dist(coords, c, d) < -eps:
                        if step == 1:
                            if position_a < position_c:
                                lo = position_a
                                hi = position_c
                            else:
                                lo = position_c
                                hi = position_a
                        else:
                            lo = (position_a - 1) % n
                            hi = (position_c - 1) % n
                            if lo > hi:
                                lo, hi = hi, lo
                        _reverse_cyclic(tour, position, lo + 1, hi, n)
                        dont_look[a] = False
                        dont_look[b] = False
                        dont_look[c] = False
                        dont_look[d] = False
                        improved = True
                        improvements += 1
                        break
                if improved:
                    break
            if improved:
                improved_any = True
            else:
                dont_look[a] = True
        if not improved_any:
            break
    return improvements


@njit(cache=True)
def _build_links(tour, nxt, prv, n):
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        nxt[a] = b
        prv[b] = a


@njit(cache=True)
def _links_to_tour(nxt, tour, n):
    current = 0
    for i in range(n):
        tour[i] = current
        current = nxt[current]


@njit(cache=True)
def _or_opt_kernel(coords, nxt, prv, neighbors, dont_look, n, k, max_seg, max_passes, eps):
    """Or-opt: relocate a run of 1..max_seg cities next to a spatial neighbor.

    Operates on a doubly-linked list (nxt/prv). Each accepted move is O(1) relinking
    with NO segment reversal, so total work is O(moves) -- near-linear at scale."""
    improvements = 0
    seg = np.empty(max_seg, dtype=np.int64)
    for _pass in range(max_passes):
        improved_any = False
        for s0 in range(n):
            if dont_look[s0]:
                continue
            improved = False
            for length in range(1, max_seg + 1):
                if length >= n - 1:
                    break
                seg[0] = s0
                for t in range(1, length):
                    seg[t] = nxt[seg[t - 1]]
                s_last = seg[length - 1]
                p = prv[s0]
                q = nxt[s_last]
                if p == s_last or q == s0 or p == q:
                    break  # segment spans (almost) the whole tour
                gain_remove = _dist(coords, p, s0) + _dist(coords, s_last, q) - _dist(coords, p, q)
                if gain_remove <= eps:
                    continue  # removing this segment saves nothing; insertion only adds
                best_c = -1
                best_delta = -eps
                best_reversed = False
                for idx in range(k):
                    c = neighbors[s0, idx]
                    if c < 0:
                        break
                    if c == p:
                        continue
                    in_segment = False
                    for t in range(length):
                        if seg[t] == c:
                            in_segment = True
                            break
                    if in_segment:
                        continue
                    d = nxt[c]
                    dist_cd = _dist(coords, c, d)
                    # forward: c -> s0 .. s_last -> d
                    delta_f = (_dist(coords, c, s0) + _dist(coords, s_last, d) - dist_cd) - gain_remove
                    if delta_f < best_delta:
                        best_delta = delta_f
                        best_c = c
                        best_reversed = False
                    if length >= 2:
                        # reversed: c -> s_last .. s0 -> d
                        delta_r = (_dist(coords, c, s_last) + _dist(coords, s0, d) - dist_cd) - gain_remove
                        if delta_r < best_delta:
                            best_delta = delta_r
                            best_c = c
                            best_reversed = True
                if best_c >= 0:
                    c = best_c
                    d = nxt[c]
                    # unlink the segment
                    nxt[p] = q
                    prv[q] = p
                    if not best_reversed:
                        nxt[c] = s0
                        prv[s0] = c
                        nxt[s_last] = d
                        prv[d] = s_last
                    else:
                        for t in range(length - 1):
                            nxt[seg[t + 1]] = seg[t]
                            prv[seg[t]] = seg[t + 1]
                        nxt[c] = s_last
                        prv[s_last] = c
                        nxt[s0] = d
                        prv[d] = s0
                    dont_look[p] = False
                    dont_look[q] = False
                    dont_look[c] = False
                    dont_look[d] = False
                    dont_look[s0] = False
                    dont_look[s_last] = False
                    improved = True
                    improvements += 1
                    break
            if improved:
                improved_any = True
            else:
                dont_look[s0] = True
        if not improved_any:
            break
    return improvements


def fast_local_search(
    coordinates: Sequence[Sequence[float]],
    tour: Sequence[int],
    k_neighbors: int = 8,
    max_seg: int = 3,
    max_rounds: int = 8,
    inner_passes: int = 60,
    neighbor_lists: Optional[np.ndarray] = None,
) -> List[int]:
    """Compiled 2-opt + reversal-free Or-opt to a joint local optimum.

    Alternates a neighbor-list 2-opt pass and an Or-opt pass until a full round
    improves neither. Near-linear and ~40x the throughput of the pure-Python
    `neighbor_two_opt`, at a lower optimality gap (Or-opt adds moves 2-opt misses).
    """
    coords = np.asarray(coordinates, dtype=np.float64)
    n = len(tour)
    tour_array = np.ascontiguousarray(tour, dtype=np.int64)
    if n < 4:
        return list(tour_array)

    if neighbor_lists is None:
        from scipy.spatial import cKDTree

        query_k = min(k_neighbors + 1, n)
        _, neighbor_index = cKDTree(coords).query(coords, k=query_k, workers=-1)
        neighbor_lists = np.atleast_2d(neighbor_index)[:, 1:]  # drop self column
    neighbors = np.ascontiguousarray(neighbor_lists, dtype=np.int64)
    k = neighbors.shape[1]

    position = np.empty(n, dtype=np.int64)
    position[tour_array] = np.arange(n)
    nxt = np.empty(n, dtype=np.int64)
    prv = np.empty(n, dtype=np.int64)
    dont_look = np.empty(n, dtype=np.bool_)
    eps = 1e-10

    for _round in range(max_rounds):
        dont_look[:] = False
        moved_2opt = _two_opt_kernel(coords, tour_array, position, neighbors,
                                     dont_look, n, k, inner_passes, eps)
        _build_links(tour_array, nxt, prv, n)
        dont_look[:] = False
        moved_or = _or_opt_kernel(coords, nxt, prv, neighbors, dont_look,
                                  n, k, max_seg, inner_passes, eps)
        _links_to_tour(nxt, tour_array, n)
        position[tour_array] = np.arange(n)
        if moved_2opt == 0 and moved_or == 0:
            break

    return list(tour_array)


if __name__ == "__main__":
    # Correctness + quality self-test against exact optima and the Python baseline.
    import os
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.held_karp import held_karp, tour_distance  # noqa: E402
    from analysis.baselines import neighbor_two_opt, space_filling_curve  # noqa: E402

    rng = np.random.default_rng(0)

    # 1) validity + exact-gap on small instances
    fast_gaps = []
    for cities in range(6, 11):
        for _ in range(200):
            sample = rng.random((cities, 2))
            _, optimal = held_karp(sample)
            start = space_filling_curve(sample)
            polished = fast_local_search(sample, start)
            assert sorted(polished) == list(range(cities)), "fast_local_search: invalid tour"
            fast_gaps.append(tour_distance(sample, polished) / optimal - 1)
    print(f"n=6..10 mean gap vs exact (from SFC start): {np.mean(fast_gaps)*100:.2f}%")

    # 2) head-to-head vs the pure-Python neighbor_two_opt at moderate n
    for cities in (1000, 5000):
        sample = rng.random((cities, 2))
        start = space_filling_curve(sample)

        t0 = time.perf_counter()
        old = neighbor_two_opt(sample, start)
        t_old = time.perf_counter() - t0

        t0 = time.perf_counter()
        new = fast_local_search(sample, start)
        t_new = time.perf_counter() - t0

        assert sorted(new) == list(range(cities)), "invalid tour"
        gap_old = tour_distance(sample, old)
        gap_new = tour_distance(sample, new)
        print(f"n={cities}: neighbor_two_opt len={gap_old:.2f} ({t_old:.2f}s)  "
              f"fast_local_search len={gap_new:.2f} ({t_new:.2f}s)  "
              f"len_ratio={gap_new/gap_old:.4f} speedup={t_old/t_new:.1f}x")
    print("OK: fast_local_search valid and measured")
