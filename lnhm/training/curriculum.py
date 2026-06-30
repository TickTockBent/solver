"""Additive curriculum scheduler for LNHM Phase 0.

Starts at the lowest level and adds one level at a time. The frontier (newest)
level takes the plurality of the batch mix; prior levels split the remainder
equally. The frontier graduates when its validation accuracy clears the
threshold (warm-up levels n=3,4 are degenerate and graduate automatically).

The mixing weight for the frontier is ``max(frontier_weight, 1/num_active)``:
this keeps the frontier at a true plurality even early on, reproducing the
spec's worked examples (2 levels -> 50/50; 3 levels -> 40/30/30;
4 levels -> 40/20/20/20).

See phase0/phase0-spec.md, "Curriculum Schedule".
"""
from __future__ import annotations

from typing import Dict, List, Set


class AdditiveCurriculum:
    def __init__(
        self,
        levels: List[int],
        warmup_levels: List[int],
        frontier_weight: float = 0.4,
    ):
        self.all_levels = sorted(levels)
        self.warmup_levels: Set[int] = set(warmup_levels)
        self.frontier_weight = frontier_weight
        self.frontier_index = 0  # index into all_levels of the current frontier

    @property
    def frontier_level(self) -> int:
        return self.all_levels[self.frontier_index]

    @property
    def active_levels(self) -> List[int]:
        return self.all_levels[: self.frontier_index + 1]

    @property
    def is_complete(self) -> bool:
        return self.frontier_index >= len(self.all_levels) - 1

    def level_weights(self) -> Dict[int, float]:
        """Batch-mix weight per active level (sums to 1)."""
        active = self.active_levels
        num_active = len(active)
        if num_active == 1:
            return {active[0]: 1.0}
        effective_frontier_weight = max(self.frontier_weight, 1.0 / num_active)
        prior_share = (1.0 - effective_frontier_weight) / (num_active - 1)
        weights = {level: prior_share for level in active[:-1]}
        weights[self.frontier_level] = effective_frontier_weight
        return weights

    def is_warmup_frontier(self) -> bool:
        return self.frontier_level in self.warmup_levels

    def should_graduate(self, frontier_val_accuracy: float, threshold: float) -> bool:
        """Warm-up levels graduate automatically; others need the threshold."""
        if self.is_warmup_frontier():
            return True
        return frontier_val_accuracy >= threshold

    def advance(self) -> bool:
        """Move the frontier to the next level. Returns False if already at the end."""
        if self.is_complete:
            return False
        self.frontier_index += 1
        return True
