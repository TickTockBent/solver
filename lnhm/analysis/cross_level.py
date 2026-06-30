"""Cross-level reinforcement experiment for LNHM Phase 0.

Trains Model A (full curriculum [3, N]), Model B (anchor level only, fixed
data), and Model C (anchor level only, unlimited fresh data) and compares
anchor-level accuracy. Success metric is A - C >= 5pp across seeds. Optional
frozen-encoder localization probe. Implemented in a later step; see
phase0/phase0-spec.md, "Cross-Level Reinforcement Test".
"""
