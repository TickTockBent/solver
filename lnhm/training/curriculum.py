"""Additive curriculum scheduler for LNHM Phase 0.

Warm-up on n=3,4 then add one level at a time; the frontier level takes the
plurality of the batch mix and prior levels split the remainder equally.
Measurement anchors at n>=5. Implemented in a later step; see
phase0/phase0-spec.md, "Curriculum Schedule".
"""
