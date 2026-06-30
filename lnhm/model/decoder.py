"""Autoregressive decoder for LNHM Phase 0.

At each step selects the next city via attention over unvisited node embeddings,
masking already-visited nodes to -inf before softmax. This is a routing-specific
head and is not expected to transfer across problem classes.

Implemented in a later step; see phase0/phase0-spec.md, "Model Architecture".
"""
