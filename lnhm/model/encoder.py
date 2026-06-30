"""Transformer encoder for LNHM Phase 0.

Maps an n x 2 coordinate matrix to n x d_model node embeddings via linear
projection and N_enc self-attention layers. Per the spec, the encoder is the
component expected to carry scale-unified structure, so it is the focus of the
cross-level reinforcement and localization-probe experiments.

Implemented in a later step; see phase0/phase0-spec.md, "Model Architecture".
"""
