"""Transformer encoder for LNHM Phase 0.

Maps an ``n x 2`` coordinate matrix to ``n x d_model`` node embeddings via a
linear projection followed by ``n_encoder_layers`` of multi-head self-attention.
No positional encoding is used: a TSP instance is an unordered *set* of nodes.

Per the Phase 0 spec, the encoder is the component expected to carry
scale-unified structure, so it is the target of the cross-level reinforcement
and frozen-encoder localization experiments. Keep it cleanly separable from the
decoder for that reason.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TspEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_encoder_layers: int = 3,
        n_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        input_dim: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

    def forward(
        self,
        coordinates: torch.Tensor,
        node_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of instances.

        Parameters
        ----------
        coordinates : (batch, num_nodes, input_dim) float tensor
        node_padding_mask : (batch, num_nodes) bool tensor, True where a node is
            padding (used to make batches of mixed n). Padded nodes are excluded
            from attention.

        Returns
        -------
        node_embeddings : (batch, num_nodes, d_model) float tensor
        """
        projected_coordinates = self.input_projection(coordinates)
        node_embeddings = self.transformer_encoder(
            projected_coordinates, src_key_padding_mask=node_padding_mask
        )
        return node_embeddings
