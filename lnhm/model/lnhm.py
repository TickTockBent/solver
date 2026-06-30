"""Full LNHM Phase 0 model: transformer encoder + autoregressive decoder.

Combines :class:`TspEncoder` and :class:`AutoregressiveDecoder`. Supervised
training calls :meth:`forward` (teacher-forced log-probabilities of the
canonical optimal tour); evaluation calls :meth:`solve` (greedy/sampled
rollout). See phase0/phase0-spec.md, "Model Architecture".
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

try:
    from .encoder import TspEncoder
    from .decoder import AutoregressiveDecoder
except ImportError:  # allow `python model/lnhm.py`-style imports
    from encoder import TspEncoder
    from decoder import AutoregressiveDecoder


class LnhmModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_encoder_layers: int = 3,
        n_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        tanh_clipping: float = 10.0,
        input_dim: int = 2,
    ):
        super().__init__()
        self.encoder = TspEncoder(
            d_model=d_model,
            n_encoder_layers=n_encoder_layers,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            input_dim=input_dim,
        )
        self.decoder = AutoregressiveDecoder(
            d_model=d_model, n_heads=n_heads, tanh_clipping=tanh_clipping
        )

    @classmethod
    def from_config(cls, config: dict) -> "LnhmModel":
        """Build from the ``model:`` section of configs/phase0.yaml."""
        return cls(
            d_model=config.get("d_model", 128),
            n_encoder_layers=config.get("n_encoder_layers", 3),
            n_heads=config.get("n_heads", 8),
            ff_dim=config.get("ff_dim", 512),
            dropout=config.get("dropout", 0.1),
        )

    def forward(
        self,
        coordinates: torch.Tensor,
        target_tours: torch.Tensor,
        node_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced per-step log-probabilities of the target tour.

        Returns (batch, seq_len - 1); training loss is the negative mean.
        """
        node_embeddings = self.encoder(coordinates, node_padding_mask)
        return self.decoder(node_embeddings, target_tours, node_padding_mask)

    @torch.no_grad()
    def solve(
        self,
        coordinates: torch.Tensor,
        node_padding_mask: torch.Tensor | None = None,
        mode: str = "greedy",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Produce tours by autoregressive rollout. Returns (tours, log_probs)."""
        node_embeddings = self.encoder(coordinates, node_padding_mask)
        return self.decoder.decode(node_embeddings, node_padding_mask, mode=mode)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
