"""Autoregressive decoder for LNHM Phase 0.

Follows the Attention Model pattern (Kool et al. 2019): at each step a context
vector forms a query, a multi-head *glimpse* attends over the node embeddings,
and a single-head compatibility layer (with tanh clipping) produces selection
logits over the remaining cities. Already-visited and padded nodes are masked to
-inf before the softmax.

Context vector (per the Phase 0 spec) = concat of:
  - the first city's embedding (return-trip awareness),
  - the last selected city's embedding,
  - the mean of the still-unvisited city embeddings.

Tours are canonicalized to start at city 0, so decoding always begins from
city 0 and the model predicts positions 1..n-1.

This is a routing-specific head and is *not* expected to transfer across problem
classes; the transferable structure is meant to live in the encoder.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gather_node(node_embeddings: torch.Tensor, node_index: torch.Tensor) -> torch.Tensor:
    """Select one embedding per batch element. (B, N, D), (B,) -> (B, D)."""
    batch_indices = torch.arange(node_embeddings.shape[0], device=node_embeddings.device)
    return node_embeddings[batch_indices, node_index]


def _masked_mean(node_embeddings: torch.Tensor, available_mask: torch.Tensor) -> torch.Tensor:
    """Mean over available nodes. (B, N, D), (B, N) bool -> (B, D)."""
    weights = available_mask.unsqueeze(-1).to(node_embeddings.dtype)
    summed = (node_embeddings * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp(min=1.0)
    return summed / count


class AutoregressiveDecoder(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 8, tanh_clipping: float = 10.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.tanh_clipping = tanh_clipping

        # Context (3*d_model from [first, last, mean_unvisited]) -> glimpse query.
        self.context_to_query = nn.Linear(3 * d_model, d_model)
        # Glimpse (multi-head) projections over the node embeddings.
        self.glimpse_key_projection = nn.Linear(d_model, d_model)
        self.glimpse_value_projection = nn.Linear(d_model, d_model)
        self.glimpse_output_projection = nn.Linear(d_model, d_model)
        # Final single-head compatibility keys for the selection logits.
        self.logit_key_projection = nn.Linear(d_model, d_model)

    def _split_heads(self, projected: torch.Tensor) -> torch.Tensor:
        # (B, N, D) -> (B, H, N, head_dim)
        batch_size, num_nodes, _ = projected.shape
        reshaped = projected.view(batch_size, num_nodes, self.n_heads, self.head_dim)
        return reshaped.permute(0, 2, 1, 3)

    def step_logits(
        self,
        node_embeddings: torch.Tensor,
        context_query: torch.Tensor,
        unavailable_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Selection logits over nodes for one decoding step.

        node_embeddings : (B, N, D)
        context_query   : (B, D)  -- query before head split
        unavailable_mask: (B, N) bool, True where a node may not be selected
                          (already visited or padding)
        returns         : (B, N) logits with -inf at unavailable nodes
        """
        batch_size, num_nodes, _ = node_embeddings.shape

        # --- multi-head glimpse ---
        query_heads = self._split_heads(self.context_to_query(context_query).unsqueeze(1))
        key_heads = self._split_heads(self.glimpse_key_projection(node_embeddings))
        value_heads = self._split_heads(self.glimpse_value_projection(node_embeddings))

        attention_scores = torch.matmul(query_heads, key_heads.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)  # (B, H, 1, N)
        mask_for_heads = unavailable_mask.view(batch_size, 1, 1, num_nodes)
        attention_scores = attention_scores.masked_fill(mask_for_heads, float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)

        glimpse_heads = torch.matmul(attention_weights, value_heads)  # (B, H, 1, head_dim)
        glimpse = glimpse_heads.permute(0, 2, 1, 3).reshape(batch_size, 1, self.d_model)
        glimpse = self.glimpse_output_projection(glimpse)  # (B, 1, D)

        # --- single-head compatibility -> logits ---
        logit_keys = self.logit_key_projection(node_embeddings)  # (B, N, D)
        compatibility = torch.matmul(logit_keys, glimpse.transpose(-2, -1)).squeeze(-1)
        compatibility = compatibility / math.sqrt(self.d_model)  # (B, N)
        clipped_logits = torch.tanh(compatibility) * self.tanh_clipping
        return clipped_logits.masked_fill(unavailable_mask, float("-inf"))

    def _initial_state(
        self, node_embeddings: torch.Tensor, node_padding_mask: torch.Tensor | None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_nodes, _ = node_embeddings.shape
        device = node_embeddings.device
        if node_padding_mask is None:
            visited_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
        else:
            visited_mask = node_padding_mask.clone()
        start_index = torch.zeros(batch_size, dtype=torch.long, device=device)  # canonical start = 0
        visited_mask[torch.arange(batch_size, device=device), start_index] = True
        first_embedding = _gather_node(node_embeddings, start_index)
        return visited_mask, start_index, first_embedding

    def forward(
        self,
        node_embeddings: torch.Tensor,
        target_tours: torch.Tensor,
        node_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced log-probabilities of the target tour.

        Returns per-step log-probabilities of the supervised next city, shape
        (B, seq_len - 1). The training loss is the negative mean of these.
        Assumes a uniform-n batch (canonical tours of equal length); mixed-n
        collation with per-step masking is handled by the training step.
        """
        sequence_length = target_tours.shape[1]
        visited_mask, last_index, first_embedding = self._initial_state(
            node_embeddings, node_padding_mask
        )

        per_step_log_probs = []
        for step in range(1, sequence_length):
            last_embedding = _gather_node(node_embeddings, last_index)
            mean_unvisited = _masked_mean(node_embeddings, ~visited_mask)
            context = torch.cat([first_embedding, last_embedding, mean_unvisited], dim=-1)

            logits = self.step_logits(node_embeddings, context, visited_mask)
            log_probabilities = F.log_softmax(logits, dim=-1)

            target_city = target_tours[:, step]
            chosen_log_prob = log_probabilities.gather(1, target_city.unsqueeze(1)).squeeze(1)
            per_step_log_probs.append(chosen_log_prob)

            visited_mask = visited_mask.clone()
            visited_mask[torch.arange(visited_mask.shape[0], device=visited_mask.device), target_city] = True
            last_index = target_city

        return torch.stack(per_step_log_probs, dim=1)

    @torch.no_grad()
    def decode(
        self,
        node_embeddings: torch.Tensor,
        node_padding_mask: torch.Tensor | None = None,
        mode: str = "greedy",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive rollout. Returns (tours, tour_log_probabilities).

        tours: (B, N) starting at city 0. Assumes a uniform-n batch (evaluation
        runs per level), so every instance has the same number of cities.
        """
        batch_size, num_nodes, _ = node_embeddings.shape
        visited_mask, last_index, first_embedding = self._initial_state(
            node_embeddings, node_padding_mask
        )

        selected_cities = [torch.zeros(batch_size, dtype=torch.long, device=node_embeddings.device)]
        total_log_probability = torch.zeros(batch_size, device=node_embeddings.device)

        for _ in range(1, num_nodes):
            last_embedding = _gather_node(node_embeddings, last_index)
            mean_unvisited = _masked_mean(node_embeddings, ~visited_mask)
            context = torch.cat([first_embedding, last_embedding, mean_unvisited], dim=-1)

            logits = self.step_logits(node_embeddings, context, visited_mask)
            log_probabilities = F.log_softmax(logits, dim=-1)

            if mode == "greedy":
                chosen_city = log_probabilities.argmax(dim=-1)
            elif mode == "sample":
                chosen_city = torch.distributions.Categorical(logits=logits).sample()
            else:
                raise ValueError(f"unknown decode mode: {mode!r}")

            total_log_probability = total_log_probability + log_probabilities.gather(
                1, chosen_city.unsqueeze(1)
            ).squeeze(1)
            selected_cities.append(chosen_city)

            visited_mask = visited_mask.clone()
            visited_mask[torch.arange(batch_size, device=visited_mask.device), chosen_city] = True
            last_index = chosen_city

        return torch.stack(selected_cities, dim=1), total_log_probability
