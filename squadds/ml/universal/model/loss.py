"""Masked multi-task loss for the Universal GNN."""

import torch
from torch import nn


class MaskedMultiTaskLoss(nn.Module):
    """Computes Mean Squared Error loss only on unmasked (non-NaN) targets.

    Handles sparse labels where different nodes/edges predict different subset of parameters.
    """

    def __init__(self, node_weights: list[float] | None = None, edge_weights: list[float] | None = None):
        """
        Args:
            node_weights: Optional list of weights for each node target.
            edge_weights: Optional list of weights for each edge target.
        """
        super().__init__()
        self.node_weights = torch.tensor(node_weights) if node_weights else None
        self.edge_weights = torch.tensor(edge_weights) if edge_weights else None
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        node_preds: torch.Tensor,
        node_targets: torch.Tensor,
        edge_preds: torch.Tensor,
        edge_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            node_preds: Predictions of shape [num_nodes, num_node_targets].
            node_targets: Ground truth of shape [num_nodes, num_node_targets] (with NaNs).
            edge_preds: Predictions of shape [num_edges, num_edge_targets].
            edge_targets: Ground truth of shape [num_edges, num_edge_targets] (with NaNs).

        Returns:
            Tuple of (total_loss, node_loss, edge_loss).
        """
        device = node_preds.device

        if self.node_weights is not None:
            self.node_weights = self.node_weights.to(device)
        if self.edge_weights is not None:
            self.edge_weights = self.edge_weights.to(device)

        # ── Node Loss ──────────────────────────────────────────────────
        node_mask = ~torch.isnan(node_targets)
        if node_mask.any():
            n_loss_elements = self.mse(node_preds[node_mask], node_targets[node_mask])
            loss_node = n_loss_elements.mean()
        else:
            loss_node = (0.0 * node_preds).sum()

        # ── Edge Loss ──────────────────────────────────────────────────
        edge_mask = ~torch.isnan(edge_targets)
        if edge_mask.any():
            e_loss_elements = self.mse(edge_preds[edge_mask], edge_targets[edge_mask])
            loss_edge = e_loss_elements.mean()
        else:
            loss_edge = (0.0 * edge_preds).sum()

        total_loss = loss_node + loss_edge

        return total_loss, loss_node, loss_edge
