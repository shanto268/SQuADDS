"""Prediction heads for the Universal GNN."""

import torch
from torch import nn


class NodeMLP(nn.Module):
    """MLP for node-level parameter predictions."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        """
        Args:
            in_features: Dimension of node embedding.
            hidden_features: Hidden dimension for MLP.
            out_features: Number of node-level targets to predict.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node embeddings of shape [num_nodes, in_features].

        Returns:
            Predictions of shape [num_nodes, out_features].
        """
        return self.net(x)


class EdgeMLP(nn.Module):
    """MLP for edge-level parameter predictions."""

    def __init__(
        self,
        node_in_features: int,
        edge_in_features: int,
        hidden_features: int,
        out_features: int,
    ):
        """
        Args:
            node_in_features: Dimension of node embedding.
            edge_in_features: Dimension of raw/processed edge attributes.
            hidden_features: Hidden dimension for MLP.
            out_features: Number of edge-level targets to predict.
        """
        super().__init__()
        in_dim = (2 * node_in_features) + edge_in_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Linear(hidden_features // 2, out_features),
        )

    def forward(self, x_src: torch.Tensor, x_dst: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_src: Source node embeddings of shape [num_edges, node_in_features].
            x_dst: Destination node embeddings of shape [num_edges, node_in_features].
            edge_attr: Edge features of shape [num_edges, edge_in_features].

        Returns:
            Predictions of shape [num_edges, out_features].
        """
        edge_input = torch.cat([x_src, x_dst, edge_attr], dim=-1)
        return self.net(edge_input)
