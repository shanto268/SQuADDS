"""GATv2 Graph Neural Network for Universal Graph Pipeline."""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from squadds.ml.universal.model.prediction_heads import EdgeMLP, NodeMLP


class UniversalGNN(nn.Module):
    """GATv2 model with virtual node support and edge features.

    Predicts both node-level and edge-level Hamilton parameters.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        node_targets: int = 3,
        edge_targets: int = 2,
    ):
        """
        Args:
            node_dim: Dimension of input node features.
            edge_dim: Dimension of input edge features.
            hidden_dim: Hidden dimension for GAT and MLPs.
            num_layers: Number of GATv2 layers.
            num_heads: Number of attention heads.
            node_targets: Number of targets to predict per node.
            edge_targets: Number of targets to predict per edge.
        """
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            concat = True if i < num_layers - 1 else False
            out_dim = hidden_dim // num_heads if concat else hidden_dim

            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=out_dim,
                    heads=num_heads,
                    concat=concat,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                )
            )

        self.node_mlp = NodeMLP(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=node_targets,
        )

        self.edge_mlp = EdgeMLP(
            node_in_features=hidden_dim,
            edge_in_features=edge_dim,
            hidden_features=hidden_dim,
            out_features=edge_targets,
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: PyG Data object containing x, edge_index, right edge_attr.

        Returns:
            Tuple of (node_preds, edge_preds)
                node_preds: [num_nodes, node_targets]
                edge_preds: [num_edges, edge_targets]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Initial node embedding
        x = self.node_embed(x)

        # Message passing layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = torch.relu(x)

        # Prediction heads
        node_preds = self.node_mlp(x)

        src, dst = edge_index
        edge_preds = self.edge_mlp(x[src], x[dst], edge_attr)

        return node_preds, edge_preds
