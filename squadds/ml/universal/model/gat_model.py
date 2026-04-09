"""GATv2 Graph Neural Network for Universal Graph Pipeline.

Uses an edge projection layer to compress high-dimensional edge features
(containing shape tensors) down to a manageable size for the attention mechanism.
"""

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from squadds.ml.universal.model.prediction_heads import EdgeMLP, NodeMLP


class UniversalGNN(nn.Module):
    """GATv2 model with edge projection and virtual node support.

    Predicts both node-level and edge-level Hamiltonian parameters.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        node_targets: int = 5,
        edge_targets: int = 5,
        edge_hidden: int = 32,
    ):
        """
        Args:
            node_dim: Dimension of input node features (static embedding).
            edge_dim: Dimension of raw edge features (may be large due to shape tensors).
            hidden_dim: Hidden dimension for GAT and MLPs.
            num_layers: Number of GATv2 layers.
            num_heads: Number of attention heads.
            node_targets: Number of targets to predict per node.
            edge_targets: Number of targets to predict per edge.
            edge_hidden: Compressed edge dimension for the attention mechanism.
        """
        super().__init__()
        self.node_embed = nn.Linear(node_dim, hidden_dim)

        # Project high-dim edge features to a compact representation
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden),
            nn.ReLU(inplace=True),
        )

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
                    edge_dim=edge_hidden,
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
            edge_in_features=edge_hidden,
            hidden_features=hidden_dim,
            out_features=edge_targets,
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: PyG Data object with x, edge_index, edge_attr.

        Returns:
            Tuple of (node_preds, edge_preds)
                node_preds: [num_nodes, node_targets]
                edge_preds: [num_edges, edge_targets]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Initial embeddings
        x = self.node_embed(x)
        edge_attr_proj = self.edge_proj(edge_attr)

        # Message passing layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr_proj)
            x = torch.relu(x)

        # Prediction heads
        node_preds = self.node_mlp(x)

        src, dst = edge_index
        edge_preds = self.edge_mlp(x[src], x[dst], edge_attr_proj)

        return node_preds, edge_preds
