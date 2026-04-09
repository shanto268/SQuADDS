"""Virtual hub injector for graph construction."""

import torch
from torch_geometric.data import Data


class VirtualHubInjector:
    """Inject a virtual node connected to all real nodes to provide global context.

    The virtual node features capture macroscopic chip-level parameters
    (e.g., layer stack, total chip area, n_components).
    The spatial edges connecting the hub to real nodes capture placement features.
    """

    def __init__(self, edge_dim: int):
        """
        Args:
            edge_dim: Dimension of the edge features.
        """
        self.edge_dim = edge_dim

    def inject(self, data: Data, global_node_features: torch.Tensor) -> Data:
        """Inject the virtual hub node into the PyG Data object.

        Args:
            data: The graph before hub injection.
            global_node_features: Tensor of shape (1, node_dim) containing the features
                for the virtual hub node.

        Returns:
            A new Data object with N+1 nodes and spatial edges added to `edge_index`
            and `edge_attr`.
        """
        num_real_nodes = data.x.size(0)
        hub_idx = num_real_nodes

        # Append hub node features
        new_x = torch.cat([data.x, global_node_features], dim=0)

        # Create spatial edges: hub <-> all real nodes
        # Edge index: pairs of (hub_idx, i) and (i, hub_idx)
        hub_src = torch.full((num_real_nodes,), hub_idx, dtype=torch.long)
        real_dst = torch.arange(num_real_nodes, dtype=torch.long)

        # Hub to real nodes
        edges_out = torch.stack([hub_src, real_dst], dim=0)
        # Real nodes to hub
        edges_in = torch.stack([real_dst, hub_src], dim=0)

        new_edge_index = torch.cat([data.edge_index, edges_out, edges_in], dim=1)

        # Create edge features for the spatial edges.
        # For simplicity, we initialize them with zeros. They can later be updated
        # with position-based/CNN features.
        spatial_edge_features = torch.zeros(
            (2 * num_real_nodes, self.edge_dim), dtype=data.edge_attr.dtype, device=data.edge_attr.device
        )
        new_edge_attr = torch.cat([data.edge_attr, spatial_edge_features], dim=0)

        # Handle targets (y) if they exist
        new_y = data.y
        if data.y is not None:
            # Append NaN targets for the hub node
            num_targets = data.y.size(1)
            hub_targets = torch.full((1, num_targets), float("nan"))
            new_y = torch.cat([data.y, hub_targets], dim=0)

        # Handle edge targets (y_edge) if they exist
        new_y_edge = data.y_edge
        if data.y_edge is not None:
            # Append NaN targets for spatial edges
            num_edge_targets = data.y_edge.size(1)
            spatial_edge_targets = torch.full((2 * num_real_nodes, num_edge_targets), float("nan"))
            new_y_edge = torch.cat([data.y_edge, spatial_edge_targets], dim=0)

        new_data = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr, y=new_y)
        new_data.y_edge = new_y_edge

        return new_data
