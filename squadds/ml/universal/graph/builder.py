"""Graph builder for generating PyG Data objects."""

import torch
from torch_geometric.data import Data

from squadds.ml.universal.graph.netlist import CircuitNetlist
from squadds.ml.universal.graph.virtual_hub import VirtualHubInjector


class UniversalGraphBuilder:
    """Builds PyG Data objects from geometric layouts and netlists."""

    def __init__(
        self,
        node_encoder,
        edge_extractor,
        hub_injector: VirtualHubInjector | None = None,
    ):
        """
        Args:
            node_encoder: Extracts node features from a component dictionary.
            edge_extractor: Extracts edge features from two component geometries.
            hub_injector: Injector for the virtual global node.
        """
        self.node_encoder = node_encoder
        self.edge_extractor = edge_extractor
        self.hub_injector = hub_injector

    def build(
        self,
        layout: dict,
        netlist: CircuitNetlist,
        global_features: dict | None = None,
    ) -> Data:
        """Construct the PyG Data object.

        Args:
            layout: Output from `build_layout` containing polygons and geometries.
            netlist: CircuitNetlist defining components and edges.
            global_features: Global design parameters for the hub node.

        Returns:
            Data object ready for PyG processing.
        """
        netlist.validate()

        # Build node features
        node_features = []
        node_targets = []
        for comp_spec in netlist.components:
            comp_name = comp_spec.name
            comp_data = layout.get(comp_name)
            if not comp_data:
                raise ValueError(f"Component '{comp_name}' found in netlist but not in layout.")

            # Assume node_encoder extracts the tensor representation of a node
            n_feat = self.node_encoder(comp_name, comp_data)
            node_features.append(n_feat)

            # Gather target values if any (using NaN as placeholder for now, since
            # actual target injection is typically done at the dataset level)
            targets = torch.full((3,), float("nan"))
            node_targets.append(targets)

        x = torch.stack(node_features, dim=0)
        y = torch.stack(node_targets, dim=0)

        # Build edge indices and features
        edge_index = netlist.to_pyg_edge_index()

        edge_features = []
        edge_targets = []

        {comp.name: i for i, comp in enumerate(netlist.components)}

        if edge_index.size(1) > 0:
            # We iterate over the netlist edges
            for edge_spec in netlist.edges:
                src_comp = edge_spec.src.split(".")[0]
                dst_comp = edge_spec.dst.split(".")[0]

                poly_a = layout[src_comp].get("trace") or layout[src_comp].get("cross") or layout[src_comp].get("arm")
                poly_b = layout[dst_comp].get("trace") or layout[dst_comp].get("cross") or layout[dst_comp].get("arm")

                e_feat_dict = self.edge_extractor.extract(poly_a, poly_b)

                # Assume edge_extractor returns a dict of scalar features that we concatenate
                # For example: shortest_gap, overlap_length, metal_area, void_area
                feat_vec = torch.tensor(
                    [
                        e_feat_dict.get("shortest_gap", 0.0),
                        e_feat_dict.get("overlap_length", 0.0),
                        e_feat_dict.get("metal_area", 0.0),
                        e_feat_dict.get("void_area", 0.0),
                    ],
                    dtype=torch.float,
                )

                edge_features.append(feat_vec)
                edge_features.append(feat_vec)  # Append twice for undirected (src->dst and dst->src)

                targets = torch.full((2,), float("nan"))
                edge_targets.append(targets)
                edge_targets.append(targets)

            edge_attr = torch.stack(edge_features, dim=0)
            y_edge = torch.stack(edge_targets, dim=0)
        else:
            edge_attr = torch.empty((0, 4), dtype=torch.float)
            y_edge = torch.empty((0, 2), dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.y_edge = y_edge

        if self.hub_injector is not None:
            # Handle global features for virtual hub injection
            if global_features is None:
                global_features = {}
            # Assume global hub vector is constructed here.
            # It needs to match node feature dimension.
            node_dim = x.size(1)
            hub_feat = torch.zeros((1, node_dim), dtype=torch.float)
            data = self.hub_injector.inject(data, hub_feat)

        return data
