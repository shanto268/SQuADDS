"""Graph builder for generating PyG Data objects from layouts and netlists.

Orchestrates: layout → static embeddings → edge features → virtual hub → PyG Data.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch
from torch_geometric.data import Data

from squadds.ml.universal.features.edge_extractor import EdgeFeatureExtractor
from squadds.ml.universal.features.node_encoder import (
    DEFAULT_SHAPE_RESOLUTION,
    compute_static_embedding,
    get_polygon_for_component,
)
from squadds.ml.universal.graph.netlist import CircuitNetlist
from squadds.ml.universal.graph.virtual_hub import VirtualHubInjector


class UniversalGraphBuilder:
    """Builds PyG Data objects from geometric layouts and netlists.

    Uses static embeddings (param_sum + moments + shape_tensor) for nodes,
    rich geometric features for edges, and a virtual hub node for global context.
    """

    def __init__(
        self,
        shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
        cache_dir: str | None = None,
    ):
        """
        Args:
            shape_resolution: Resolution for all shape tensors (nodes, edges, hub).
                Change this single value to scale up for production.
            cache_dir: If set, cache constructed graphs to disk keyed by
                a hash of the design parameters.  Avoids redundant recomputation.
        """
        self.shape_resolution = shape_resolution
        self.edge_extractor = EdgeFeatureExtractor(shape_resolution=shape_resolution)
        self.hub_injector = VirtualHubInjector(shape_resolution=shape_resolution)
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, layout: dict, netlist: CircuitNetlist) -> str:
        """Compute a deterministic hash for caching."""
        # Hash the design params + netlist structure
        params = layout.get("design_params", {})
        netlist_sig = json.dumps(
            {
                "components": [c.name for c in netlist.components],
                "edges": [(e.src, e.dst, e.coupling_type) for e in netlist.edges],
            },
            sort_keys=True,
        )
        combined = json.dumps(params, sort_keys=True) + netlist_sig
        return hashlib.md5(combined.encode()).hexdigest()

    def _try_load_cache(self, key: str) -> Data | None:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, f"{key}.pt")
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        return None

    def _save_cache(self, key: str, data: Data) -> None:
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, f"{key}.pt")
        torch.save(data, path)

    def build(
        self,
        layout: dict,
        netlist: CircuitNetlist,
        global_features: dict[str, float] | None = None,
    ) -> Data:
        """Construct the PyG Data object from layout + netlist.

        Args:
            layout: Output from ``build_layout`` or ``build_composite_layout``.
            netlist: CircuitNetlist defining components and edges.
            global_features: Global info (dielectric_constant, etc.) for hub node.

        Returns:
            PyG Data object with static embeddings, edge features, and virtual hub.
        """
        netlist.validate()

        # ── Check cache ───────────────────────────────────────────────
        cache_key = self._cache_key(layout, netlist)
        cached = self._try_load_cache(cache_key)
        if cached is not None:
            return cached

        R = self.shape_resolution

        # ── Node features ─────────────────────────────────────────────
        node_features = []
        component_polygons = []

        for comp_spec in netlist.components:
            comp_name = comp_spec.name
            comp_data = layout.get(comp_name)
            if not comp_data:
                raise ValueError(f"Component '{comp_name}' found in netlist but not in layout.")

            polygon = get_polygon_for_component(comp_data)
            params = comp_data.get("params", {})

            embedding = compute_static_embedding(polygon, params=params, shape_resolution=R)
            node_features.append(torch.from_numpy(embedding))
            component_polygons.append(polygon)

        x = torch.stack(node_features, dim=0)  # (N, embed_dim)

        # ── Edge features ─────────────────────────────────────────────
        edge_index = netlist.to_pyg_edge_index()
        comp_name_to_idx = {comp.name: i for i, comp in enumerate(netlist.components)}

        edge_features = []
        if edge_index.size(1) > 0:
            for edge_spec in netlist.edges:
                src_comp = edge_spec.src.split(".")[0]
                dst_comp = edge_spec.dst.split(".")[0]

                poly_a = component_polygons[comp_name_to_idx[src_comp]]
                poly_b = component_polygons[comp_name_to_idx[dst_comp]]

                feat = self.edge_extractor.extract(poly_a, poly_b, coupling_type=edge_spec.coupling_type)
                feat_tensor = torch.from_numpy(feat)

                # Undirected: same features for both directions
                edge_features.append(feat_tensor)
                edge_features.append(feat_tensor)

            edge_attr = torch.stack(edge_features, dim=0)
        else:
            edge_attr = torch.empty((0, self.edge_extractor.dim), dtype=torch.float)

        # ── Placeholder targets (filled by dataset builder) ───────────
        y = torch.full((x.size(0), 5), float("nan"))  # 5 Hamiltonian targets
        y_edge = torch.full((edge_attr.size(0), 5), float("nan"))

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.y_edge = y_edge

        # ── Virtual hub node ──────────────────────────────────────────
        layout_params = layout.get("design_params", {})
        data = self.hub_injector.inject(
            data,
            component_polygons=component_polygons,
            layout_params=layout_params,
            global_info=global_features,
        )

        # ── Cache ─────────────────────────────────────────────────────
        self._save_cache(cache_key, data)

        return data
