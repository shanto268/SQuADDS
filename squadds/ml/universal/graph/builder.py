"""Graph builder producing PyG HeteroData objects.

Schema:
    Node types: 'component', 'virtual'
    Edge types:
        ('component', 'physical', 'component')
        ('component', 'spatial_in', 'virtual')
        ('virtual', 'spatial_out', 'component')
"""

from __future__ import annotations

import hashlib
import json
import os

import torch
from torch_geometric.data import HeteroData

from squadds.ml.universal.features.edge_extractor import EdgeFeatureExtractor
from squadds.ml.universal.features.node_encoder import (
    DEFAULT_SHAPE_RESOLUTION,
    compute_static_embedding,
    get_polygon_for_component,
)
from squadds.ml.universal.graph.netlist import CircuitNetlist
from squadds.ml.universal.graph.virtual_hub import compute_hub_embedding, compute_spatial_edge_features
from squadds.ml.universal.model.gat_model import NUM_NODE_TARGETS


class UniversalGraphBuilder:
    """Builds PyG HeteroData objects from layouts and netlists.

    Produces typed nodes and edges following the heterogeneous graph schema.
    Supports disk caching keyed by design parameter hash.
    """

    def __init__(
        self,
        shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
        cache_dir: str | None = None,
    ):
        self.shape_resolution = shape_resolution
        self.edge_extractor = EdgeFeatureExtractor(shape_resolution=shape_resolution)
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, layout: dict, netlist: CircuitNetlist) -> str:
        params = layout.get("design_params", {})
        netlist_sig = json.dumps(
            {
                "components": [(c.name, c.component_type) for c in netlist.components],
                "edges": [(e.src, e.dst, e.coupling_type) for e in netlist.edges],
            },
            sort_keys=True,
        )
        combined = json.dumps(params, sort_keys=True) + netlist_sig
        return hashlib.md5(combined.encode()).hexdigest()

    def _try_load_cache(self, key: str) -> HeteroData | None:
        if not self.cache_dir:
            return None
        path = os.path.join(self.cache_dir, f"{key}.pt")
        if os.path.exists(path):
            return torch.load(path, weights_only=False)
        return None

    def _save_cache(self, key: str, data: HeteroData) -> None:
        if not self.cache_dir:
            return
        path = os.path.join(self.cache_dir, f"{key}.pt")
        torch.save(data, path)

    def build(
        self,
        layout: dict,
        netlist: CircuitNetlist,
        global_features: dict[str, float] | None = None,
    ) -> HeteroData:
        """Construct a HeteroData object from layout + netlist.

        Args:
            layout: Output from build_layout() or build_composite_layout().
            netlist: CircuitNetlist defining components and edges.
            global_features: Layer-stack info (dielectric_constant, etc.).

        Returns:
            PyG HeteroData with typed nodes and edges.
        """
        netlist.validate()

        cache_key = self._cache_key(layout, netlist)
        cached = self._try_load_cache(cache_key)
        if cached is not None:
            return cached

        R = self.shape_resolution
        data = HeteroData()

        # ── Component nodes ───────────────────────────────────────────
        node_features = []
        component_polygons = []
        component_types = []

        for comp_spec in netlist.components:
            comp_data = layout.get(comp_spec.name)
            if not comp_data:
                raise ValueError(f"Component '{comp_spec.name}' not in layout.")

            polygon = get_polygon_for_component(comp_data)
            params = comp_data.get("params", {})
            embedding = compute_static_embedding(polygon, params=params, shape_resolution=R)

            node_features.append(torch.from_numpy(embedding))
            component_polygons.append(polygon)
            component_types.append(comp_spec.component_type)

        data["component"].x = torch.stack(node_features, dim=0)

        # Store component types and names as metadata
        data["component"].component_type = component_types
        data["component"].component_name = [c.name for c in netlist.components]

        # ── Node targets: ALL nodes get ALL 5 targets ─────────────────
        # No NaN masking during training — the GNN learns which design
        # parameters affect which Hamiltonian targets through message passing.
        # Targets are filled by the dataset builder (tutorial).
        n_comp = len(netlist.components)
        data["component"].y = torch.zeros(n_comp, NUM_NODE_TARGETS)

        # Metadata: component types and names for inference readout
        from squadds.ml.universal.model.gat_model import INFERENCE_READOUT

        data["component"].inference_readout = [INFERENCE_READOUT.get(ct, []) for ct in component_types]

        # ── Physical edges (component <-> component) ──────────────────
        comp_name_to_idx = {comp.name: i for i, comp in enumerate(netlist.components)}

        if netlist.edges:
            phys_src, phys_dst = [], []
            phys_features = []

            for edge_spec in netlist.edges:
                src_name = edge_spec.src.split(".")[0]
                dst_name = edge_spec.dst.split(".")[0]
                src_idx = comp_name_to_idx[src_name]
                dst_idx = comp_name_to_idx[dst_name]

                poly_a = component_polygons[src_idx]
                poly_b = component_polygons[dst_idx]
                feat = self.edge_extractor.extract(poly_a, poly_b, coupling_type=edge_spec.coupling_type)
                feat_tensor = torch.from_numpy(feat)

                # Undirected: both directions
                phys_src.extend([src_idx, dst_idx])
                phys_dst.extend([dst_idx, src_idx])
                phys_features.extend([feat_tensor, feat_tensor])

            data["component", "physical", "component"].edge_index = torch.tensor([phys_src, phys_dst], dtype=torch.long)
            data["component", "physical", "component"].edge_attr = torch.stack(phys_features)
        else:
            data["component", "physical", "component"].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["component", "physical", "component"].edge_attr = torch.empty(
                (0, self.edge_extractor.dim), dtype=torch.float
            )

        # ── Virtual node ──────────────────────────────────────────────
        hub_embedding = compute_hub_embedding(component_polygons, layout.get("design_params", {}), global_features, R)
        data["virtual"].x = torch.from_numpy(hub_embedding).unsqueeze(0)

        # ── Spatial edges (component <-> virtual) ─────────────────────
        n_comp = len(netlist.components)
        spatial_features = compute_spatial_edge_features(component_polygons, R)

        # component -> virtual
        spat_src_in = torch.arange(n_comp, dtype=torch.long)
        spat_dst_in = torch.zeros(n_comp, dtype=torch.long)
        data["component", "spatial_in", "virtual"].edge_index = torch.stack([spat_src_in, spat_dst_in])
        data["component", "spatial_in", "virtual"].edge_attr = spatial_features

        # virtual -> component
        spat_src_out = torch.zeros(n_comp, dtype=torch.long)
        spat_dst_out = torch.arange(n_comp, dtype=torch.long)
        data["virtual", "spatial_out", "component"].edge_index = torch.stack([spat_src_out, spat_dst_out])
        data["virtual", "spatial_out", "component"].edge_attr = spatial_features.clone()

        self._save_cache(cache_key, data)
        return data
