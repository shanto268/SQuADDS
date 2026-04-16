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
from squadds.ml.universal.features.node_encoder import DEFAULT_SHAPE_RESOLUTION, get_polygon_for_component
from squadds.ml.universal.features.protocol import (
    DEFAULT_EMBEDDING_CONFIG,
    EmbeddingConfig,
    EmbeddingMode,
    compute_component_embedding,
)
from squadds.ml.universal.graph.netlist import CircuitNetlist
from squadds.ml.universal.graph.virtual_hub import compute_hub_embedding, compute_spatial_edge_features
from squadds.ml.universal.model.gat_model import NUM_EDGE_TARGETS, NUM_NODE_TARGETS


class UniversalGraphBuilder:
    """Builds PyG HeteroData objects from layouts and netlists.

    Produces typed nodes and edges following the heterogeneous graph schema.
    Supports disk caching keyed by design parameter hash.
    """

    def __init__(
        self,
        shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
        cache_dir: str | None = None,
        embedding_config: EmbeddingConfig | None = None,
    ):
        if embedding_config is None:
            embedding_config = EmbeddingConfig(
                version=DEFAULT_EMBEDDING_CONFIG.version,
                mode=DEFAULT_EMBEDDING_CONFIG.mode,
                shape_resolution=shape_resolution,
                shared_bounds_padding=DEFAULT_EMBEDDING_CONFIG.shared_bounds_padding,
                shared_bounds_padding_fraction=DEFAULT_EMBEDDING_CONFIG.shared_bounds_padding_fraction,
                param_hash_dim=DEFAULT_EMBEDDING_CONFIG.param_hash_dim,
            )
        elif embedding_config.shape_resolution != shape_resolution:
            raise ValueError(
                "shape_resolution must match embedding_config.shape_resolution "
                f"(got {shape_resolution} and {embedding_config.shape_resolution})."
            )

        self.embedding_config = embedding_config
        self.shape_resolution = embedding_config.shape_resolution
        self.edge_extractor = EdgeFeatureExtractor(shape_resolution=self.shape_resolution)
        self.cache_dir = cache_dir
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, layout: dict, netlist: CircuitNetlist) -> str:
        params = layout.get("design_params", {})
        netlist_sig = json.dumps(
            {
                "components": [(c.name, c.component_type) for c in netlist.components],
                "edges": [(e.src, e.dst, e.coupling_type) for e in netlist.edges],
                "embedding_config": self.embedding_config.to_metadata(),
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
        component_polygons = []
        component_params = []
        component_types = []

        for comp_spec in netlist.components:
            comp_data = layout.get(comp_spec.name)
            if not comp_data:
                raise ValueError(f"Component '{comp_spec.name}' not in layout.")

            polygon = get_polygon_for_component(comp_data)
            params = comp_data.get("params", {})
            component_polygons.append(polygon)
            component_params.append(params)
            component_types.append(comp_spec.component_type)

        shared_reference_polygons = component_polygons if self.embedding_config.mode != EmbeddingMode.GEOMETRY_ONLY else None
        node_features = []
        for polygon, params in zip(component_polygons, component_params):
            embedding = compute_component_embedding(
                polygon,
                params=params,
                config=self.embedding_config,
                reference_polygons=shared_reference_polygons,
            )
            node_features.append(torch.from_numpy(embedding))

        data["component"].x = torch.stack(node_features, dim=0)

        # Store component types and names as metadata
        data["component"].component_type = component_types
        data["component"].component_name = [c.name for c in netlist.components]
        data["component"].embedding_config = self.embedding_config.to_metadata()

        # ── Node targets: ALL nodes get ALL 5 targets ─────────────────
        # No NaN masking during training — the GNN learns which design
        # parameters affect which Hamiltonian targets through message passing.
        # Targets are filled by the dataset builder (tutorial).
        n_comp = len(netlist.components)
        data["component"].y = torch.zeros(n_comp, NUM_NODE_TARGETS)

        # Metadata: component types and names for inference readout
        from squadds.ml.universal.model.gat_model import NODE_INFERENCE_READOUT

        data["component"].inference_readout = [NODE_INFERENCE_READOUT.get(ct, []) for ct in component_types]

        # ── Physical edges (component <-> component) ──────────────────
        comp_name_to_idx = {comp.name: i for i, comp in enumerate(netlist.components)}

        if netlist.edges:
            phys_src, phys_dst = [], []
            phys_features = []
            edge_component_types = []  # (src_type, dst_type) per edge

            for edge_spec in netlist.edges:
                src_name = edge_spec.src.split(".")[0]
                dst_name = edge_spec.dst.split(".")[0]
                src_idx = comp_name_to_idx[src_name]
                dst_idx = comp_name_to_idx[dst_name]

                poly_a = component_polygons[src_idx]
                poly_b = component_polygons[dst_idx]
                feat = self.edge_extractor.extract(poly_a, poly_b, coupling_type=edge_spec.coupling_type)
                feat_tensor = torch.from_numpy(feat)

                src_type = component_types[src_idx]
                dst_type = component_types[dst_idx]

                # Undirected: both directions
                phys_src.extend([src_idx, dst_idx])
                phys_dst.extend([dst_idx, src_idx])
                phys_features.extend([feat_tensor, feat_tensor])
                edge_component_types.extend([(src_type, dst_type), (dst_type, src_type)])

            data["component", "physical", "component"].edge_index = torch.tensor([phys_src, phys_dst], dtype=torch.long)
            data["component", "physical", "component"].edge_attr = torch.stack(phys_features)
            # ALL edges get ALL edge targets (filled by dataset builder)
            n_edges = len(phys_src)
            data["component", "physical", "component"].y = torch.zeros(n_edges, NUM_EDGE_TARGETS)
            data["component", "physical", "component"].edge_component_types = edge_component_types
        else:
            data["component", "physical", "component"].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["component", "physical", "component"].edge_attr = torch.empty(
                (0, self.edge_extractor.dim), dtype=torch.float
            )
            data["component", "physical", "component"].y = torch.empty((0, NUM_EDGE_TARGETS), dtype=torch.float)
            data["component", "physical", "component"].edge_component_types = []

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
