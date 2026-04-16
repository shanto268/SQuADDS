"""Universal geometry/graph ML stack for quantum chip design.

This subpackage contains the newer geometry-first workflow that:

1. Builds Shapely geometries from design parameters
2. Computes deterministic component and edge embeddings
3. Builds heterogeneous ``torch_geometric`` graphs with a virtual hub node
4. Trains downstream graph models for Hamiltonian prediction

Most symbols are lazy-loaded so importing :mod:`squadds.ml.universal` does not
eagerly import every optional dependency up front.
"""

from __future__ import annotations

import importlib

_LAZY = {
    # geometry
    "PlacedComponent": "squadds.ml.universal.geometry.composite",
    "build_composite_layout": "squadds.ml.universal.geometry.composite",
    "build_layout": "squadds.ml.universal.geometry.layout",
    "plot_component": "squadds.ml.universal.geometry.viz",
    "plot_layout": "squadds.ml.universal.geometry.viz",
    # features
    "compute_static_embedding": "squadds.ml.universal.features.node_encoder",
    "static_embedding_dim": "squadds.ml.universal.features.node_encoder",
    "get_polygon_for_component": "squadds.ml.universal.features.node_encoder",
    "EmbeddingVersion": "squadds.ml.universal.features.protocol",
    "EmbeddingMode": "squadds.ml.universal.features.protocol",
    "EmbeddingConfig": "squadds.ml.universal.features.protocol",
    "DEFAULT_EMBEDDING_CONFIG": "squadds.ml.universal.features.protocol",
    "compute_component_embedding": "squadds.ml.universal.features.protocol",
    "embedding_dim": "squadds.ml.universal.features.protocol",
    "encode_params": "squadds.ml.universal.features.protocol",
    "param_feature_dim": "squadds.ml.universal.features.protocol",
    "compute_shared_frame_embedding": "squadds.ml.universal.features.arithmetic",
    "compute_shared_frame_shape_embedding": "squadds.ml.universal.features.arithmetic",
    "embedding_cosine_similarity": "squadds.ml.universal.features.arithmetic",
    "EdgeFeatureExtractor": "squadds.ml.universal.features.edge_extractor",
    "edge_feature_dim": "squadds.ml.universal.features.edge_extractor",
    # netlist / builder
    "CircuitNetlist": "squadds.ml.universal.graph.netlist",
    "ComponentSpec": "squadds.ml.universal.graph.netlist",
    "EdgeSpec": "squadds.ml.universal.graph.netlist",
    "Port": "squadds.ml.universal.graph.netlist",
    "UniversalGraphBuilder": "squadds.ml.universal.graph.builder",
    # model / trainer
    "UniversalGNN": "squadds.ml.universal.model.gat_model",
    "UniversalTrainer": "squadds.ml.universal.trainer",
    "NODE_TARGET_NAMES": "squadds.ml.universal.model.gat_model",
    "EDGE_TARGET_NAMES": "squadds.ml.universal.model.gat_model",
    "NODE_INFERENCE_READOUT": "squadds.ml.universal.model.gat_model",
    "EDGE_INFERENCE_READOUT": "squadds.ml.universal.model.gat_model",
    # workflows
    "UniversalRowSchema": "squadds.ml.universal.workflows",
    "UniversalModelDims": "squadds.ml.universal.workflows",
    "TargetScales": "squadds.ml.universal.workflows",
    "NodePrediction": "squadds.ml.universal.workflows",
    "EdgePrediction": "squadds.ml.universal.workflows",
    "PredictionSummary": "squadds.ml.universal.workflows",
    "STANDARD_SQUADDS_ROW_SCHEMA": "squadds.ml.universal.workflows",
    "STANDARD_TARGET_SCALES": "squadds.ml.universal.workflows",
    "make_standard_qubit_cavity_netlist": "squadds.ml.universal.workflows",
    "extract_layout_params": "squadds.ml.universal.workflows",
    "build_layout_from_row": "squadds.ml.universal.workflows",
    "attach_targets_from_row": "squadds.ml.universal.workflows",
    "build_graph_from_row": "squadds.ml.universal.workflows",
    "build_graph_dataset": "squadds.ml.universal.workflows",
    "infer_model_dims": "squadds.ml.universal.workflows",
    "build_model_from_graph": "squadds.ml.universal.workflows",
    "read_node_predictions": "squadds.ml.universal.workflows",
    "read_edge_predictions": "squadds.ml.universal.workflows",
    "read_prediction_summary": "squadds.ml.universal.workflows",
    # visualization
    "NeighborResult": "squadds.ml.universal.visualization",
    "DifferenceMatch": "squadds.ml.universal.visualization",
    "compute_embedding_projection": "squadds.ml.universal.visualization",
    "compute_embedding_projections": "squadds.ml.universal.visualization",
    "compute_cosine_similarity_matrix": "squadds.ml.universal.visualization",
    "compute_label_centroids": "squadds.ml.universal.visualization",
    "find_nearest_neighbors": "squadds.ml.universal.visualization",
    "rank_difference_vector": "squadds.ml.universal.visualization",
    "plot_embedding_projection": "squadds.ml.universal.visualization",
    "plot_projection_grid": "squadds.ml.universal.visualization",
    "plot_similarity_bars": "squadds.ml.universal.visualization",
}


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_LAZY)
