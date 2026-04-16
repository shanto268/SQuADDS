"""
Machine Learning analysis tools for SQuADDS.

Includes Explainable Boosting Machines (EBM), Symbolic Regression (PySR),
Graph Neural Network forward models (requires ``pip install SQuADDS[graph]``),
and the universal geometry/graph stack.
"""


def __getattr__(name):
    """Lazy-load optional ML subpackages to keep base imports lightweight."""
    _graph_symbols = {
        "GraphForwardModel",
        "GraphTrainer",
        "ComponentFeaturizer",
        "CircuitGraphBuilder",
        "SQuADDSGraphDataset",
        "build_vocab",
    }
    _universal_symbols = {
        "PlacedComponent",
        "build_layout",
        "build_composite_layout",
        "plot_component",
        "plot_layout",
        "compute_static_embedding",
        "static_embedding_dim",
        "get_polygon_for_component",
        "EmbeddingVersion",
        "EmbeddingMode",
        "EmbeddingConfig",
        "DEFAULT_EMBEDDING_CONFIG",
        "compute_component_embedding",
        "embedding_dim",
        "encode_params",
        "param_feature_dim",
        "compute_shared_frame_embedding",
        "compute_shared_frame_shape_embedding",
        "embedding_cosine_similarity",
        "EdgeFeatureExtractor",
        "edge_feature_dim",
        "CircuitNetlist",
        "ComponentSpec",
        "EdgeSpec",
        "Port",
        "UniversalGraphBuilder",
        "UniversalGNN",
        "UniversalTrainer",
        "NODE_TARGET_NAMES",
        "EDGE_TARGET_NAMES",
        "NODE_INFERENCE_READOUT",
        "EDGE_INFERENCE_READOUT",
        "UniversalRowSchema",
        "UniversalModelDims",
        "TargetScales",
        "NodePrediction",
        "EdgePrediction",
        "PredictionSummary",
        "STANDARD_SQUADDS_ROW_SCHEMA",
        "STANDARD_TARGET_SCALES",
        "make_standard_qubit_cavity_netlist",
        "extract_layout_params",
        "build_layout_from_row",
        "attach_targets_from_row",
        "build_graph_from_row",
        "build_graph_dataset",
        "infer_model_dims",
        "build_model_from_graph",
        "read_node_predictions",
        "read_edge_predictions",
        "read_prediction_summary",
        "NeighborResult",
        "DifferenceMatch",
        "compute_embedding_projection",
        "compute_embedding_projections",
        "compute_cosine_similarity_matrix",
        "compute_label_centroids",
        "find_nearest_neighbors",
        "rank_difference_vector",
        "plot_embedding_projection",
        "plot_projection_grid",
        "plot_similarity_bars",
    }
    if name in _graph_symbols:
        from squadds.ml.graph import __getattr__ as _graph_getattr

        return _graph_getattr(name)
    if name in _universal_symbols:
        from squadds.ml.universal import __getattr__ as _universal_getattr

        return _universal_getattr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
