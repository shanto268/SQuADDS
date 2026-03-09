"""
Graph Neural Network forward models for SQuADDS.

Provides a graph-based architecture that treats quantum circuits as graphs,
where components are nodes and physical connections are edges. A GNN performs
message passing to produce context-aware embeddings for Hamiltonian parameter
prediction.

Requires: ``pip install SQuADDS[graph]``
"""

try:
    import keras  # noqa: F401
    import spektral  # noqa: F401

    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


def _check_deps():
    if not GRAPH_AVAILABLE:
        raise ImportError(
            "Keras and Spektral are required for the graph ML module. Install them with: pip install SQuADDS[graph]"
        )


# Lazy imports to avoid pulling TF/Keras on every ``import squadds``
def __getattr__(name):
    _lazy = {
        # featurizer
        "ComponentFeaturizer": "squadds.ml.graph.featurizer",
        "CircuitGraphBuilder": "squadds.ml.graph.featurizer",
        "SQuADDSGraphDataset": "squadds.ml.graph.featurizer",
        "build_vocab": "squadds.ml.graph.featurizer",
        # encoders
        "LayerStackEncoder": "squadds.ml.graph.encoders",
        "GeometricEncoder": "squadds.ml.graph.encoders",
        "PortEncoder": "squadds.ml.graph.encoders",
        "NodeEncoder": "squadds.ml.graph.encoders",
        # gnn model
        "GraphForwardModel": "squadds.ml.graph.gnn_model",
        "GCNConvK3": "squadds.ml.graph.gnn_model",
        "GraphAttentionConvK3": "squadds.ml.graph.gnn_model",
        "GlobalAttentionPoolK3": "squadds.ml.graph.gnn_model",
        "UnpackNodeFeatures": "squadds.ml.graph.gnn_model",
        # trainer
        "GraphTrainer": "squadds.ml.graph.trainer",
        "plot_predictions": "squadds.ml.graph.trainer",
    }
    if name in _lazy:
        _check_deps()
        import importlib

        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
