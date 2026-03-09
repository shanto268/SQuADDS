"""
Graph Neural Network forward models for SQuADDS.

Provides a graph-based architecture that treats quantum circuits as graphs,
where components are nodes and physical connections are edges. A GNN performs
message passing to produce context-aware embeddings for Hamiltonian parameter
prediction.

Requires: ``pip install SQuADDS[graph]``
"""

try:
    import spektral  # noqa: F401
    import tensorflow as tf  # noqa: F401

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def _check_deps():
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow and Spektral are required for the graph ML module. "
            "Install them with: pip install SQuADDS[graph]"
        )


# Lazy imports to avoid pulling TF on every ``import squadds``
def __getattr__(name):
    _lazy = {
        "ComponentFeaturizer": "squadds.ml.graph.featurizer",
        "CircuitGraphBuilder": "squadds.ml.graph.featurizer",
        "SQuADDSGraphDataset": "squadds.ml.graph.featurizer",
        "build_vocab": "squadds.ml.graph.featurizer",
        "LayerStackEncoder": "squadds.ml.graph.encoders",
        "GeometricEncoder": "squadds.ml.graph.encoders",
        "PortEncoder": "squadds.ml.graph.encoders",
        "NodeEncoder": "squadds.ml.graph.encoders",
        "GraphForwardModel": "squadds.ml.graph.gnn_model",
        "GraphTrainer": "squadds.ml.graph.trainer",
    }
    if name in _lazy:
        _check_deps()
        import importlib

        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
