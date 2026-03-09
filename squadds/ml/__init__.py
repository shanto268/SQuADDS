"""
Machine Learning analysis tools for SQuADDS.

Includes Explainable Boosting Machines (EBM), Symbolic Regression (PySR),
and Graph Neural Network forward models (requires ``pip install SQuADDS[graph]``).
"""


def __getattr__(name):
    """Lazy-load graph module symbols to avoid pulling TensorFlow on import."""
    _graph_symbols = {
        "GraphForwardModel",
        "GraphTrainer",
        "ComponentFeaturizer",
        "CircuitGraphBuilder",
        "SQuADDSGraphDataset",
        "build_vocab",
    }
    if name in _graph_symbols:
        from squadds.ml.graph import __getattr__ as _graph_getattr

        return _graph_getattr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
