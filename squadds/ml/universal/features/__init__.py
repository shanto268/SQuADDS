"""Feature extraction modules (moments, rasterizer, encoders, arithmetic)."""

from squadds.ml.universal.features.arithmetic import (
    compute_shared_frame_embedding,
    compute_shared_frame_shape_embedding,
    embedding_cosine_similarity,
)
from squadds.ml.universal.features.cnn_encoder import CNNEncoder
from squadds.ml.universal.features.deepsets import DeepSetsEncoder
from squadds.ml.universal.features.edge_extractor import EdgeFeatureExtractor, edge_feature_dim, extract_edge_features
from squadds.ml.universal.features.moments import compute_moments, moment_names
from squadds.ml.universal.features.node_encoder import (
    DEFAULT_SHAPE_RESOLUTION,
    compute_static_embedding,
    get_polygon_for_component,
    static_embedding_dim,
)
from squadds.ml.universal.features.protocol import (
    DEFAULT_EMBEDDING_CONFIG,
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingVersion,
    compute_component_embedding,
    embedding_dim,
    encode_params,
    param_feature_dim,
)
from squadds.ml.universal.features.rasterizer import compute_shared_bounds, rasterize_fast, rasterize_in_bounds

__all__ = [
    "CNNEncoder",
    "DEFAULT_SHAPE_RESOLUTION",
    "DEFAULT_EMBEDDING_CONFIG",
    "DeepSetsEncoder",
    "EdgeFeatureExtractor",
    "EmbeddingConfig",
    "EmbeddingMode",
    "EmbeddingVersion",
    "compute_moments",
    "compute_component_embedding",
    "compute_shared_bounds",
    "compute_shared_frame_embedding",
    "compute_shared_frame_shape_embedding",
    "compute_static_embedding",
    "embedding_dim",
    "edge_feature_dim",
    "encode_params",
    "embedding_cosine_similarity",
    "extract_edge_features",
    "get_polygon_for_component",
    "moment_names",
    "param_feature_dim",
    "rasterize_fast",
    "rasterize_in_bounds",
    "static_embedding_dim",
]
