"""Versioned, mode-aware embedding protocol for universal component embeddings."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from squadds.ml.universal.features.arithmetic import compute_shared_frame_shape_embedding
from squadds.ml.universal.features.moments import MOMENT_DIM, compute_moments
from squadds.ml.universal.features.node_encoder import DEFAULT_SHAPE_RESOLUTION, compute_static_embedding
from squadds.ml.universal.features.rasterizer import rasterize_fast


class EmbeddingVersion(str, Enum):
    """Version identifier for deterministic embedding algorithms."""

    V1_LEGACY = "v1_legacy"
    V2_HASHED_PARAMS = "v2_hashed_params"


class EmbeddingMode(str, Enum):
    """Coordinate-frame mode for component embeddings."""

    GEOMETRY_ONLY = "geometry_only"
    COMPONENT_CONTEXT = "component_context"
    DEVICE_CONTEXT = "device_context"


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the deterministic universal embedding protocol."""

    version: EmbeddingVersion = EmbeddingVersion.V1_LEGACY
    mode: EmbeddingMode = EmbeddingMode.GEOMETRY_ONLY
    shape_resolution: int = DEFAULT_SHAPE_RESOLUTION
    shared_bounds_padding: float = 0.0
    shared_bounds_padding_fraction: float = 0.05
    param_hash_dim: int = 8

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-serializable representation for caching/metadata."""

        return {
            key: (value.value if isinstance(value, Enum) else value)
            for key, value in asdict(self).items()
        }


DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()


def param_feature_dim(config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG) -> int:
    """Return the deterministic parameter-feature dimension for the config."""

    if config.version == EmbeddingVersion.V1_LEGACY:
        return 1
    if config.version == EmbeddingVersion.V2_HASHED_PARAMS:
        return 8 + config.param_hash_dim
    raise ValueError(f"Unsupported embedding version: {config.version}")


def embedding_dim(config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG) -> int:
    """Return the total embedding dimension for the config."""

    return param_feature_dim(config) + MOMENT_DIM + config.shape_resolution * config.shape_resolution


def _stable_bucket(name: str, num_buckets: int) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little") % num_buckets


def encode_params(
    params: dict[str, float] | None = None,
    *,
    config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
) -> np.ndarray:
    """Encode parameter dictionaries according to the configured version."""

    if config.version == EmbeddingVersion.V1_LEGACY:
        param_sum = np.float32(sum(params.values())) if params else np.float32(0.0)
        return np.array([param_sum], dtype=np.float32)

    if config.version != EmbeddingVersion.V2_HASHED_PARAMS:
        raise ValueError(f"Unsupported embedding version: {config.version}")

    if not params:
        return np.zeros(param_feature_dim(config), dtype=np.float32)

    values = np.array([float(value) for value in params.values()], dtype=np.float32)
    stats = np.array(
        [
            float(len(values)),
            float(values.sum()),
            float(values.mean()),
            float(values.std()),
            float(values.min()),
            float(values.max()),
            float(np.abs(values).sum()),
            float(np.linalg.norm(values)),
        ],
        dtype=np.float32,
    )

    bins = np.zeros(config.param_hash_dim, dtype=np.float32)
    for name, value in sorted(params.items()):
        bins[_stable_bucket(name, config.param_hash_dim)] += np.float32(value)

    return np.concatenate([stats, bins]).astype(np.float32)


def _compute_shape_features(
    polygon: Polygon | MultiPolygon,
    *,
    config: EmbeddingConfig,
    reference_polygons: list[Polygon | MultiPolygon] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    if config.mode == EmbeddingMode.GEOMETRY_ONLY:
        return rasterize_fast(polygon, resolution=config.shape_resolution).flatten().astype(np.float32)

    if bounds is None and reference_polygons is None:
        raise ValueError(
            f"Embedding mode {config.mode.value!r} requires reference_polygons or bounds for shared-frame rasterization."
        )

    return compute_shared_frame_shape_embedding(
        polygon,
        reference_polygons=reference_polygons,
        bounds=bounds,
        shape_resolution=config.shape_resolution,
        padding=config.shared_bounds_padding,
        padding_fraction=config.shared_bounds_padding_fraction,
    )


def compute_component_embedding(
    polygon: Polygon | MultiPolygon,
    *,
    params: dict[str, float] | None = None,
    config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
    reference_polygons: list[Polygon | MultiPolygon] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Compute a deterministic, versioned component embedding."""

    if config.version == EmbeddingVersion.V1_LEGACY and config.mode == EmbeddingMode.GEOMETRY_ONLY:
        return compute_static_embedding(
            polygon,
            params=params,
            shape_resolution=config.shape_resolution,
        )

    param_features = encode_params(params, config=config)
    moments = compute_moments(polygon)
    shape_features = _compute_shape_features(
        polygon,
        config=config,
        reference_polygons=reference_polygons,
        bounds=bounds,
    )

    vector = np.concatenate([param_features, moments, shape_features]).astype(np.float32)
    if vector.shape != (embedding_dim(config),):
        raise ValueError(
            f"Embedding dimension mismatch for config {config.to_metadata()}: "
            f"expected {embedding_dim(config)}, got {vector.shape[0]}."
        )
    return vector


__all__ = [
    "DEFAULT_EMBEDDING_CONFIG",
    "EmbeddingConfig",
    "EmbeddingMode",
    "EmbeddingVersion",
    "compute_component_embedding",
    "embedding_dim",
    "encode_params",
    "param_feature_dim",
]
