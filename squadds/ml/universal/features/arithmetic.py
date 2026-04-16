"""Shared-frame embedding utilities for embedding arithmetic experiments.

The default static embedding normalizes each polygon to its own bounding box,
which is useful for shape recognition but not ideal for compositional
operations like:

    embed(qubit + claw) - embed(claw) ~= embed(qubit)

Those operations only make geometric sense when all shapes are rasterized in a
shared coordinate frame. This module provides that shared-frame path.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from squadds.ml.universal.features.moments import compute_moments
from squadds.ml.universal.features.node_encoder import DEFAULT_SHAPE_RESOLUTION, static_embedding_dim
from squadds.ml.universal.features.rasterizer import compute_shared_bounds, rasterize_in_bounds


def compute_shared_frame_shape_embedding(
    polygon: Polygon | MultiPolygon,
    *,
    reference_polygons: Iterable[Polygon | MultiPolygon] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
    padding: float = 0.0,
    padding_fraction: float = 0.0,
) -> np.ndarray:
    """Rasterize a polygon in a shared coordinate frame and flatten the mask."""

    if bounds is None:
        if reference_polygons is None:
            raise ValueError("Provide either bounds or reference_polygons for shared-frame embedding.")
        bounds = compute_shared_bounds(
            list(reference_polygons),
            padding=padding,
            padding_fraction=padding_fraction,
        )
    return rasterize_in_bounds(polygon, bounds, shape_resolution).flatten().astype(np.float32)


def compute_shared_frame_embedding(
    polygon: Polygon | MultiPolygon,
    *,
    reference_polygons: Iterable[Polygon | MultiPolygon] | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    params: dict[str, float] | None = None,
    shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
    padding: float = 0.0,
    padding_fraction: float = 0.0,
) -> np.ndarray:
    """Compute the standard static embedding using a shared-frame shape tensor."""

    param_sum = np.float32(sum(params.values())) if params else np.float32(0.0)
    moments = compute_moments(polygon)
    shape_flat = compute_shared_frame_shape_embedding(
        polygon,
        reference_polygons=reference_polygons,
        bounds=bounds,
        shape_resolution=shape_resolution,
        padding=padding,
        padding_fraction=padding_fraction,
    )
    vector = np.concatenate([[param_sum], moments, shape_flat]).astype(np.float32)
    if vector.shape != (static_embedding_dim(shape_resolution),):
        raise ValueError("Shared-frame embedding produced an unexpected dimension.")
    return vector


def embedding_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D embedding vectors."""

    a_vec = np.asarray(a, dtype=np.float32).ravel()
    b_vec = np.asarray(b, dtype=np.float32).ravel()
    if a_vec.shape != b_vec.shape:
        raise ValueError(f"Embedding shapes must match, got {a_vec.shape} and {b_vec.shape}.")

    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_vec, b_vec) / denom)


__all__ = [
    "compute_shared_frame_embedding",
    "compute_shared_frame_shape_embedding",
    "embedding_cosine_similarity",
]
