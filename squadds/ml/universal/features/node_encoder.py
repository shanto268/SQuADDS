"""Static node embedding: param_sum + geometric_moments + shape_tensor.

Produces a deterministic, fixed-size embedding for any Shapely polygon
with associated design parameters.  No learned weights — this is pure
feature engineering that makes the embedding universal across component types.

The embedding vector is:

    [param_sum (1)] || [moments (8)] || [shape_tensor (R*R)]

where R = ``shape_resolution`` (default 16, configurable).
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from squadds.ml.universal.features.moments import MOMENT_DIM, compute_moments
from squadds.ml.universal.features.rasterizer import rasterize_fast

# ── Default resolution for shape tensors ──────────────────────────────
# Change this single constant to scale up for production.
DEFAULT_SHAPE_RESOLUTION = 16


def compute_static_embedding(
    polygon: Polygon | MultiPolygon,
    params: dict[str, float] | None = None,
    shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
) -> np.ndarray:
    """Compute a deterministic static embedding for one component.

    Args:
        polygon: The component's Shapely polygon (metal trace).
        params: Design parameter dict (e.g. ``{"cross_length": 310, ...}``).
            Sum of values is used as a permutation-invariant scalar.
        shape_resolution: Side length of the square shape tensor.
            Total embedding dim = 1 + 8 + shape_resolution².

    Returns:
        ``np.ndarray`` of shape ``(embedding_dim,)`` with dtype ``float32``.
    """
    # 1. Permutation-invariant parameter aggregate
    param_sum = np.float32(sum(params.values())) if params else np.float32(0.0)

    # 2. Geometric moments (8 scalars)
    moments = compute_moments(polygon)  # (8,)

    # 3. Scale-invariant shape tensor
    #    The rasterizer already normalizes to the bbox, so this captures
    #    ONLY shape (size info is in the moments).
    shape_tensor = rasterize_fast(polygon, resolution=shape_resolution)  # (R, R)
    shape_flat = shape_tensor.flatten()  # (R*R,)

    return np.concatenate([[param_sum], moments, shape_flat]).astype(np.float32)


def static_embedding_dim(shape_resolution: int = DEFAULT_SHAPE_RESOLUTION) -> int:
    """Return the total dimension of the static embedding vector."""
    return 1 + MOMENT_DIM + shape_resolution * shape_resolution


def get_polygon_for_component(comp_data: dict) -> Polygon | MultiPolygon:
    """Extract the primary polygon from a component dictionary.

    Tries standard keys in order: trace, cross, arm.

    Args:
        comp_data: Output dict from a ``make_*`` geometry function.

    Returns:
        The primary Shapely polygon for the component.

    Raises:
        ValueError: If no polygon key is found.
    """
    for key in ("trace", "cross", "arm"):
        if key in comp_data and isinstance(comp_data[key], (Polygon, MultiPolygon)):
            return comp_data[key]
    raise ValueError(f"No polygon found in component data. Keys: {list(comp_data.keys())}")
