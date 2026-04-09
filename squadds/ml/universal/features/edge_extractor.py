"""Edge feature extractor — geometric + coupling features between two components.

Computes a rich edge feature vector encoding:
1. Coupling type (one-hot: capacitive, galvanic, inductive)
2. Center-to-center distance (dx, dy)
3. Overlap geometry moments (area, perimeter, bbox_area)
4. Overlap shape tensor (scale-invariant rasterized overlap region)
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from squadds.ml.universal.features.node_encoder import DEFAULT_SHAPE_RESOLUTION
from squadds.ml.universal.features.rasterizer import rasterize_fast

# Supported coupling types → one-hot indices
COUPLING_TYPES = {"capacitive": 0, "galvanic": 1, "inductive": 2}
NUM_COUPLING_TYPES = len(COUPLING_TYPES)

# Edge feature breakdown:
#   3 (coupling one-hot) + 2 (dx, dy) + 3 (overlap moments) + R*R (overlap shape)
EDGE_SCALAR_DIM = NUM_COUPLING_TYPES + 2 + 3  # = 8


def edge_feature_dim(shape_resolution: int = DEFAULT_SHAPE_RESOLUTION) -> int:
    """Return the total dimension of the edge feature vector."""
    return EDGE_SCALAR_DIM + shape_resolution * shape_resolution


def extract_edge_features(
    poly_a: Polygon | MultiPolygon,
    poly_b: Polygon | MultiPolygon,
    coupling_type: str = "capacitive",
    shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
) -> np.ndarray:
    """Compute edge features between two component polygons.

    Args:
        poly_a: First component polygon.
        poly_b: Second component polygon.
        coupling_type: One of ``"capacitive"``, ``"galvanic"``, ``"inductive"``.
        shape_resolution: Resolution for the overlap shape tensor.

    Returns:
        ``np.ndarray`` of shape ``(edge_feature_dim,)`` with dtype ``float32``.
    """
    if isinstance(poly_a, MultiPolygon):
        poly_a = max(poly_a.geoms, key=lambda g: g.area)
    if isinstance(poly_b, MultiPolygon):
        poly_b = max(poly_b.geoms, key=lambda g: g.area)

    # 1. Coupling type one-hot
    coupling_vec = np.zeros(NUM_COUPLING_TYPES, dtype=np.float32)
    idx = COUPLING_TYPES.get(coupling_type, 0)
    coupling_vec[idx] = 1.0

    # 2. Center-to-center distance
    ca = poly_a.centroid
    cb = poly_b.centroid
    dx = np.float32(cb.x - ca.x)
    dy = np.float32(cb.y - ca.y)

    # 3. Overlap geometry
    #    For non-galvanic: use buffer-expanded intersection to capture the gap region
    #    For galvanic: direct intersection (shared boundary)
    if coupling_type == "galvanic":
        overlap = poly_a.intersection(poly_b)
    else:
        # Buffer both polygons slightly to capture the near-field gap
        buf_a = poly_a.buffer(50.0)
        buf_b = poly_b.buffer(50.0)
        overlap = buf_a.intersection(buf_b)

    if overlap.is_empty:
        overlap_area = np.float32(0.0)
        overlap_perimeter = np.float32(0.0)
        overlap_bbox_area = np.float32(0.0)
        overlap_shape = np.zeros(shape_resolution * shape_resolution, dtype=np.float32)
    else:
        overlap_area = np.float32(overlap.area)
        overlap_perimeter = np.float32(overlap.length)
        ominx, ominy, omaxx, omaxy = overlap.bounds
        overlap_bbox_area = np.float32((omaxx - ominx) * (omaxy - ominy))
        # 4. Scale-invariant overlap shape tensor
        overlap_shape = rasterize_fast(overlap, resolution=shape_resolution).flatten()

    scalars = np.array(
        [*coupling_vec, dx, dy, overlap_area, overlap_perimeter, overlap_bbox_area],
        dtype=np.float32,
    )

    return np.concatenate([scalars, overlap_shape]).astype(np.float32)


class EdgeFeatureExtractor:
    """Stateless class wrapper for edge feature extraction."""

    def __init__(self, shape_resolution: int = DEFAULT_SHAPE_RESOLUTION):
        self.shape_resolution = shape_resolution

    def extract(
        self,
        poly_a: Polygon | MultiPolygon,
        poly_b: Polygon | MultiPolygon,
        coupling_type: str = "capacitive",
    ) -> np.ndarray:
        """Extract edge features between two polygons.

        Returns:
            ``np.ndarray`` of shape ``(edge_feature_dim,)`` with dtype ``float32``.
        """
        return extract_edge_features(
            poly_a, poly_b, coupling_type=coupling_type, shape_resolution=self.shape_resolution
        )

    @property
    def dim(self) -> int:
        """Total edge feature dimension."""
        return edge_feature_dim(self.shape_resolution)
