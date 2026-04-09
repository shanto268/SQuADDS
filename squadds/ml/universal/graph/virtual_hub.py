"""Virtual hub node feature computation for heterogeneous graphs.

Provides functions to compute:
- Hub node embedding from the layout union polygon
- Spatial edge features (hub <-> component)
"""

from __future__ import annotations

import numpy as np
import torch
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from squadds.ml.universal.features.moments import MOMENT_DIM, compute_moments
from squadds.ml.universal.features.node_encoder import DEFAULT_SHAPE_RESOLUTION
from squadds.ml.universal.features.rasterizer import rasterize_fast


def _rasterize_in_bounds(
    polygon: Polygon | MultiPolygon,
    bounds: tuple[float, float, float, float],
    resolution: int,
) -> np.ndarray:
    """Rasterize a polygon within a specific bounding box (not its own).

    Produces a 'masked' view: the component's shape rendered within
    the full layout's coordinate frame.
    """
    mask = np.zeros((resolution, resolution), dtype=np.float32)
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda g: g.area)
    if polygon.is_empty:
        return mask

    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    if width < 1e-10 or height < 1e-10:
        return mask

    col_coords = np.linspace(minx, maxx, resolution)
    row_coords = np.linspace(maxy, miny, resolution)
    cols, rows = np.meshgrid(col_coords, row_coords)
    points = np.column_stack([cols.ravel(), rows.ravel()])

    try:
        from shapely import contains_xy

        mask_flat = contains_xy(polygon, points[:, 0], points[:, 1])
    except (ImportError, AttributeError):
        from shapely.geometry import Point
        from shapely.prepared import prep

        prepared = prep(polygon)
        mask_flat = np.array([prepared.contains(Point(x, y)) for x, y in points])

    return mask_flat.reshape(resolution, resolution).astype(np.float32)


def compute_hub_embedding(
    component_polygons: list[Polygon | MultiPolygon],
    layout_params: dict[str, float] | None = None,
    global_info: dict[str, float] | None = None,
    shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
) -> np.ndarray:
    """Compute the virtual hub node embedding.

    The hub embedding contains:
    - Full layout union shape tensor (R*R)
    - Full layout geometric moments (8)
    - Layout param sum (1)
    - Layer stack / global info (variable)

    Args:
        component_polygons: List of component polygons.
        layout_params: Design parameters for param sum.
        global_info: Global features (dielectric_constant, etc.).
        shape_resolution: Resolution for the shape tensor.

    Returns:
        Hub embedding vector as np.ndarray.
    """
    R = shape_resolution

    valid_polys = [p for p in component_polygons if not p.is_empty]
    layout_union = unary_union(valid_polys) if valid_polys else Polygon()

    # Shape tensor of the full layout
    shape_tensor = rasterize_fast(layout_union, resolution=R).flatten()

    # Geometric moments of the full layout
    moments = compute_moments(layout_union)

    # Param sum
    param_sum = np.float32(sum(layout_params.values())) if layout_params else np.float32(0.0)

    # Total chip metrics
    chip_area = np.float32(layout_union.area) if not layout_union.is_empty else np.float32(0)
    metal_fill = (
        np.float32(
            layout_union.area
            / ((layout_union.bounds[2] - layout_union.bounds[0]) * (layout_union.bounds[3] - layout_union.bounds[1]))
        )
        if not layout_union.is_empty and layout_union.area > 0
        else np.float32(0)
    )

    parts = [shape_tensor, moments, [param_sum, chip_area, metal_fill]]

    # Global info (layer stack) — fixed-size slot to prevent dim mismatches
    global_vec = np.zeros(N_GLOBAL_SLOTS, dtype=np.float32)
    if global_info:
        vals = list(global_info.values())
        n = min(len(vals), N_GLOBAL_SLOTS)
        global_vec[:n] = vals[:n]
    parts.append(global_vec)

    return np.concatenate(parts).astype(np.float32)


# Fixed number of global feature slots (zero-padded if fewer are provided)
N_GLOBAL_SLOTS = 5


def hub_embedding_dim(
    shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
) -> int:
    """Return the dimension of the hub node embedding."""
    return shape_resolution * shape_resolution + MOMENT_DIM + 3 + N_GLOBAL_SLOTS


# ── Spatial edge features ─────────────────────────────────────────────
# Each spatial edge carries:
#   - relative center (dx, dy): 2
#   - area fraction: 1
#   - perimeter fraction: 1
#   - masked shape tensor (component in layout bounds): R*R
SPATIAL_SCALAR_DIM = 4


def spatial_edge_feature_dim(shape_resolution: int = DEFAULT_SHAPE_RESOLUTION) -> int:
    """Return the dimension of a spatial edge feature vector."""
    return SPATIAL_SCALAR_DIM + shape_resolution * shape_resolution


def compute_spatial_edge_features(
    component_polygons: list[Polygon | MultiPolygon],
    shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
) -> torch.Tensor:
    """Compute spatial edge features for all components.

    Args:
        component_polygons: List of component polygons (same order as nodes).
        shape_resolution: Resolution for masked shape tensors.

    Returns:
        Tensor of shape (N_comp, spatial_edge_dim).
    """
    R = shape_resolution

    valid_polys = [p for p in component_polygons if not p.is_empty]
    layout_union = unary_union(valid_polys) if valid_polys else Polygon()
    layout_bounds = layout_union.bounds if not layout_union.is_empty else (0, 0, 1, 1)

    layout_cx = layout_union.centroid.x if not layout_union.is_empty else 0
    layout_cy = layout_union.centroid.y if not layout_union.is_empty else 0
    layout_area = layout_union.area if not layout_union.is_empty else 1.0
    layout_perimeter = layout_union.length if not layout_union.is_empty else 1.0

    features = []
    for poly in component_polygons:
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)

        if poly.is_empty:
            feat = np.zeros(spatial_edge_feature_dim(R), dtype=np.float32)
        else:
            cx_rel = np.float32(poly.centroid.x - layout_cx)
            cy_rel = np.float32(poly.centroid.y - layout_cy)
            area_frac = np.float32(poly.area / layout_area)
            perim_frac = np.float32(poly.length / layout_perimeter)
            masked_shape = _rasterize_in_bounds(poly, layout_bounds, R).flatten()

            feat = np.concatenate([[cx_rel, cy_rel, area_frac, perim_frac], masked_shape]).astype(np.float32)

        features.append(torch.from_numpy(feat))

    return torch.stack(features)
