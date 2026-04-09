"""Geometric moment extraction from Shapely polygons.

Computes 8 geometric properties directly from the Shapely polygon —
no JSON equations or qiskit-metal dependencies.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Polygon


def compute_moments(polygon: Polygon | MultiPolygon) -> np.ndarray:
    """Compute 8 geometric features from a Shapely polygon.

    All values are computed **directly** from the polygon geometry.

    Returns:
        ``np.ndarray`` of shape ``(8,)`` containing:

        ======  ======================  ================================
        Index   Name                    Formula
        ======  ======================  ================================
        0       area                    ``polygon.area``
        1       perimeter               ``polygon.length``
        2       aspect_ratio            ``bbox_width / bbox_height``
        3       fill_factor             ``area / bbox_area``
        4       centroid_x              ``polygon.centroid.x``
        5       centroid_y              ``polygon.centroid.y``
        6       I_x (2nd moment, x)     Numerical integration
        7       I_y (2nd moment, y)     Numerical integration
        ======  ======================  ================================
    """
    if isinstance(polygon, MultiPolygon):
        # Merge into largest component for consistent moments
        polygon = max(polygon.geoms, key=lambda g: g.area)

    if polygon.is_empty:
        return np.zeros(8, dtype=np.float32)

    # ── Basic properties ───────────────────────────────────────────────
    area = polygon.area
    perimeter = polygon.length

    # Bounding box
    minx, miny, maxx, maxy = polygon.bounds
    bbox_width = maxx - minx
    bbox_height = maxy - miny

    aspect_ratio = bbox_width / bbox_height if bbox_height > 1e-10 else 1.0
    bbox_area = bbox_width * bbox_height
    fill_factor = area / bbox_area if bbox_area > 1e-10 else 0.0

    # Centroid (position information retained)
    cx = polygon.centroid.x
    cy = polygon.centroid.y

    # ── Second moments of area (Ix, Iy) ───────────────────────────────
    # Computed via the shoelace-like formula over the exterior ring
    Ix, Iy = _compute_second_moments(polygon, cx, cy)

    return np.array([area, perimeter, aspect_ratio, fill_factor, cx, cy, Ix, Iy], dtype=np.float32)


def _compute_second_moments(
    polygon: Polygon,
    cx: float,
    cy: float,
) -> tuple[float, float]:
    """Compute second moments of area about the centroid.

    Uses the Green's theorem formulation over the polygon boundary.

    Args:
        polygon: Input Shapely polygon.
        cx: Centroid x-coordinate.
        cy: Centroid y-coordinate.

    Returns:
        ``(Ix, Iy)`` — second moments of area about the centroid.
    """
    coords = np.array(polygon.exterior.coords)
    x = coords[:, 0] - cx
    y = coords[:, 1] - cy
    n = len(x) - 1  # last point duplicates first

    # Shoelace-based second moment computation
    # Ix = Σ (x_i * y_{i+1} - x_{i+1} * y_i) * (y_i² + y_i*y_{i+1} + y_{i+1}²) / 12
    # Iy = Σ (x_i * y_{i+1} - x_{i+1} * y_i) * (x_i² + x_i*x_{i+1} + x_{i+1}²) / 12
    Ix = 0.0
    Iy = 0.0
    for i in range(n):
        cross = x[i] * y[i + 1] - x[i + 1] * y[i]
        Ix += cross * (y[i] ** 2 + y[i] * y[i + 1] + y[i + 1] ** 2)
        Iy += cross * (x[i] ** 2 + x[i] * x[i + 1] + x[i + 1] ** 2)

    Ix = abs(Ix) / 12.0
    Iy = abs(Iy) / 12.0

    return Ix, Iy


def moment_names() -> list[str]:
    """Return the names of the 8 moment features (for labelling)."""
    return [
        "area",
        "perimeter",
        "aspect_ratio",
        "fill_factor",
        "centroid_x",
        "centroid_y",
        "I_x",
        "I_y",
    ]
