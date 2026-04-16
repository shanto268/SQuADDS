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
    The vector is fully scale-descriptive: it captures size (area, perimeter,
    bbox dims) and shape ratios (fill_factor, compactness, aspect_ratio).

    Returns:
        ``np.ndarray`` of shape ``(8,)`` containing:

        ======  ======================  ================================
        Index   Name                    Formula
        ======  ======================  ================================
        0       area                    ``polygon.area``
        1       perimeter               ``polygon.length``
        2       bbox_area               ``bbox_width * bbox_height``
        3       bbox_perimeter          ``2 * (bbox_width + bbox_height)``
        4       fill_factor             ``area / bbox_area``
        5       compactness             ``perimeter / bbox_perimeter``
        6       aspect_ratio            ``bbox_width / bbox_height``
        7       circularity             ``4pi * area / perimeter^2``
        ======  ======================  ================================
    """
    if polygon.is_empty:
        return np.zeros(8, dtype=np.float32)

    area = polygon.area
    perimeter = polygon.length

    minx, miny, maxx, maxy = polygon.bounds
    bbox_width = maxx - minx
    bbox_height = maxy - miny

    bbox_area = bbox_width * bbox_height
    bbox_perimeter = 2 * (bbox_width + bbox_height)

    fill_factor = area / bbox_area if bbox_area > 1e-10 else 0.0
    compactness = perimeter / bbox_perimeter if bbox_perimeter > 1e-10 else 0.0
    aspect_ratio = bbox_width / bbox_height if bbox_height > 1e-10 else 1.0
    circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 1e-10 else 0.0

    return np.array(
        [area, perimeter, bbox_area, bbox_perimeter, fill_factor, compactness, aspect_ratio, circularity],
        dtype=np.float32,
    )


MOMENT_DIM = 8


def moment_names() -> list[str]:
    """Return the names of the 8 moment features (for labelling)."""
    return [
        "area",
        "perimeter",
        "bbox_area",
        "bbox_perimeter",
        "fill_factor",
        "compactness",
        "aspect_ratio",
        "circularity",
    ]
