"""Rasterize Shapely polygons to fixed-size binary masks.

Produces aspect-ratio-preserving rasterizations with the polygon
centred in the grid.  Used as input to the CNN encoder.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon


def rasterize(
    polygon: Polygon | MultiPolygon,
    resolution: int = 64,
) -> np.ndarray:
    """Rasterize a polygon to a binary mask.

    The polygon is scaled to fit the longest bounding-box dimension
    within ``resolution`` pixels, preserving aspect ratio.  The polygon
    is centred in the grid.

    Args:
        polygon: Input Shapely polygon.
        resolution: Output grid size (``resolution × resolution``).

    Returns:
        ``np.ndarray`` of shape ``(resolution, resolution)`` with dtype
        ``float32``.  Values are 1.0 inside the polygon, 0.0 outside.
    """
    mask = np.zeros((resolution, resolution), dtype=np.float32)

    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda g: g.area)

    if polygon.is_empty:
        return mask

    # ── Compute transform ──────────────────────────────────────────────
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny

    if width < 1e-10 and height < 1e-10:
        return mask

    max_dim = max(width, height)
    # Leave 2-pixel padding on each side
    usable = resolution - 4
    scale = usable / max_dim

    # Offset to centre in grid
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    offset_x = resolution / 2 - cx * scale
    offset_y = resolution / 2 - cy * scale

    # ── Rasterize via point-in-polygon ─────────────────────────────────
    # For each pixel centre, check if it's inside the polygon
    # This is straightforward and correct for arbitrary polygons
    prepared = polygon  # could use shapely.prepared for speed

    for row in range(resolution):
        for col in range(resolution):
            # Map pixel (col, row) → world coordinates
            # row=0 is top of image → highest y value
            wx = (col - offset_x) / scale
            wy = ((resolution - 1 - row) - offset_y) / scale
            if prepared.contains(Point(wx, wy)):
                mask[row, col] = 1.0

    return mask


def rasterize_fast(
    polygon: Polygon | MultiPolygon,
    resolution: int = 64,
) -> np.ndarray:
    """Vectorized rasterization using coordinate meshgrid.

    Significantly faster than the point-by-point version for large
    resolutions, using Shapely's vectorized contains check.

    Args:
        polygon: Input Shapely polygon.
        resolution: Output grid size.

    Returns:
        ``np.ndarray`` of shape ``(resolution, resolution)``, float32.
    """
    mask = np.zeros((resolution, resolution), dtype=np.float32)

    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda g: g.area)

    if polygon.is_empty:
        return mask

    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny

    if width < 1e-10 and height < 1e-10:
        return mask

    max_dim = max(width, height)
    usable = resolution - 4
    scale = usable / max_dim

    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2

    # Create world-coordinate grid
    pixel_coords = np.arange(resolution)
    col_world = (pixel_coords - resolution / 2) / scale + cx
    row_world = ((resolution - 1 - pixel_coords) - resolution / 2) / scale + cy

    cols, rows = np.meshgrid(col_world, row_world)
    points = np.column_stack([cols.ravel(), rows.ravel()])

    # Vectorized contains check
    try:
        from shapely import contains_xy

        mask_flat = contains_xy(polygon, points[:, 0], points[:, 1])
    except (ImportError, AttributeError):
        # Fallback: use prepared geometry
        from shapely.prepared import prep

        prepared = prep(polygon)
        mask_flat = np.array([prepared.contains(Point(x, y)) for x, y in points])

    mask = mask_flat.reshape(resolution, resolution).astype(np.float32)
    return mask
