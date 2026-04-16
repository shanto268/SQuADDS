"""Rasterize Shapely polygons to fixed-size binary masks.

Produces aspect-ratio-preserving rasterizations with the polygon
centred in the grid.  Used as input to the CNN encoder.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union


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


def compute_shared_bounds(
    polygons: list[Polygon | MultiPolygon],
    *,
    padding: float = 0.0,
    padding_fraction: float = 0.0,
) -> tuple[float, float, float, float]:
    """Compute a shared bounding box covering all polygons.

    Args:
        polygons: Polygons that should share a common coordinate frame.
        padding: Absolute padding added to all sides in world coordinates.
        padding_fraction: Additional padding relative to the largest bbox side.

    Returns:
        ``(minx, miny, maxx, maxy)`` covering the input polygons.

    Raises:
        ValueError: If no non-empty polygons are provided.
    """
    valid_polygons: list[Polygon | MultiPolygon] = []
    for polygon in polygons:
        if polygon.is_empty:
            continue
        valid_polygons.append(polygon)

    if not valid_polygons:
        raise ValueError("Cannot compute shared bounds for an empty polygon list.")

    union = unary_union(valid_polygons)
    minx, miny, maxx, maxy = union.bounds
    width = maxx - minx
    height = maxy - miny
    pad = float(padding) + max(width, height) * float(padding_fraction)
    return (minx - pad, miny - pad, maxx + pad, maxy + pad)


def rasterize_in_bounds(
    polygon: Polygon | MultiPolygon,
    bounds: tuple[float, float, float, float],
    resolution: int,
) -> np.ndarray:
    """Rasterize a polygon inside a caller-specified bounding box.

    This is the key primitive for shared-frame embeddings: multiple polygons can
    be rasterized within the same coordinate frame instead of each normalizing
    to its own bounding box.
    """
    mask = np.zeros((resolution, resolution), dtype=np.float32)

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
        from shapely.prepared import prep

        prepared = prep(polygon)
        mask_flat = np.array([prepared.contains(Point(x, y)) for x, y in points])

    return mask_flat.reshape(resolution, resolution).astype(np.float32)
