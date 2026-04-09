"""Edge feature extractor — geometric features from the micro-gap between two components.

Computes spatial relationship features from the physical proximity of
two Shapely polygons, including gap distance, overlap metrics, and a
rasterized interaction-window mask.
"""

from __future__ import annotations

from shapely.geometry import MultiPolygon, Polygon, box

from squadds.ml.universal.features.rasterizer import rasterize_fast


class NoInteractionError(Exception):
    """Raised when two polygons are too far apart to interact."""


def extract_edge_features(
    poly_a: Polygon | MultiPolygon,
    poly_b: Polygon | MultiPolygon,
    padding: float = 50.0,
    mask_resolution: int = 32,
) -> dict:
    """Extract geometric features from the gap between two components.

    Args:
        poly_a: First component polygon (metal trace).
        poly_b: Second component polygon (metal trace).
        padding: Bounding-box expansion in μm for the interaction window.
        mask_resolution: Resolution of the interaction window mask.

    Returns:
        Dictionary with keys:

        * ``shortest_gap`` – minimum distance between the polygons (μm).
        * ``overlap_length`` – length of the shared boundary region.
        * ``metal_area`` – total metal area in the interaction window.
        * ``void_area`` – total void area in the interaction window.
        * ``mask_a`` – rasterized mask of poly_a in the window, shape
          ``(mask_resolution, mask_resolution)``.
        * ``mask_b`` – rasterized mask of poly_b in the window, shape
          ``(mask_resolution, mask_resolution)``.
        * ``window_bounds`` – ``(minx, miny, maxx, maxy)`` of the window.

    Raises:
        NoInteractionError: If the padded bounding boxes do not overlap.
    """
    if isinstance(poly_a, MultiPolygon):
        poly_a = max(poly_a.geoms, key=lambda g: g.area)
    if isinstance(poly_b, MultiPolygon):
        poly_b = max(poly_b.geoms, key=lambda g: g.area)

    # ── Interaction window ─────────────────────────────────────────────
    bbox_a = box(*poly_a.bounds).buffer(padding)
    bbox_b = box(*poly_b.bounds).buffer(padding)

    window = bbox_a.intersection(bbox_b)
    if window.is_empty:
        raise NoInteractionError(
            f"Polygons are too far apart (>{padding}μm padding). BBox A: {poly_a.bounds}, BBox B: {poly_b.bounds}"
        )

    # ── Geometric metrics ──────────────────────────────────────────────
    shortest_gap = poly_a.distance(poly_b)

    # Clip polygons to the interaction window
    clipped_a = poly_a.intersection(window)
    clipped_b = poly_b.intersection(window)

    metal_area = clipped_a.area + clipped_b.area

    window_area = window.area
    void_area = max(0, window_area - metal_area)

    # Overlap length: length of shared boundary (if polygons touch/overlap)
    overlap = clipped_a.intersection(clipped_b)
    overlap_length = overlap.length if not overlap.is_empty else 0.0

    # ── Rasterize interaction window ───────────────────────────────────
    mask_a = rasterize_fast(clipped_a, resolution=mask_resolution)
    mask_b = rasterize_fast(clipped_b, resolution=mask_resolution)

    return {
        "shortest_gap": shortest_gap,
        "overlap_length": overlap_length,
        "metal_area": metal_area,
        "void_area": void_area,
        "mask_a": mask_a,
        "mask_b": mask_b,
        "window_bounds": window.bounds,
    }


class EdgeFeatureExtractor:
    """Class wrapper for edge geometric feature extraction."""

    def __init__(self, padding: float = 50.0, mask_resolution: int = 32):
        self.padding = padding
        self.mask_resolution = mask_resolution

    def extract(self, poly_a: Polygon | MultiPolygon, poly_b: Polygon | MultiPolygon) -> dict:
        """Extract features, handling no-interaction gracefully for the batch builder."""
        try:
            return extract_edge_features(poly_a, poly_b, padding=self.padding, mask_resolution=self.mask_resolution)
        except NoInteractionError:
            return {
                "shortest_gap": float("inf"),
                "overlap_length": 0.0,
                "metal_area": 0.0,
                "void_area": 0.0,
            }
