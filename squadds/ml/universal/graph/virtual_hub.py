"""Virtual hub node injector for graph construction.

The global node carries:
- Static embedding of the ENTIRE layout (union of all component polygons)
- Layer-stack info (dielectric_constant, substrate_thickness)
- Geometry bounds (total_width, total_height)

Edges from global → component nodes carry:
- Component center (cx, cy) relative to layout centroid
- component_area / layout_area
- component_perimeter / layout_perimeter
- Masked shape tensor: component rasterized within full layout bounds
"""

from __future__ import annotations

import numpy as np
import torch
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from torch_geometric.data import Data

from squadds.ml.universal.features.node_encoder import (
    DEFAULT_SHAPE_RESOLUTION,
    compute_static_embedding,
)


def _rasterize_in_bounds(
    polygon: Polygon | MultiPolygon,
    bounds: tuple[float, float, float, float],
    resolution: int,
) -> np.ndarray:
    """Rasterize a polygon within a specific bounding box (not its own).

    This produces a "masked" view: the component's shape rendered within
    the full layout's coordinate frame.

    Args:
        polygon: The polygon to rasterize.
        bounds: ``(minx, miny, maxx, maxy)`` of the target frame.
        resolution: Output grid size.

    Returns:
        ``np.ndarray`` of shape ``(resolution, resolution)``, float32.
    """
    mask = np.zeros((resolution, resolution), dtype=np.float32)
    if polygon.is_empty:
        return mask

    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    if width < 1e-10 or height < 1e-10:
        return mask

    # Map world coords → pixel coords
    # Use the full layout bounds as the coordinate frame
    col_coords = np.linspace(minx, maxx, resolution)
    row_coords = np.linspace(maxy, miny, resolution)  # top to bottom

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


# Hub-to-component edge: 2 (rel center) + 1 (area frac) + 1 (perim frac) + R*R (masked shape)
HUB_EDGE_SCALAR_DIM = 4


def hub_edge_feature_dim(shape_resolution: int = DEFAULT_SHAPE_RESOLUTION) -> int:
    """Dimension of a hub → component edge feature vector."""
    return HUB_EDGE_SCALAR_DIM + shape_resolution * shape_resolution


class VirtualHubInjector:
    """Inject a virtual global node with meaningful embedding and edge features.

    The hub node's static embedding is computed from the union of all
    component polygons, placing it in the same embedding space as
    component nodes.  Hub edges encode each component's spatial
    relationship to the full layout.
    """

    def __init__(
        self,
        shape_resolution: int = DEFAULT_SHAPE_RESOLUTION,
    ):
        self.shape_resolution = shape_resolution

    def inject(
        self,
        data: Data,
        component_polygons: list[Polygon | MultiPolygon],
        layout_params: dict[str, float] | None = None,
        global_info: dict[str, float] | None = None,
    ) -> Data:
        """Inject the virtual hub node into the PyG Data object.

        Args:
            data: The graph before hub injection.
            component_polygons: List of polygons, one per real node (same order).
            layout_params: Design params for the full layout (for param_sum).
            global_info: Extra global features (dielectric_constant, etc.).

        Returns:
            New Data with N+1 nodes, updated edges, and hub features.
        """
        num_real_nodes = data.x.size(0)
        hub_idx = num_real_nodes
        R = self.shape_resolution

        # ── Compute union polygon ─────────────────────────────────────
        valid_polys = [p for p in component_polygons if not p.is_empty]
        if valid_polys:
            layout_union = unary_union(valid_polys)
        else:
            layout_union = Polygon()

        # Hub node embedding: same space as component embeddings
        hub_embedding = compute_static_embedding(layout_union, params=layout_params, shape_resolution=R)

        # Append global info if provided
        if global_info:
            extra = np.array(list(global_info.values()), dtype=np.float32)
            hub_embedding = np.concatenate([hub_embedding, extra])

        # Pad hub to match node dim (pad with zeros if needed)
        node_dim = data.x.size(1)
        if len(hub_embedding) < node_dim:
            hub_embedding = np.pad(hub_embedding, (0, node_dim - len(hub_embedding)))
        elif len(hub_embedding) > node_dim:
            hub_embedding = hub_embedding[:node_dim]

        hub_feat = torch.from_numpy(hub_embedding).unsqueeze(0)  # (1, node_dim)
        new_x = torch.cat([data.x, hub_feat], dim=0)

        # ── Hub edges ─────────────────────────────────────────────────
        layout_bounds = layout_union.bounds if not layout_union.is_empty else (0, 0, 1, 1)
        layout_centroid = layout_union.centroid if not layout_union.is_empty else type("", (), {"x": 0, "y": 0})()
        layout_area = layout_union.area if not layout_union.is_empty else 1.0
        layout_perimeter = layout_union.length if not layout_union.is_empty else 1.0

        hub_edge_features = []
        for poly in component_polygons:
            if isinstance(poly, MultiPolygon):
                poly = max(poly.geoms, key=lambda g: g.area)

            if poly.is_empty:
                feat = np.zeros(hub_edge_feature_dim(R), dtype=np.float32)
            else:
                cx_rel = np.float32(poly.centroid.x - layout_centroid.x)
                cy_rel = np.float32(poly.centroid.y - layout_centroid.y)
                area_frac = np.float32(poly.area / layout_area) if layout_area > 0 else np.float32(0)
                perim_frac = np.float32(poly.length / layout_perimeter) if layout_perimeter > 0 else np.float32(0)

                masked_shape = _rasterize_in_bounds(poly, layout_bounds, R).flatten()

                feat = np.concatenate([[cx_rel, cy_rel, area_frac, perim_frac], masked_shape]).astype(np.float32)

            hub_edge_features.append(feat)

        # Build edge indices: hub <-> all real nodes
        hub_src = torch.full((num_real_nodes,), hub_idx, dtype=torch.long)
        real_dst = torch.arange(num_real_nodes, dtype=torch.long)
        edges_out = torch.stack([hub_src, real_dst], dim=0)
        edges_in = torch.stack([real_dst, hub_src], dim=0)
        new_edge_index = torch.cat([data.edge_index, edges_out, edges_in], dim=1)

        # Build hub edge attr — same feature for both directions
        hub_edge_tensors = [torch.from_numpy(f) for f in hub_edge_features]
        hub_edge_attr = torch.stack(hub_edge_tensors, dim=0)  # (N, hub_edge_dim)
        # Duplicate for reverse edges
        hub_edge_attr_both = torch.cat([hub_edge_attr, hub_edge_attr], dim=0)  # (2N, hub_edge_dim)

        # Pad existing edge_attr or hub_edge_attr to match dimensions
        existing_edge_dim = data.edge_attr.size(1) if data.edge_attr.size(0) > 0 else 0
        hub_edge_dim = hub_edge_attr_both.size(1)
        target_edge_dim = max(existing_edge_dim, hub_edge_dim)

        if existing_edge_dim < target_edge_dim and data.edge_attr.size(0) > 0:
            pad = torch.zeros(data.edge_attr.size(0), target_edge_dim - existing_edge_dim)
            existing_padded = torch.cat([data.edge_attr, pad], dim=1)
        else:
            existing_padded = data.edge_attr

        if hub_edge_dim < target_edge_dim:
            pad = torch.zeros(hub_edge_attr_both.size(0), target_edge_dim - hub_edge_dim)
            hub_padded = torch.cat([hub_edge_attr_both, pad], dim=1)
        else:
            hub_padded = hub_edge_attr_both

        new_edge_attr = torch.cat([existing_padded, hub_padded], dim=0)

        # ── Targets ───────────────────────────────────────────────────
        new_y = data.y
        if data.y is not None:
            num_targets = data.y.size(1)
            hub_targets = torch.full((1, num_targets), float("nan"))
            new_y = torch.cat([data.y, hub_targets], dim=0)

        new_y_edge = data.y_edge
        if data.y_edge is not None:
            num_edge_targets = data.y_edge.size(1)
            spatial_edge_targets = torch.full((2 * num_real_nodes, num_edge_targets), float("nan"))
            new_y_edge = torch.cat([data.y_edge, spatial_edge_targets], dim=0)

        new_data = Data(x=new_x, edge_index=new_edge_index, edge_attr=new_edge_attr, y=new_y)
        new_data.y_edge = new_y_edge

        return new_data
