"""Standalone Shapely polygon generator for the CPW resonator.

Faithfully reproduces:
1. The ``second_cpw`` L-shaped coupling arm from
   ``CoupledLineTee.make()`` (coupled_line_tee.py L98-106).
2. The meandered CPW path from ``RouteMeander.connect_meandered()``
   (meandered.py L103-311), including the exact interleaving pattern,
   snap-to-grid, lead-in/lead-out, and jog extensions.
3. Corner filleting (rounded corners) matching qiskit-metal's fillet
   parameter, which rounds each 90° bend into a smooth arc.

The resonator polygon is the union of the meander path and the CLT
coupling arm, buffered to the CPW trace width.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm
from shapely import affinity
from shapely.geometry import LineString, Polygon

# ── Utility functions (matching qiskit-metal) ──────────────────────────


def _snap_unit_vector(vec: np.ndarray) -> np.ndarray:
    """Snap a 2D unit vector to the nearest axis (qiskit-metal snap)."""
    m = np.argmax(np.abs(vec))
    v = np.array([0.0, 0.0])
    v[m] = np.sign(vec[m])
    return v


def _rotate_vec(vec: np.ndarray, radians: float) -> np.ndarray:
    """Rotate a 2D vector counter-clockwise by radians."""
    c, s = np.cos(radians), np.sin(radians)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def _get_unit_vectors(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    snap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute forward and sideways unit vectors (matching QRoute.get_unit_vectors)."""
    v = end_pos - start_pos
    v_norm = norm(v)
    if v_norm < 1e-10:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    direction = v / v_norm
    if snap:
        direction = _snap_unit_vector(direction)
    sideways = _rotate_vec(direction, np.pi / 2)
    return direction, sideways


def _get_index_for_side1_meander(num_root_pts: int) -> tuple[np.ndarray, int]:
    """Exact copy of RouteMeander.get_index_for_side1_meander."""
    num_2pts, odd = divmod(num_root_pts, 2)
    x = np.array(range(num_2pts), dtype=int) * 4
    z = np.zeros(num_2pts * 2, dtype=int)
    z[::2] = x
    z[1::2] = x + 1
    return z, odd


def _fillet_path(pts: np.ndarray, radius: float, segments_per_corner: int = 16) -> np.ndarray:
    """Apply fillet (corner rounding) to a polyline, replacing each
    sharp corner with an arc of the given radius.

    This matches qiskit-metal's fillet behaviour, which rounds every
    90° corner of the meander path into a smooth circular arc.

    Args:
        pts: (N, 2) array of path vertices.
        radius: Fillet radius in μm. Corners where the adjacent
            segments are shorter than ``radius`` are left sharp.
        segments_per_corner: Number of arc segments per 90° corner.

    Returns:
        (M, 2) array of the filleted path (M >= N).
    """
    if radius <= 0 or len(pts) < 3:
        return pts

    result = [pts[0].copy()]

    for i in range(1, len(pts) - 1):
        p_prev = pts[i - 1]
        p_curr = pts[i]
        p_next = pts[i + 1]

        # Vectors from corner to neighbours
        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        len1 = norm(v1)
        len2 = norm(v2)

        # Skip degenerate segments
        if len1 < 1e-6 or len2 < 1e-6:
            result.append(p_curr.copy())
            continue

        # Clamp radius so the arc doesn't exceed half the segment length
        r = min(radius, len1 / 2.0, len2 / 2.0)
        if r < 1e-6:
            result.append(p_curr.copy())
            continue

        # Unit vectors
        u1 = v1 / len1
        u2 = v2 / len2

        # Angle between the two edges
        cos_a = np.clip(np.dot(u1, u2), -1.0, 1.0)
        angle = np.arccos(cos_a)

        # Near straight or near u-turn: skip fillet
        if angle < 1e-6 or abs(angle - np.pi) < 1e-6:
            result.append(p_curr.copy())
            continue

        # Tangent points
        t1 = p_curr + u1 * r
        t2 = p_curr + u2 * r

        # Arc centre: offset from corner along the angle bisector
        bisector = u1 + u2
        bisector_len = norm(bisector)
        if bisector_len < 1e-10:
            result.append(p_curr.copy())
            continue
        bisector = bisector / bisector_len

        # Distance from corner to arc centre
        d_center = r / np.sin(angle / 2)
        center = p_curr + bisector * d_center

        # Generate arc points from t1 to t2 around center
        a1 = np.arctan2(t1[1] - center[1], t1[0] - center[0])
        a2 = np.arctan2(t2[1] - center[1], t2[0] - center[0])

        # Determine sweep direction (shortest arc)
        da = a2 - a1
        if da > np.pi:
            da -= 2 * np.pi
        elif da < -np.pi:
            da += 2 * np.pi

        # Number of segments proportional to arc angle
        n_segs = max(2, int(abs(da) / (np.pi / 2) * segments_per_corner))

        for j in range(n_segs + 1):
            t = j / n_segs
            angle_j = a1 + t * da
            arc_pt = center + r * np.array([np.cos(angle_j), np.sin(angle_j)])
            result.append(arc_pt)

    result.append(pts[-1].copy())
    return np.array(result)


# ── CLT second_cpw (coupling arm) ─────────────────────────────────────


def _make_clt_second_cpw(
    coupling_length: float,
    down_length: float,
    prime_width: float,
    prime_gap: float,
    second_width: float,
    second_gap: float,
    coupling_space: float,
    open_termination: bool,
    mirror: bool,
    pos_x: float,
    pos_y: float,
    orientation: float,
) -> dict:
    """Generate the CoupledLineTee second_cpw (L-shaped coupling arm).

    Returns dict with 'centerline', 'trace', 'etch', 'pin_second_end'
    (position), 'pin_second_end_direction'.
    """
    second_flip = -1 if mirror else 1

    second_y = -(prime_width / 2 + prime_gap + coupling_space + second_gap + second_width / 2)

    # L-shaped path: horizontal run + vertical drop
    pts = [
        [second_flip * (-coupling_length / 2), second_y],
        [second_flip * (coupling_length / 2), second_y],
        [second_flip * (coupling_length / 2), second_y - down_length],
    ]
    centerline = LineString(pts)

    # Etch path (may extend for open termination)
    second_termination = second_gap if open_termination else 0
    etch_pts = [
        [second_flip * (-coupling_length / 2 - second_termination), second_y],
        [second_flip * (coupling_length / 2), second_y],
        [second_flip * (coupling_length / 2), second_y - down_length],
    ]
    etch_line = LineString(etch_pts)

    # Buffer to create polygons
    trace: Polygon = centerline.buffer(second_width / 2, cap_style="flat", join_style="mitre")
    etch: Polygon = etch_line.buffer((second_width + 2 * second_gap) / 2, cap_style="flat", join_style="mitre")

    # Pin at the bottom of the L
    pin_pos_raw = np.array(pts[-1], dtype=float)
    # Direction: pointing downward (away from the coupling region)
    pin_dir_raw = np.array([0.0, -1.0])

    # Rotate and translate
    geoms = [centerline, trace, etch]
    geoms = [affinity.rotate(g, orientation, origin=(0, 0)) for g in geoms]
    geoms = [affinity.translate(g, pos_x, pos_y) for g in geoms]
    centerline, trace, etch = geoms

    theta = np.radians(orientation)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    offset = np.array([pos_x, pos_y])
    pin_pos = rot @ pin_pos_raw + offset
    pin_dir = rot @ pin_dir_raw

    return {
        "centerline": centerline,
        "trace": trace,
        "etch": etch,
        "pin_second_end": pin_pos,
        "pin_second_end_direction": pin_dir,
    }


# ── Lead generation (matching QRouteLead) ──────────────────────────────


def _build_lead(
    pin_pos: np.ndarray,
    pin_dir: np.ndarray,
    straight: float,
    jog_extension: list[tuple[str, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build a lead (straight + jogs) from a pin.

    Args:
        pin_pos: Pin starting position.
        pin_dir: Pin direction (outward normal).
        straight: Length of the initial straight segment.
        jog_extension: List of (turn, length) tuples for jogs. e.g.
            [("R90", 100)].

    Returns:
        (pts, direction, total_length) — points array (N,2), final direction,
        and total path length.
    """
    pts = [pin_pos.copy()]
    direction = pin_dir.copy()
    total = 0.0

    # Minimum lead length
    lead_len = max(straight, 5.0)  # trace_width/2 fallback
    tip = pts[-1] + direction * lead_len
    pts.append(tip)
    total += lead_len

    # Jog extensions
    if jog_extension:
        for turn, length in jog_extension:
            # Parse turn direction
            if isinstance(turn, str):
                if turn.startswith("R"):
                    angle_str = turn[1:] if len(turn) > 1 else "90"
                    angle_deg = float(angle_str)
                    direction = _rotate_vec(direction, -np.radians(angle_deg))
                elif turn.startswith("L"):
                    angle_str = turn[1:] if len(turn) > 1 else "90"
                    angle_deg = float(angle_str)
                    direction = _rotate_vec(direction, np.radians(angle_deg))
                else:
                    direction = _rotate_vec(direction, np.radians(float(turn)))
            else:
                direction = _rotate_vec(direction, np.radians(float(turn)))

            tip = pts[-1] + direction * length
            pts.append(tip)
            total += length

    return np.array(pts), direction, total


# ── Meander routing (faithful to RouteMeander.connect_meandered) ───────


def _connect_meandered(
    start_pos: np.ndarray,
    start_dir: np.ndarray,
    end_pos: np.ndarray,
    end_dir: np.ndarray,
    length_meander: float,
    spacing: float,
    asymmetry: float = 0.0,
    snap: bool = True,
) -> np.ndarray:
    """Generate meander points between two endpoints.

    This is a faithful reproduction of RouteMeander.connect_meandered(),
    using the exact same interleaving logic and index patterns.

    Args:
        start_pos: Position of meander start (after lead-in).
        start_dir: Direction at meander start.
        end_pos: Position of meander end (before lead-out).
        end_dir: Direction at meander end.
        length_meander: Total path length to distribute in the meander.
        spacing: Spacing between adjacent meander curves.
        asymmetry: Offset of meander center-line.
        snap: Whether to snap to grid.

    Returns:
        (M, 2) array of meander path coordinates.
    """
    # ── Coordinate system ──────────────────────────────────────────────
    forward, sideways = _get_unit_vectors(start_pos, end_pos, snap=snap)

    # Calculate distances
    dist = end_pos - start_pos
    if snap:
        length_direct = abs(np.dot(dist, forward))
    else:
        length_direct = norm(dist)

    # Number of meander segments
    meander_number = int(np.floor(length_direct / spacing))
    if meander_number < 1:
        return np.empty((0, 2), float)

    # Adjust parity (matching L165-176 of meandered.py)
    start_sw = np.dot(start_dir, sideways)
    end_sw = np.dot(end_dir, sideways)
    if round(start_sw * end_sw, 10) > 0 and (meander_number % 2) == 0:
        meander_number -= 1
    elif round(start_sw * end_sw, 10) < 0 and (meander_number % 2) == 1:
        meander_number -= 1

    meander_number = max(1, meander_number)

    # First meander direction (L178-192)
    if start_sw > 0:
        first_meander_sideways = True
    elif start_sw < 0:
        first_meander_sideways = False
    else:
        if end_sw > 0:
            first_meander_sideways = (meander_number % 2) == 1
        elif end_sw < 0:
            first_meander_sideways = (meander_number % 2) == 0
        else:
            first_meander_sideways = True

    # Perpendicular height (L194-197)
    length_excess = length_meander - length_direct - 2 * abs(asymmetry)
    length_perp = max(0, length_excess / (meander_number * 2.0))

    # ── Root points along forward axis (L199-205) ──────────────────────
    middle_points = np.array([forward] * int(meander_number + 1))
    scale_bys = spacing * np.arange(int(meander_number + 1))[:, None]
    middle_points = scale_bys * middle_points

    # ── Top and bottom deviation (L222-227) ────────────────────────────
    side_shift_vecs = np.array([sideways * length_perp] * len(middle_points))
    asymmetry_vecs = np.array([sideways * asymmetry] * len(middle_points))
    root_pts = middle_points + asymmetry_vecs
    top_pts = root_pts + side_shift_vecs
    bot_pts = root_pts - side_shift_vecs

    # ── Interleave into meander path (L237-249) ────────────────────────
    # This is the exact qiskit-metal interleaving algorithm
    pts = np.zeros((len(top_pts) + len(bot_pts) + 1 - 2, 2))
    pts[-1, :] = root_pts[-1, :]

    idx_side1_meander, odd = _get_index_for_side1_meander(len(root_pts))
    idx_side2_meander = 2 + idx_side1_meander[: None if odd else -2]

    if first_meander_sideways:
        pts[idx_side1_meander, :] = top_pts[: -1 if odd else None]
        pts[idx_side2_meander, :] = bot_pts[1 : None if odd else -1]
    else:
        pts[idx_side1_meander, :] = bot_pts[: -1 if odd else None]
        pts[idx_side2_meander, :] = top_pts[1 : None if odd else -1]

    # Move to start position (L249)
    pts += start_pos

    # ── Snap adjustments (L251-281) ────────────────────────────────────
    if snap:
        fwd_idx = int(abs(forward[0]))  # 0 if forward is x-aligned, 1 if y-aligned
        # Align last root point's forward coord with end position
        if np.dot(start_dir, end_dir) >= 0 or np.dot(forward, start_dir) > 0:
            pts[-1, fwd_idx] = end_pos[fwd_idx]

    return pts


# ── Public API ─────────────────────────────────────────────────────────


def make_resonator(
    total_length: float,
    coupling_length: float,
    start_pos: tuple,
    start_direction: tuple,
    end_pos: tuple | None = None,
    end_direction: tuple | None = None,
    trace_width: float = 11.7,
    trace_gap: float = 5.1,
    second_width: float = 11.7,
    second_gap: float = 5.1,
    prime_width: float = 11.7,
    prime_gap: float = 5.1,
    coupling_space: float = 7.9,
    down_length: float = 50.0,
    open_termination: bool = False,
    mirror: bool = False,
    spacing: float = 100.0,
    asymmetry: float | None = None,
    fillet: float = 49.9,
    start_straight: float = 50.0,
    end_straight: float = 50.0,
    jog_extension: list[tuple[str, float]] | None = None,
    clt_pos_x: float = 0.0,
    clt_pos_y: float = 0.0,
    clt_orientation: float = -90.0,
) -> dict:
    """Generate Shapely polygons for a complete CPW resonator.

    Includes the meandered CPW path **and** the CoupledLineTee's
    ``second_cpw`` coupling arm (L-shaped segment running parallel to
    the feedline then curving down).

    All dimensions in **micrometres**.

    Args:
        total_length: Target total resonator length.
        coupling_length: CLT coupling_length parameter.
        start_pos: ``(x, y)`` of the claw pin (meander start).
        start_direction: Unit direction vector at the claw pin.
        end_pos: ``(x, y)`` of the CLT second_end pin. If None, auto-computed.
        end_direction: Direction at the CLT pin. If None, auto-computed.
        trace_width: CPW trace width for the meander.
        trace_gap: CPW gap width for the meander.
        second_width: CLT second CPW trace width.
        second_gap: CLT second CPW gap width.
        prime_width: CLT prime CPW trace width (feedline).
        prime_gap: CLT prime CPW gap width (feedline).
        coupling_space: Ground gap between resonator and feedline CPWs.
        down_length: Length of the CLT coupling arm's vertical segment.
        open_termination: Whether the CLT termination is open.
        mirror: Whether the CLT layout is mirrored.
        spacing: Meander curve spacing.
        asymmetry: Meander asymmetry. If None, computed from coupling_length.
        fillet: Corner rounding radius (49.9μm matches SQuADDS default).
        start_straight: Lead-in straight length.
        end_straight: Lead-out straight length.
        jog_extension: Lead-in jog extension. If None, computed from coupling_length.
            Format: [("R90", length_um), ...]
        clt_pos_x: CoupledLineTee centre X.
        clt_pos_y: CoupledLineTee centre Y.
        clt_orientation: CoupledLineTee orientation in degrees.

    Returns:
        Dictionary with geometry and metadata.
    """
    # ── Compute defaults ───────────────────────────────────────────────
    adj_distance = coupling_length if coupling_length > 150 else 0.0

    if asymmetry is None:
        asymmetry = adj_distance / 3.0

    if jog_extension is None and adj_distance > 0:
        jog_extension = [("R90", adj_distance / 1.5)]

    # ── Generate CLT second_cpw ────────────────────────────────────────
    clt_arm = _make_clt_second_cpw(
        coupling_length=coupling_length,
        down_length=down_length,
        prime_width=prime_width,
        prime_gap=prime_gap,
        second_width=second_width,
        second_gap=second_gap,
        coupling_space=coupling_space,
        open_termination=open_termination,
        mirror=mirror,
        pos_x=clt_pos_x,
        pos_y=clt_pos_y,
        orientation=clt_orientation,
    )

    # The meander end is the CLT second_end pin
    if end_pos is None:
        end_pos = tuple(clt_arm["pin_second_end"])
    if end_direction is None:
        end_direction = tuple(clt_arm["pin_second_end_direction"])

    start_pos_arr = np.array(start_pos, dtype=float)
    start_dir_arr = np.array(start_direction, dtype=float)
    end_pos_arr = np.array(end_pos, dtype=float)
    end_dir_arr = np.array(end_direction, dtype=float)

    # ── Build lead-in (from claw pin) ──────────────────────────────────
    head_pts, head_dir, head_length = _build_lead(start_pos_arr, start_dir_arr, start_straight, jog_extension)

    # ── Build lead-out (from CLT second_end pin) ───────────────────────
    tail_pts, tail_dir, tail_length = _build_lead(end_pos_arr, end_dir_arr, end_straight)

    # ── Meander computation ────────────────────────────────────────────
    meander_start = head_pts[-1]
    meander_start_dir = head_dir
    meander_end = tail_pts[-1]
    meander_end_dir = tail_dir

    # Account for CLT arm length in total budget
    clt_arm_length = clt_arm["centerline"].length
    length_for_meander = total_length - head_length - tail_length - clt_arm_length

    meander_pts = _connect_meandered(
        start_pos=meander_start,
        start_dir=meander_start_dir,
        end_pos=meander_end,
        end_dir=meander_end_dir,
        length_meander=max(length_for_meander, 0),
        spacing=spacing,
        asymmetry=asymmetry,
        snap=True,
    )

    # ── Assemble full path: head + meander + tail(reversed) + CLT arm ──
    # The CLT arm centerline is appended so the L-shaped corner also
    # gets the fillet treatment (smooth bend instead of sharp 90°).
    path_segments = [head_pts]
    if len(meander_pts) > 0:
        path_segments.append(meander_pts)
    path_segments.append(tail_pts[::-1])  # tail is reversed (like QRoute)

    # Append CLT arm centerline (in world coordinates, from pin → coupling start)
    # The arm centerline is stored as a Shapely LineString in world coords.
    # We traverse it in reverse (from pin end → far end) since the meander
    # path arrives at the pin_second_end.
    clt_cl_coords = np.array(clt_arm["centerline"].coords)
    # The pin_second_end is the last point of the CLT arm.  We need to
    # walk from there backward through the L to the coupling start.
    clt_cl_reversed = clt_cl_coords[::-1]
    path_segments.append(clt_cl_reversed)

    all_pts = np.concatenate(path_segments, axis=0)

    # Remove consecutive duplicates
    mask = [True]
    for i in range(1, len(all_pts)):
        if norm(all_pts[i] - all_pts[i - 1]) > 1e-6:
            mask.append(True)
        else:
            mask.append(False)
    all_pts = all_pts[mask]

    # ── Apply fillet (corner rounding) ─────────────────────────────────
    # This now covers ALL corners: meander bends, jogs, AND the CLT
    # L-shaped coupling arm corner.
    filleted_pts = _fillet_path(all_pts, radius=fillet)

    # ── Create single unified trace polygon ────────────────────────────
    if len(filleted_pts) >= 2:
        full_line = LineString(filleted_pts)
        trace: Polygon = full_line.buffer(trace_width / 2, cap_style="flat", join_style="round")
        etch: Polygon = full_line.buffer((trace_width + 2 * trace_gap) / 2, cap_style="flat", join_style="round")
        centerline = full_line
    else:
        trace = Polygon()
        etch = Polygon()
        centerline = LineString()

    path_length = centerline.length if not centerline.is_empty else 0.0

    return {
        "trace": trace,
        "etch": etch,
        "centerline": centerline,
        "meander_points": all_pts,
        "filleted_points": filleted_pts,
        "clt_arm": clt_arm,
        "position": (trace.centroid.x, trace.centroid.y) if not trace.is_empty else (0.0, 0.0),
        "start_pos": tuple(start_pos_arr),
        "end_pos": tuple(end_pos_arr),
        "path_length": path_length,
        "params": {
            "total_length": total_length,
            "coupling_length": coupling_length,
            "trace_width": trace_width,
            "trace_gap": trace_gap,
            "spacing": spacing,
            "fillet": fillet,
        },
    }
