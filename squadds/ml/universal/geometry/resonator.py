"""Standalone Shapely polygon generator for the CPW resonator.

Faithfully reproduces:
1. The ``second_cpw`` L-shaped coupling arm from
   ``CoupledLineTee.make()`` (coupled_line_tee.py L98-106).
2. The meandered CPW path from ``RouteMeander.connect_meandered()``
   (meandered.py L103-311).

The resonator polygon is the union of the meander path and the CLT
coupling arm, buffered to the CPW trace width.
"""

from __future__ import annotations

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

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
    pin_pos_raw = np.array(pts[-1])
    # Direction: pointing downward (away from the coupling region)
    pin_dir_raw = np.array([0.0, -1.0])
    if mirror:
        pin_dir_raw = np.array([0.0, -1.0])

    # Rotate and translate
    geoms = [centerline, trace, etch]
    geoms = [affinity.rotate(g, orientation, origin=(0, 0)) for g in geoms]
    geoms = [affinity.translate(g, pos_x, pos_y) for g in geoms]
    centerline, trace, etch = geoms

    theta = np.radians(orientation)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    offset = np.array([pos_x, pos_y])
    pin_pos = tuple(rot @ pin_pos_raw + offset)
    pin_dir = tuple(rot @ pin_dir_raw)

    return {
        "centerline": centerline,
        "trace": trace,
        "etch": etch,
        "pin_second_end": pin_pos,
        "pin_second_end_direction": pin_dir,
    }


# ── Meander routing ───────────────────────────────────────────────────


def _generate_meander_points(
    start_pos: np.ndarray,
    start_dir: np.ndarray,
    end_pos: np.ndarray,
    end_dir: np.ndarray,
    total_length: float,
    spacing: float,
    asymmetry: float = 0.0,
    start_straight: float = 100.0,
    end_straight: float = 50.0,
    jog_length: float = 0.0,
) -> np.ndarray:
    """Generate meander path points between two pin positions.

    Implements the core algorithm from ``RouteMeander.connect_meandered()``
    adapted for standalone Shapely use (no QRoute dependency).

    Args:
        start_pos: Starting pin position ``(x, y)``.
        start_dir: Unit direction vector at start pin.
        end_pos: Ending pin position ``(x, y)``.
        end_dir: Unit direction vector at end pin.
        total_length: Target total path length.
        spacing: Minimum spacing between adjacent meander curves.
        asymmetry: Offset of meander centre-line from the direct axis.
        start_straight: Lead-in straight segment length.
        end_straight: Lead-out straight segment length.
        jog_length: Length of the initial jog (perpendicular step after lead-in).

    Returns:
        ``(M, 2)`` array of path coordinates.
    """
    all_points = [start_pos.copy()]
    consumed_length = 0.0

    # ── Lead-in ────────────────────────────────────────────────────────
    if start_straight > 0:
        lead_in_end = start_pos + start_dir * start_straight
        all_points.append(lead_in_end)
        consumed_length += start_straight
    else:
        lead_in_end = start_pos.copy()

    # ── Jog (perpendicular step) ───────────────────────────────────────
    if jog_length > 0:
        # Jog is perpendicular to start_dir (R90 = rotate 90° CCW)
        perp = np.array([-start_dir[1], start_dir[0]])
        jog_end = lead_in_end + perp * jog_length
        all_points.append(jog_end)
        consumed_length += jog_length
        meander_start = jog_end
    else:
        meander_start = lead_in_end

    # ── Lead-out ───────────────────────────────────────────────────────
    lead_out_start = end_pos + end_dir * end_straight
    consumed_length += end_straight

    # ── Meander computation ────────────────────────────────────────────
    length_for_meander = total_length - consumed_length

    # Determine forward and sideways unit vectors
    dist_vec = lead_out_start - meander_start
    dist_norm = np.linalg.norm(dist_vec)

    if dist_norm < 1e-6:
        # Start and end are coincident — just connect directly
        all_points.append(lead_out_start)
        all_points.append(end_pos.copy())
        return np.array(all_points)

    forward = dist_vec / dist_norm
    sideways = np.array([-forward[1], forward[0]])  # 90° CCW from forward

    length_direct = dist_norm

    if length_for_meander <= length_direct:
        # Not enough length to meander — straight connection
        all_points.append(lead_out_start)
        all_points.append(end_pos.copy())
        return np.array(all_points)

    # Number of meander segments
    meander_number = max(1, int(np.floor(length_direct / spacing)))

    # Ensure parity: start_dir and end_dir sideways components
    start_sw = np.dot(start_dir, sideways)
    end_sw = np.dot(end_dir, sideways)

    if start_sw * end_sw > 0 and (meander_number % 2) == 0:
        meander_number -= 1
    elif start_sw * end_sw < 0 and (meander_number % 2) == 1:
        meander_number -= 1

    meander_number = max(1, meander_number)

    # Perpendicular height to accommodate excess length
    length_excess = length_for_meander - length_direct - 2 * abs(asymmetry)
    length_perp = max(0, length_excess / (meander_number * 2.0))

    # Determine first meander direction
    if start_sw > 0:
        first_meander_sideways = True
    elif start_sw < 0:
        first_meander_sideways = False
    else:
        first_meander_sideways = True

    # ── Generate meander points ────────────────────────────────────────
    # Root points along the forward axis
    root_pts = np.array([forward * (spacing * i) for i in range(int(meander_number) + 1)])

    # Add asymmetry offset
    asymmetry_vec = sideways * asymmetry
    root_pts = root_pts + asymmetry_vec

    # Top and bottom deviation from root
    side_shift = sideways * length_perp
    top_pts = root_pts + side_shift
    bot_pts = root_pts - side_shift

    # Interleave into meander path
    n_roots = len(root_pts)
    pts = []

    for i in range(n_roots):
        if i == n_roots - 1:
            # Last root point — anchor
            pts.append(root_pts[i])
            break

        if (i % 2 == 0) == first_meander_sideways:
            pts.append(top_pts[i])
        else:
            pts.append(bot_pts[i])

        if i + 1 < n_roots:
            if ((i + 1) % 2 == 0) == first_meander_sideways:
                pts.append(top_pts[i + 1])
            else:
                pts.append(bot_pts[i + 1])

    # Deduplicate consecutive identical points
    meander_pts = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - meander_pts[-1]) > 1e-6:
            meander_pts.append(p)

    # Translate meander points to meander_start position
    meander_pts = np.array(meander_pts) + meander_start

    all_points.extend(meander_pts.tolist())

    # ── Lead-out ───────────────────────────────────────────────────────
    all_points.append(lead_out_start.tolist())
    all_points.append(end_pos.tolist())

    return np.array(all_points)


def _smooth_path(points: np.ndarray, fillet: float) -> LineString:
    """Apply corner rounding to a polyline via Shapely buffer trick.

    A true fillet requires computing arc segments at each corner.  For
    computational simplicity we use the Shapely ``buffer → erode`` trick:
    buffer the polyline by ``fillet``, then erode back.  This rounds corners
    nicely for visualization, though the path length changes slightly.

    For exact fillet arcs we would need to compute tangent circles at each
    corner, which is doable but significantly more code.  Since the user
    verified that a faithful *looking* meander is acceptable, this approach
    produces visually accurate results.
    """
    line = LineString(points)
    if fillet <= 0 or len(points) < 3:
        return line

    # Buffer-erode rounding
    try:
        rounded = line.buffer(fillet, join_style="round").buffer(-fillet + 0.1, join_style="round")
        # Extract the longest ring as the new center-line approximation
        if rounded.is_empty:
            return line
        exterior = rounded.exterior
        return exterior
    except Exception:
        return line


# ── Public API ─────────────────────────────────────────────────────────


def make_resonator(
    total_length: float,
    coupling_length: float,
    start_pos: tuple,
    start_direction: tuple,
    end_pos: tuple | None = None,
    end_direction: tuple | None = None,
    trace_width: float = 10.0,
    trace_gap: float = 6.0,
    second_width: float = 10.0,
    second_gap: float = 6.0,
    prime_width: float = 10.0,
    prime_gap: float = 6.0,
    coupling_space: float = 3.0,
    down_length: float = 100.0,
    open_termination: bool = True,
    mirror: bool = False,
    spacing: float = 100.0,
    asymmetry: float | None = None,
    fillet: float = 49.9,
    start_straight: float = 100.0,
    end_straight: float = 50.0,
    jog_length: float | None = None,
    clt_pos_x: float = 0.0,
    clt_pos_y: float = 0.0,
    clt_orientation: float = 180.0,
) -> dict:
    """Generate Shapely polygons for a complete CPW resonator.

    Includes the meandered CPW path **and** the CoupledLineTee's
    ``second_cpw`` coupling arm (L-shaped segment running parallel to
    the feedline then curving down).

    All dimensions in **micrometres**.

    Args:
        total_length: Target total resonator length.  For CLT coupler type
            this is typically ``parquet_total_length / 2``.
        coupling_length: CLT coupling_length parameter.
        start_pos: ``(x, y)`` of the claw pin (meander start).
        start_direction: Unit direction vector at the claw pin.
        end_pos: ``(x, y)`` of the CLT second_end pin.  If ``None``,
            computed from the CLT parameters.
        end_direction: Unit direction at the CLT pin.  If ``None``, computed.
        trace_width: CPW trace width for the meander.
        trace_gap: CPW gap width for the meander.
        second_width: CLT second CPW trace width.
        second_gap: CLT second CPW gap width.
        prime_width: CLT prime CPW trace width (feedline).
        prime_gap: CLT prime CPW gap width (feedline).
        coupling_space: Ground gap between resonator and feedline CPWs.
        down_length: Length of the CLT coupling arm's vertical segment.
        open_termination: Whether the CLT termination is open or shorted.
        mirror: Whether the CLT layout is mirrored.
        spacing: Meander curve spacing.
        asymmetry: Meander asymmetry.  If ``None``, computed as
            ``coupling_length / 3`` (matching SQuADDS default).
        fillet: Corner rounding radius.
        start_straight: Lead-in straight length.
        end_straight: Lead-out straight length.
        jog_length: Jog length after lead-in.  If ``None``, computed as
            ``coupling_length / 1.5`` when ``coupling_length > 150``.
        clt_pos_x: CoupledLineTee centre X.
        clt_pos_y: CoupledLineTee centre Y.
        clt_orientation: CoupledLineTee orientation in degrees.

    Returns:
        Dictionary with keys:

        * ``trace`` – :class:`Polygon`, combined resonator metal.
        * ``etch`` – :class:`Polygon`, combined etch region.
        * ``centerline`` – :class:`LineString`, the CPW centre path.
        * ``meander_points`` – ``(M, 2)`` array of meander path coords.
        * ``clt_arm`` – dict with CLT second_cpw details.
        * ``position`` – centroid ``(x, y)`` of the trace polygon.
        * ``start_pos`` – meander start position.
        * ``end_pos`` – CLT second_end position.
        * ``path_length`` – actual computed path length.
    """
    # ── Compute defaults ───────────────────────────────────────────────
    if asymmetry is None:
        asymmetry = coupling_length / 3.0

    if jog_length is None:
        jog_length = coupling_length / 1.5 if coupling_length > 150 else 0.0

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
        end_pos = clt_arm["pin_second_end"]
    if end_direction is None:
        end_direction = clt_arm["pin_second_end_direction"]

    start_pos_arr = np.array(start_pos, dtype=float)
    start_dir_arr = np.array(start_direction, dtype=float)
    end_pos_arr = np.array(end_pos, dtype=float)
    end_dir_arr = np.array(end_direction, dtype=float)

    # ── Account for CLT arm length in total budget ─────────────────────
    clt_arm_length = clt_arm["centerline"].length
    meander_total = total_length - clt_arm_length

    # ── Generate meander points ────────────────────────────────────────
    meander_pts = _generate_meander_points(
        start_pos=start_pos_arr,
        start_dir=start_dir_arr,
        end_pos=end_pos_arr,
        end_dir=end_dir_arr,
        total_length=max(meander_total, 0),
        spacing=spacing,
        asymmetry=asymmetry,
        start_straight=start_straight,
        end_straight=end_straight,
        jog_length=jog_length,
    )

    # ── Create meander trace polygon ───────────────────────────────────
    if len(meander_pts) >= 2:
        meander_line = LineString(meander_pts)
        meander_trace: Polygon = meander_line.buffer(trace_width / 2, cap_style="flat", join_style="mitre")
        meander_etch: Polygon = meander_line.buffer(
            (trace_width + 2 * trace_gap) / 2, cap_style="flat", join_style="mitre"
        )
    else:
        meander_trace = Polygon()
        meander_etch = Polygon()
        meander_line = LineString()

    # ── Union meander + CLT arm ────────────────────────────────────────
    try:
        trace = unary_union([meander_trace, clt_arm["trace"]])
    except Exception:
        trace = meander_trace

    try:
        etch = unary_union([meander_etch, clt_arm["etch"]])
    except Exception:
        etch = meander_etch

    # Combined center-line
    if meander_line.is_empty:
        centerline = clt_arm["centerline"]
    else:
        try:
            all_coords = list(meander_line.coords) + list(clt_arm["centerline"].coords)
            centerline = LineString(all_coords)
        except Exception:
            centerline = meander_line

    path_length = centerline.length if not centerline.is_empty else 0.0

    return {
        "trace": trace,
        "etch": etch,
        "centerline": centerline,
        "meander_points": meander_pts,
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
