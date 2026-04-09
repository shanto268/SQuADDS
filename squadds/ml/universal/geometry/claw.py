"""Standalone Shapely polygon generator for the claw coupler.

Faithfully reproduces the geometry from
``qiskit_metal.qlibrary.qubits.transmon_cross.TransmonCross.make_connection_pad()``.

The claw is a U-shaped metal structure (``connector_arm``) with a short CPW
stub (``claw_cpw``) connecting it to the qubit arm.  The etch region is a
uniform buffer around the arm.  The claw is placed relative to the qubit
centre at a distance determined by the cross arm length, gap, ground spacing,
and claw gap.
"""

from __future__ import annotations

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union


def make_claw(
    claw_length: float,
    ground_spacing: float,
    cross_length: float,
    cross_gap: float,
    cross_width: float = 20.0,
    claw_width: float = 10.0,
    claw_gap: float = 6.0,
    claw_cpw_length: float = 40.0,
    claw_cpw_width: float = 10.0,
    connector_location: int = 0,
    pos_x: float = 0.0,
    pos_y: float = 0.0,
    orientation: float = 0.0,
) -> dict:
    """Generate Shapely polygons for a claw coupler.

    The claw is positioned relative to the **qubit centre** at
    ``(pos_x, pos_y)`` — the same coordinate used for
    :func:`make_transmon_cross`.

    All dimensions in **micrometres**.

    Args:
        claw_length: Length of the claw "arms" (U-shape depth).
        ground_spacing: Ground-plane gap between claw and cross arm.
        cross_length: Qubit arm length (needed for radial offset).
        cross_gap: Qubit CPW gap (needed for claw height).
        cross_width: Qubit trace width (needed for claw height).
        claw_width: Width of the CPW trace making up the claw.
        claw_gap: Gap of the CPW trace making up the claw.
        claw_cpw_length: Length of the short CPW stub connecting claw to qubit.
        claw_cpw_width: Width of that CPW stub.
        connector_location: Which arm to attach to.
            ``0`` → west, ``90`` → north, ``180`` → east.
        pos_x: Qubit centre X (claw placed relative to this).
        pos_y: Qubit centre Y.
        orientation: Qubit orientation in degrees.

    Returns:
        Dictionary with keys:

        * ``arm`` – :class:`Polygon`, the U-shaped metal.
        * ``etch`` – :class:`Polygon`, the gap/etch region.
        * ``pin`` – :class:`LineString`, the connection pin at the CPW tip.
        * ``position`` – ``(cx, cy)`` centroid of the arm polygon.
        * ``orientation`` – combined rotation angle.
        * ``pin_position`` – ``(x, y)`` of the pin endpoint.
    """
    # ── Claw geometry at origin, facing west ───────────────────────────
    # Short CPW stub
    claw_cpw = box(-claw_width, -claw_cpw_width / 2, -claw_cpw_length - claw_width, claw_cpw_width / 2)

    # Total claw height (matches transmon_cross.py L187-188)
    t_claw_height = 2 * claw_gap + 2 * claw_width + 2 * ground_spacing + 2 * cross_gap + cross_width

    # U-shaped claw base
    claw_base = box(-claw_width, -t_claw_height / 2, claw_length, t_claw_height / 2)
    claw_subtract = box(0, -t_claw_height / 2 + claw_width, claw_length, t_claw_height / 2 - claw_width)
    claw_base = claw_base.difference(claw_subtract)

    connector_arm: Polygon = unary_union([claw_base, claw_cpw])
    connector_etch: Polygon = connector_arm.buffer(claw_gap)

    # Pin line at the CPW tip
    pin_line = LineString(
        [(-claw_cpw_length - claw_width, -claw_cpw_width / 2), (-claw_cpw_length - claw_width, claw_cpw_width / 2)]
    )

    # ── Position relative to qubit centre ──────────────────────────────
    # Offset = distance from qubit centre to the claw (matches L217)
    offset = -(cross_length + cross_gap + ground_spacing + claw_gap)
    geoms = [connector_arm, connector_etch, pin_line]
    geoms = [affinity.translate(g, offset, 0) for g in geoms]

    # Rotate for connector_location (0=west already, 90=north, 180=east)
    claw_rotate = 0
    if connector_location > 135:
        claw_rotate = 180
    elif connector_location > 45:
        claw_rotate = -90

    geoms = [affinity.rotate(g, claw_rotate, origin=(0, 0)) for g in geoms]

    # Apply qubit orientation + translation
    geoms = [affinity.rotate(g, orientation, origin=(0, 0)) for g in geoms]
    geoms = [affinity.translate(g, pos_x, pos_y) for g in geoms]
    connector_arm, connector_etch, pin_line = geoms

    # Extract pin endpoint (the tip of the CPW stub, away from the qubit)
    pin_coords = list(pin_line.coords)
    pin_position = ((pin_coords[0][0] + pin_coords[1][0]) / 2, (pin_coords[0][1] + pin_coords[1][1]) / 2)

    # Pin direction: points away from qubit (outward along the CPW stub)
    # In the un-rotated frame, the pin points in the -x direction
    total_rot = orientation + claw_rotate
    theta = np.radians(total_rot)
    base_dir = np.array([-1.0, 0.0])
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pin_direction = tuple(rot @ base_dir)

    return {
        "arm": connector_arm,
        "etch": connector_etch,
        "pin": pin_line,
        "position": (connector_arm.centroid.x, connector_arm.centroid.y),
        "orientation": orientation + claw_rotate,
        "pin_position": pin_position,
        "pin_direction": pin_direction,
        "params": {
            "claw_length": claw_length,
            "ground_spacing": ground_spacing,
            "claw_width": claw_width,
            "claw_gap": claw_gap,
            "claw_cpw_length": claw_cpw_length,
            "claw_cpw_width": claw_cpw_width,
        },
    }
