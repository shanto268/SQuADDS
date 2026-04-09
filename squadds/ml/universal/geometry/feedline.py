"""Standalone Shapely polygon generator for the feedline (CoupledLineTee prime CPW).

Faithfully reproduces the ``prime_cpw`` portion of
``qiskit_metal.qlibrary.couplers.coupled_line_tee.CoupledLineTee.make()``.

The feedline is a horizontal CPW line of length ``2 * coupling_length``
centred at ``(pos_x, pos_y)``.
"""

from __future__ import annotations

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon


def make_feedline(
    coupling_length: float,
    prime_width: float = 10.0,
    prime_gap: float = 6.0,
    pos_x: float = 0.0,
    pos_y: float = 0.0,
    orientation: float = 180.0,
) -> dict:
    """Generate Shapely polygons for a feedline (CoupledLineTee prime CPW).

    All dimensions in **micrometres**.

    Args:
        coupling_length: Half-length of the feedline segment (the CLT
            ``coupling_length`` parameter).  Total feedline length is
            ``2 * coupling_length``.
        prime_width: Trace width of the feedline CPW.
        prime_gap: Gap width of the feedline CPW.
        pos_x: Centre X-coordinate of the CoupledLineTee.
        pos_y: Centre Y-coordinate of the CoupledLineTee.
        orientation: Rotation in degrees (default 180° to match SQuADDS).

    Returns:
        Dictionary with keys:

        * ``trace`` – :class:`Polygon`, feedline metal trace.
        * ``etch`` – :class:`Polygon`, feedline etch/gap region.
        * ``centerline`` – :class:`LineString`, the CPW centre path.
        * ``position`` – ``(pos_x, pos_y)``.
        * ``orientation`` – rotation angle in degrees.
        * ``pins`` – ``{"prime_start": (x,y), "prime_end": (x,y)}``.
        * ``pin_directions`` – unit direction vectors at each pin.
    """
    prime_cpw_length = coupling_length * 2

    # ── Build at origin ────────────────────────────────────────────────
    centerline = LineString([[-prime_cpw_length / 2, 0], [prime_cpw_length / 2, 0]])
    trace: Polygon = centerline.buffer(prime_width / 2, cap_style="flat")
    etch: Polygon = centerline.buffer((prime_width + 2 * prime_gap) / 2, cap_style="flat")

    # Pin positions (before rotation)
    pin_start_raw = np.array([-prime_cpw_length / 2, 0.0])
    pin_end_raw = np.array([prime_cpw_length / 2, 0.0])

    # Pin directions: start points left (-x), end points right (+x)
    dir_start_raw = np.array([-1.0, 0.0])
    dir_end_raw = np.array([1.0, 0.0])

    # ── Rotate and translate ───────────────────────────────────────────
    geoms = [trace, etch, centerline]
    geoms = [affinity.rotate(g, orientation, origin=(0, 0)) for g in geoms]
    geoms = [affinity.translate(g, pos_x, pos_y) for g in geoms]
    trace, etch, centerline = geoms

    theta = np.radians(orientation)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    offset = np.array([pos_x, pos_y])

    pin_start = tuple(rot @ pin_start_raw + offset)
    pin_end = tuple(rot @ pin_end_raw + offset)
    dir_start = tuple(rot @ dir_start_raw)
    dir_end = tuple(rot @ dir_end_raw)

    return {
        "trace": trace,
        "etch": etch,
        "centerline": centerline,
        "position": (pos_x, pos_y),
        "orientation": orientation,
        "pins": {"prime_start": pin_start, "prime_end": pin_end},
        "pin_directions": {"prime_start": dir_start, "prime_end": dir_end},
        "params": {
            "coupling_length": coupling_length,
            "prime_width": prime_width,
            "prime_gap": prime_gap,
        },
    }
