"""Standalone Shapely polygon generator for the TransmonCross qubit.

Faithfully reproduces the geometry from
``qiskit_metal.qlibrary.qubits.transmon_cross.TransmonCross.make_pocket()``.

The cross is formed by two perpendicular line segments of length
``2 * cross_length``, buffered to width ``cross_width / 2``, unioned into a
single island.  The etch region is a further buffer of ``cross_gap``.  The
Josephson junction is a short line segment extending from the south arm.
"""

from __future__ import annotations

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def make_transmon_cross(
    cross_length: float,
    cross_gap: float,
    cross_width: float = 20.0,
    pos_x: float = 0.0,
    pos_y: float = 0.0,
    orientation: float = 0.0,
) -> dict:
    """Generate Shapely polygons for a TransmonCross qubit.

    All dimensions are in **micrometres**.

    Args:
        cross_length: Length of one arm measured from the centre.
        cross_gap: CPW gap surrounding the cross island.
        cross_width: Width of the CPW trace making up the cross.
        pos_x: X-coordinate of the cross centre.
        pos_y: Y-coordinate of the cross centre.
        orientation: Counter-clockwise rotation in degrees.

    Returns:
        Dictionary with keys:

        * ``cross`` – :class:`Polygon`, the metal island.
        * ``cross_etch`` – :class:`Polygon`, the gap/etch region.
        * ``jj`` – :class:`LineString`, the Josephson-junction line.
        * ``position`` – ``(pos_x, pos_y)`` tuple.
        * ``orientation`` – rotation angle in degrees.
        * ``pins`` – dict mapping cardinal directions to ``(x, y)`` tip
          coordinates (after rotation/translation).
    """
    # ── Build cross at origin ──────────────────────────────────────────
    vertical = LineString([(0, cross_length), (0, -cross_length)])
    horizontal = LineString([(cross_length, 0), (-cross_length, 0)])
    cross_line = unary_union([vertical, horizontal])

    cross: Polygon = cross_line.buffer(cross_width / 2, cap_style="square")
    cross_etch: Polygon = cross.buffer(cross_gap, cap_style="square", join_style="mitre")

    # Josephson junction – line extending south from the cross island
    jj = LineString([(0, -cross_length), (0, -cross_length - cross_gap)])

    # Pin tip positions (before rotation/translation)
    pins_raw = {
        "N": np.array([0.0, cross_length + cross_gap]),
        "S": np.array([0.0, -(cross_length + cross_gap)]),
        "E": np.array([cross_length + cross_gap, 0.0]),
        "W": np.array([-(cross_length + cross_gap), 0.0]),
    }

    # ── Rotate and translate ───────────────────────────────────────────
    geoms = [cross, cross_etch, jj]
    geoms = [affinity.rotate(g, orientation, origin=(0, 0)) for g in geoms]
    geoms = [affinity.translate(g, pos_x, pos_y) for g in geoms]
    cross, cross_etch, jj = geoms

    # Rotate pin positions
    theta = np.radians(orientation)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pins = {name: tuple(rot @ pos + np.array([pos_x, pos_y])) for name, pos in pins_raw.items()}

    return {
        "cross": cross,
        "cross_etch": cross_etch,
        "jj": jj,
        "position": (pos_x, pos_y),
        "orientation": orientation,
        "pins": pins,
        "params": {
            "cross_length": cross_length,
            "cross_gap": cross_gap,
            "cross_width": cross_width,
        },
    }
