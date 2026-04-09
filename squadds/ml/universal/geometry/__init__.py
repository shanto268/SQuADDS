"""Shapely polygon generators for quantum circuit components.

Each module provides a standalone function that converts parametric design
dictionaries into positioned Shapely polygons — no qiskit-metal dependency.
"""

from squadds.ml.universal.geometry.claw import make_claw
from squadds.ml.universal.geometry.feedline import make_feedline
from squadds.ml.universal.geometry.layout import build_layout
from squadds.ml.universal.geometry.qubit import make_transmon_cross
from squadds.ml.universal.geometry.resonator import make_resonator
from squadds.ml.universal.geometry.viz import plot_component, plot_layout

__all__ = [
    "make_transmon_cross",
    "make_claw",
    "make_feedline",
    "make_resonator",
    "build_layout",
    "plot_layout",
    "plot_component",
]
