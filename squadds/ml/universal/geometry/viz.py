"""Matplotlib visualization helpers for quantum circuit layouts.

Provides functions to plot individual components and full 4-component
layouts with labelled pins, component boundaries, and positioning info.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from shapely.geometry import LineString, MultiPolygon, Polygon

# ── Shapely → Matplotlib conversion ───────────────────────────────────


def _shapely_to_patch(polygon: Polygon | MultiPolygon, **kwargs) -> list[PathPatch]:
    """Convert a Shapely polygon (or MultiPolygon) to Matplotlib PathPatch(es)."""
    patches = []
    if isinstance(polygon, MultiPolygon):
        geoms = list(polygon.geoms)
    elif isinstance(polygon, Polygon):
        geoms = [polygon]
    else:
        return patches

    for geom in geoms:
        if geom.is_empty:
            continue
        exterior = np.array(geom.exterior.coords)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(exterior) - 2) + [Path.CLOSEPOLY]
        vertices = exterior

        # Handle holes
        for interior in geom.interiors:
            hole = np.array(interior.coords)
            hole_codes = [Path.MOVETO] + [Path.LINETO] * (len(hole) - 2) + [Path.CLOSEPOLY]
            vertices = np.concatenate([vertices, hole])
            codes.extend(hole_codes)

        path = Path(vertices, codes)
        patches.append(PathPatch(path, **kwargs))

    return patches


# ── Color palette ─────────────────────────────────────────────────────

COMPONENT_COLORS = {
    "qubit": {"metal": "#4A90D9", "etch": "#A8CCE8", "label": "#2C5F8A"},
    "claw": {"metal": "#E8833A", "etch": "#F4C89A", "label": "#A85A20"},
    "resonator": {"metal": "#50B86C", "etch": "#A8DDB5", "label": "#2D7A3E"},
    "feedline": {"metal": "#9B59B6", "etch": "#D7B4E8", "label": "#6C3483"},
}


# ── Public API ─────────────────────────────────────────────────────────


def plot_component(
    component: dict,
    name: str = "component",
    ax: plt.Axes | None = None,
    show_pins: bool = True,
    show_etch: bool = True,
    color_scheme: dict | None = None,
) -> plt.Figure:
    """Plot a single component (output of any ``make_*`` function).

    Args:
        component: Output dict from ``make_transmon_cross``, ``make_claw``,
            ``make_feedline``, or ``make_resonator``.
        name: Component name (used for colour lookup and labelling).
        ax: Optional axes to plot on.
        show_pins: Whether to annotate pin positions.
        show_etch: Whether to show the etch/gap region.
        color_scheme: Override colours ``{"metal": ..., "etch": ...}``.

    Returns:
        The Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    colors = color_scheme or COMPONENT_COLORS.get(name, COMPONENT_COLORS["qubit"])

    # ── Determine which keys hold geometry ─────────────────────────────
    metal_key = None
    etch_key = None
    for k in ("cross", "arm", "trace"):
        if k in component and isinstance(component[k], (Polygon, MultiPolygon)):
            metal_key = k
            break
    for k in ("cross_etch", "etch"):
        if k in component and isinstance(component[k], (Polygon, MultiPolygon)):
            etch_key = k
            break

    # Plot etch first (behind metal)
    if show_etch and etch_key:
        for patch in _shapely_to_patch(component[etch_key], facecolor=colors["etch"], edgecolor="none", alpha=0.5):
            ax.add_patch(patch)

    # Plot metal
    if metal_key:
        for patch in _shapely_to_patch(
            component[metal_key],
            facecolor=colors["metal"],
            edgecolor=colors.get("label", "#333"),
            linewidth=0.8,
            alpha=0.85,
        ):
            ax.add_patch(patch)

    # Plot centerline if present (resonator/feedline)
    if "centerline" in component:
        cl = component["centerline"]
        if isinstance(cl, LineString) and not cl.is_empty:
            coords = np.array(cl.coords)
            ax.plot(coords[:, 0], coords[:, 1], ":", color=colors.get("label", "#333"), linewidth=0.6, alpha=0.5)

    # Plot JJ line if present (qubit)
    if "jj" in component:
        jj = component["jj"]
        if isinstance(jj, LineString):
            coords = np.array(jj.coords)
            ax.plot(coords[:, 0], coords[:, 1], "-", color="#E74C3C", linewidth=2.5, label="JJ", zorder=5)

    # ── Annotate pins ──────────────────────────────────────────────────
    if show_pins:
        # Handle different pin formats
        if "pins" in component:
            for pin_name, pos in component["pins"].items():
                ax.plot(*pos, "o", color=colors.get("label", "#333"), markersize=5, zorder=6)
                ax.annotate(
                    pin_name,
                    pos,
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=7,
                    color=colors.get("label", "#333"),
                )
        if "pin_position" in component:
            pos = component["pin_position"]
            ax.plot(*pos, "s", color=colors.get("label", "#333"), markersize=6, zorder=6)
            ax.annotate("pin", pos, textcoords="offset points", xytext=(5, 5), fontsize=7)

    # ── Position marker ────────────────────────────────────────────────
    if "position" in component:
        pos = component["position"]
        ax.plot(*pos, "+", color=colors.get("label", "#333"), markersize=10, markeredgewidth=1.5, zorder=7)

    ax.set_aspect("equal")
    ax.set_title(name.replace("_", " ").title(), fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)

    return fig


def plot_layout(
    layout: dict,
    ax: plt.Axes | None = None,
    show_labels: bool = True,
    show_pins: bool = True,
    show_etch: bool = True,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """Plot a full 4-component layout.

    Args:
        layout: Output dict from :func:`build_layout`.
        ax: Optional axes.  If ``None``, a new figure is created.
        show_labels: Annotate component names.
        show_pins: Show pin positions.
        show_etch: Show etch/gap polygons.
        figsize: Figure size.

    Returns:
        The Matplotlib figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    component_order = ["feedline", "resonator", "claw", "qubit"]

    for comp_name in component_order:
        if comp_name not in layout:
            continue
        comp = layout[comp_name]
        colors = COMPONENT_COLORS.get(comp_name, COMPONENT_COLORS["qubit"])

        # ── Etch ───────────────────────────────────────────────────────
        if show_etch:
            for k in ("cross_etch", "etch"):
                if k in comp and isinstance(comp[k], (Polygon, MultiPolygon)):
                    for patch in _shapely_to_patch(comp[k], facecolor=colors["etch"], edgecolor="none", alpha=0.35):
                        ax.add_patch(patch)
                    break

        # ── Metal ──────────────────────────────────────────────────────
        for k in ("cross", "arm", "trace"):
            if k in comp and isinstance(comp[k], (Polygon, MultiPolygon)):
                for patch in _shapely_to_patch(
                    comp[k],
                    facecolor=colors["metal"],
                    edgecolor=colors["label"],
                    linewidth=0.6,
                    alpha=0.85,
                    label=comp_name.title(),
                ):
                    ax.add_patch(patch)
                break

        # ── JJ ─────────────────────────────────────────────────────────
        if "jj" in comp:
            jj = comp["jj"]
            if isinstance(jj, LineString):
                coords = np.array(jj.coords)
                ax.plot(coords[:, 0], coords[:, 1], "-", color="#E74C3C", linewidth=2.5, zorder=5)

        # ── Centerline ─────────────────────────────────────────────────
        if "centerline" in comp:
            cl = comp["centerline"]
            if isinstance(cl, LineString) and not cl.is_empty:
                coords = np.array(cl.coords)
                ax.plot(coords[:, 0], coords[:, 1], ":", color=colors["label"], linewidth=0.5, alpha=0.4)

        # ── Pins ───────────────────────────────────────────────────────
        if show_pins:
            if "pins" in comp:
                for pin_name, pos in comp["pins"].items():
                    ax.plot(*pos, "o", color=colors["label"], markersize=4, zorder=6)
                    if show_labels:
                        ax.annotate(
                            f"{comp_name}.{pin_name}",
                            pos,
                            textcoords="offset points",
                            xytext=(4, 4),
                            fontsize=5,
                            color=colors["label"],
                            alpha=0.7,
                        )
            if "pin_position" in comp:
                pos = comp["pin_position"]
                ax.plot(*pos, "s", color=colors["label"], markersize=5, zorder=6)

        # ── Component label ────────────────────────────────────────────
        if show_labels and "position" in comp:
            pos = comp["position"]
            ax.annotate(
                comp_name.upper(),
                pos,
                fontsize=8,
                fontweight="bold",
                color=colors["label"],
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor=colors["label"]),
                zorder=8,
            )

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xlabel("x (μm)", fontsize=10)
    ax.set_ylabel("y (μm)", fontsize=10)
    ax.set_title("Quantum Circuit Layout", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.15)

    fig.tight_layout()
    return fig
