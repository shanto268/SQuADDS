"""Composable layout builder for arbitrary circuit topologies.

Builds layouts from the 4 atomic building blocks (TransmonCross, Claw,
RouteMeander, CoupledLineTee) in arbitrary combinations and arrangements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from squadds.ml.universal.geometry.claw import make_claw
from squadds.ml.universal.geometry.feedline import make_feedline
from squadds.ml.universal.geometry.qubit import make_transmon_cross
from squadds.ml.universal.geometry.resonator import make_resonator


@dataclass
class PlacedComponent:
    """A component with position and orientation."""

    name: str
    component_type: str  # "TransmonCross", "Claw", "RouteMeander", "CoupledLineTee"
    params: dict = field(default_factory=dict)
    pos_x: float = 0.0
    pos_y: float = 0.0
    orientation: float = 0.0
    # For resonators: which components to connect between
    connect_from: str | None = None
    connect_to: str | None = None


def build_composite_layout(
    components: list[PlacedComponent],
    default_params: dict | None = None,
) -> dict:
    """Build a layout from an arbitrary list of placed components.

    This generalizes ``build_layout`` to support any combination of the
    4 building blocks.  Each component is built independently and placed
    at its specified position.

    Args:
        components: List of PlacedComponent specifications.
        default_params: Default design parameters applied to all components.

    Returns:
        Dictionary keyed by component name, each containing the geometry
        output dict.  Also includes ``design_params`` with merged params.
    """
    defaults = {
        "cross_length": 310.0,
        "cross_gap": 30.0,
        "cross_width": 30.0,
        "claw_length": 160.0,
        "claw_width": 15.0,
        "claw_gap": 5.1,
        "ground_spacing": 10.0,
        "coupling_length": 200.0,
        "total_length": 4700.0,
        "prime_width": 11.7,
        "prime_gap": 5.1,
        "second_width": 11.7,
        "second_gap": 5.1,
        "coupling_space": 7.9,
        "down_length": 50.0,
        "spacing": 100.0,
        "fillet": 49.9,
        "trace_width": 11.7,
        "trace_gap": 5.1,
        "start_straight": 50.0,
        "end_straight": 50.0,
        "connector_location": 90,
    }
    if default_params:
        defaults.update(default_params)

    layout = {}
    # First pass: build non-resonator components
    for comp in components:
        p = {**defaults, **comp.params}

        if comp.component_type == "TransmonCross":
            layout[comp.name] = make_transmon_cross(
                cross_length=p["cross_length"],
                cross_gap=p["cross_gap"],
                cross_width=p["cross_width"],
                pos_x=comp.pos_x,
                pos_y=comp.pos_y,
                orientation=comp.orientation,
            )

        elif comp.component_type == "Claw":
            layout[comp.name] = make_claw(
                claw_length=p["claw_length"],
                ground_spacing=p["ground_spacing"],
                cross_length=p["cross_length"],
                cross_gap=p["cross_gap"],
                cross_width=p["cross_width"],
                claw_width=p["claw_width"],
                claw_gap=p["claw_gap"],
                connector_location=p.get("connector_location", 90),
                pos_x=comp.pos_x,
                pos_y=comp.pos_y,
                orientation=comp.orientation,
            )

        elif comp.component_type == "CoupledLineTee":
            layout[comp.name] = make_feedline(
                coupling_length=p["coupling_length"],
                prime_width=p["prime_width"],
                prime_gap=p["prime_gap"],
                pos_x=comp.pos_x,
                pos_y=comp.pos_y,
                orientation=comp.orientation,
            )

    # Second pass: build resonators (need pin positions from other components)
    for comp in components:
        if comp.component_type != "RouteMeander":
            continue

        p = {**defaults, **comp.params}

        # Determine start/end positions from connected components
        start_comp = comp.connect_from
        end_comp = comp.connect_to

        if start_comp and start_comp in layout:
            start_data = layout[start_comp]
            start_pos = start_data.get("pin_position", (comp.pos_x, comp.pos_y))
            start_dir = start_data.get("pin_direction", (1.0, 0.0))
        else:
            start_pos = (comp.pos_x, comp.pos_y)
            start_dir = (1.0, 0.0)

        adj_distance = p["coupling_length"] if p["coupling_length"] > 150 else 0.0
        asymmetry = -(adj_distance / 2.0)
        jog_ext = [("R90", adj_distance / 1.5)] if adj_distance > 0 else None

        # Determine CLT position for the resonator endpoint
        clt_pos_x = comp.pos_x
        clt_pos_y = comp.pos_y
        clt_orientation = comp.orientation

        # If connected to a feedline, use its position
        if end_comp and end_comp in layout:
            end_data = layout[end_comp]
            clt_pos_x = end_data.get("position", (0, 0))[0]
            clt_pos_y = end_data.get("position", (0, 0))[1]
            clt_orientation = end_data.get("orientation", -90.0)

        layout[comp.name] = make_resonator(
            total_length=p["total_length"],
            coupling_length=p["coupling_length"],
            start_pos=start_pos,
            start_direction=start_dir,
            trace_width=p["trace_width"],
            trace_gap=p["trace_gap"],
            second_width=p["second_width"],
            second_gap=p["second_gap"],
            prime_width=p["prime_width"],
            prime_gap=p["prime_gap"],
            coupling_space=p["coupling_space"],
            down_length=p["down_length"],
            spacing=p["spacing"],
            asymmetry=asymmetry,
            fillet=p["fillet"],
            start_straight=p["start_straight"],
            end_straight=p["end_straight"],
            jog_extension=jog_ext,
            clt_pos_x=clt_pos_x,
            clt_pos_y=clt_pos_y,
            clt_orientation=clt_orientation,
        )

    # Collect all varied design params
    all_params = {}
    for comp in components:
        all_params.update(comp.params)
    layout["design_params"] = {**defaults, **all_params}

    return layout
