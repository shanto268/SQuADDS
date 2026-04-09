"""Compose all four component polygons into a positioned chip layout.

Replicates the placement logic from
``CavityClaw.make()`` / ``QubitCavity.make()``:
- CoupledLineTee at the origin (or user-specified position)
- Qubit at ``(-1500, 0)`` or ``(-1000, 0)`` μm depending on ``total_length``
- Claw attached to the qubit's west arm (connector_location=0)
- Meander routed from the claw pin to the CLT second_end pin
"""

from __future__ import annotations

from squadds.ml.universal.geometry.claw import make_claw
from squadds.ml.universal.geometry.feedline import make_feedline
from squadds.ml.universal.geometry.qubit import make_transmon_cross
from squadds.ml.universal.geometry.resonator import make_resonator


def build_layout(
    cross_length: float,
    cross_gap: float,
    claw_length: float,
    ground_spacing: float,
    coupling_length: float,
    total_length: float,
    # ── Component defaults (overridable) ──
    cross_width: float = 20.0,
    claw_width: float = 10.0,
    claw_gap: float = 6.0,
    claw_cpw_length: float = 40.0,
    claw_cpw_width: float = 10.0,
    connector_location: int = 0,
    prime_width: float = 10.0,
    prime_gap: float = 6.0,
    second_width: float = 10.0,
    second_gap: float = 6.0,
    coupling_space: float = 3.0,
    down_length: float = 100.0,
    open_termination: bool = True,
    mirror: bool = False,
    spacing: float = 100.0,
    fillet: float = 49.9,
    start_straight: float = 100.0,
    end_straight: float = 50.0,
    # ── Placement overrides ──
    qubit_orientation: float = 0.0,
    clt_pos_x: float = 0.0,
    clt_pos_y: float = 0.0,
    clt_orientation: float = 180.0,
    qubit_pos_x: float | None = None,
    qubit_pos_y: float = 0.0,
) -> dict:
    """Build a complete 4-component layout from design parameters.

    All dimensions in **micrometres**.

    Args:
        cross_length: Qubit arm length (from parquet).
        cross_gap: Qubit CPW gap (from parquet).
        claw_length: Claw arm length (from parquet).
        ground_spacing: Ground gap between claw and qubit (from parquet).
        coupling_length: CLT coupling length (from parquet).
        total_length: Full resonator length (from parquet).
        (remaining args): Component defaults, see individual generators.

    Returns:
        Dictionary with keys:

        * ``qubit`` – output of :func:`make_transmon_cross`
        * ``claw`` – output of :func:`make_claw`
        * ``resonator`` – output of :func:`make_resonator`
        * ``feedline`` – output of :func:`make_feedline`
        * ``design_params`` – the 6 swept design parameters
    """
    # ── Qubit placement (matching CavityClaw.make_qubit) ───────────────
    if qubit_pos_x is None:
        qubit_pos_x = -1500.0 if total_length > 2500.0 else -1000.0

    # ── 1. Qubit ───────────────────────────────────────────────────────
    qubit = make_transmon_cross(
        cross_length=cross_length,
        cross_gap=cross_gap,
        cross_width=cross_width,
        pos_x=qubit_pos_x,
        pos_y=qubit_pos_y,
        orientation=qubit_orientation,
    )

    # ── 2. Claw (attached to qubit) ────────────────────────────────────
    claw = make_claw(
        claw_length=claw_length,
        ground_spacing=ground_spacing,
        cross_length=cross_length,
        cross_gap=cross_gap,
        cross_width=cross_width,
        claw_width=claw_width,
        claw_gap=claw_gap,
        claw_cpw_length=claw_cpw_length,
        claw_cpw_width=claw_cpw_width,
        connector_location=connector_location,
        pos_x=qubit_pos_x,
        pos_y=qubit_pos_y,
        orientation=qubit_orientation,
    )

    # ── 3. Feedline (CLT prime_cpw) ────────────────────────────────────
    feedline = make_feedline(
        coupling_length=coupling_length,
        prime_width=prime_width,
        prime_gap=prime_gap,
        pos_x=clt_pos_x,
        pos_y=clt_pos_y,
        orientation=clt_orientation,
    )

    # ── 4. Resonator (meander + CLT second_cpw) ───────────────────────
    # Meander routes from: claw pin → CLT second_end
    # For CLT coupler type, resonator length = total_length / 2
    resonator_length = total_length / 2.0

    # Compute asymmetry and jog from coupling_length (matching CavityClaw.make_cpws)
    adj_distance = coupling_length if coupling_length > 150 else 0.0
    asymmetry = adj_distance / 3.0
    jog_length = adj_distance / 1.5

    resonator = make_resonator(
        total_length=resonator_length,
        coupling_length=coupling_length,
        start_pos=claw["pin_position"],
        start_direction=claw["pin_direction"],
        trace_width=second_width,
        trace_gap=second_gap,
        second_width=second_width,
        second_gap=second_gap,
        prime_width=prime_width,
        prime_gap=prime_gap,
        coupling_space=coupling_space,
        down_length=down_length,
        open_termination=open_termination,
        mirror=mirror,
        spacing=spacing,
        asymmetry=asymmetry,
        fillet=fillet,
        start_straight=start_straight,
        end_straight=end_straight,
        jog_length=jog_length,
        clt_pos_x=clt_pos_x,
        clt_pos_y=clt_pos_y,
        clt_orientation=clt_orientation,
    )

    return {
        "qubit": qubit,
        "claw": claw,
        "resonator": resonator,
        "feedline": feedline,
        "design_params": {
            "cross_length": cross_length,
            "cross_gap": cross_gap,
            "claw_length": claw_length,
            "ground_spacing": ground_spacing,
            "coupling_length": coupling_length,
            "total_length": total_length,
        },
    }
