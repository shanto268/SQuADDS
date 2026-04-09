"""Compose all four component polygons into a positioned chip layout.

Replicates the placement logic from
``CavityClaw.make()`` / ``QubitCavity.make()``:

Actual SQuADDS design options (from the database):
- **Qubit** at ``(-1500, 1200)`` μm, orientation ``-90`` degrees
  → JJ port faces **west**, cross arms: N=east, S=west, E=south, W=north
- **Claw** ``connector_location=90`` → attached to the **East arm** of the qubit
  (which, after -90° rotation, points **south**)
  Wait actually connector_location=90 means the qubit's 90° arm.
  The TransmonCross arms are: 0=west, 90=north, 180=east, 270=south.
  With qubit orientation= -90°:
    0 (west) → south
    90 (north) → west... no.
  Actually, connector_location is applied BEFORE the qubit orientation rotation.
  So connector_location=90 is the north arm of the un-rotated cross.
  After qubit rotates -90°, the north arm points EAST.
  → Claw is on the EAST side of the qubit.

- **CLT** (feedline) at ``(0, 1200)``, orientation ``-90``
  → Feedline is **vertical**
- **Resonator** routes from CLT ``second_end`` pin to claw ``readout`` pin
  → Meander with fillet=49.9μm, spacing=100μm
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
    # ── Component defaults matching SQuADDS DB ──
    cross_width: float = 30.0,
    claw_width: float = 15.0,
    claw_gap: float = 5.1,
    claw_cpw_length: float = 0.0,
    claw_cpw_width: float = 11.7,
    connector_location: int = 90,  # North arm (before rotation) → East (after -90°)
    prime_width: float = 11.7,
    prime_gap: float = 5.1,
    second_width: float = 11.7,
    second_gap: float = 5.1,
    coupling_space: float = 7.9,
    down_length: float = 50.0,
    open_termination: bool = False,
    mirror: bool = False,
    spacing: float = 100.0,
    fillet: float = 49.9,
    start_straight: float = 50.0,
    end_straight: float = 50.0,
    trace_width: float = 11.7,
    trace_gap: float = 5.1,
    # ── Placement (matching SQuADDS DB) ──
    qubit_orientation: float = -90.0,
    clt_pos_x: float = 0.0,
    clt_pos_y: float = 1200.0,
    clt_orientation: float = -90.0,
    qubit_pos_x: float = -1500.0,
    qubit_pos_y: float = 1200.0,
) -> dict:
    """Build a complete 4-component layout from design parameters.

    All dimensions in **micrometres**.  The default placement matches
    the actual SQuADDS database design options:

    - Qubit at (-1500, 1200), orientation -90° (JJ faces west)
    - Claw on the East arm (connector_location=90, post-rotation)
    - Feedline vertical at (0, 1200), orientation -90°
    - Meander with fillet=49.9μm, rounded corners

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
    # In SQuADDS: start_pin=cplr.second_end, end_pin=claw.readout
    # The RouteMeander goes FROM the CLT's second_end TO the claw's readout
    # For quarter-wave: full total_length is used
    resonator_length = total_length

    # Compute asymmetry from coupling_length (matching CavityClaw.make_cpws)
    # In the SQuADDS DB we see asymmetry can be negative
    adj_distance = coupling_length if coupling_length > 150 else 0.0
    asymmetry = -(adj_distance / 2.0)  # negative = shifts meander center downward

    # Jog extension on the end lead (matching CavityClaw jog pattern)
    jog_ext = [("R90", adj_distance / 1.5)] if adj_distance > 0 else None

    resonator = make_resonator(
        total_length=resonator_length,
        coupling_length=coupling_length,
        # Meander START = CLT second_end pin, END = claw readout pin
        start_pos=claw["pin_position"],
        start_direction=claw["pin_direction"],
        trace_width=trace_width,
        trace_gap=trace_gap,
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
        jog_extension=jog_ext,
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
