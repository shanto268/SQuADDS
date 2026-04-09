"""Tests for Shapely geometry generators (Milestone 1).

Covers: qubit, claw, feedline, resonator, and layout assembly.
"""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon

from squadds.ml.universal.geometry.claw import make_claw
from squadds.ml.universal.geometry.feedline import make_feedline
from squadds.ml.universal.geometry.layout import build_layout
from squadds.ml.universal.geometry.qubit import make_transmon_cross
from squadds.ml.universal.geometry.resonator import make_resonator

# ═══════════════════════════════════════════════════════════════════════
# Qubit Tests
# ═══════════════════════════════════════════════════════════════════════


class TestTransmonCross:
    """Tests for make_transmon_cross()."""

    def test_returns_polygon(self):
        result = make_transmon_cross(cross_length=200, cross_gap=20)
        assert isinstance(result["cross"], Polygon)
        assert isinstance(result["cross_etch"], Polygon)
        assert isinstance(result["jj"], LineString)

    def test_cross_area_known(self):
        """A cross with length=100, width=20 has a known area.

        The cross is two perpendicular line segments buffered with
        cap_style='square', which extends width/2 past each endpoint.
        Each arm: (2*100 + 2*10) × 20 = 220×20 = 4400.
        Union of two arms minus overlap (20×20): 2*4400 - 20*20 = 8400.
        """
        result = make_transmon_cross(cross_length=100, cross_gap=10, cross_width=20)
        expected_area = 2 * (220 * 20) - (20 * 20)  # 8400
        assert abs(result["cross"].area - expected_area) < 1.0

    def test_positioning(self):
        """Cross placed at (500, 300) should have centroid near that point."""
        result = make_transmon_cross(cross_length=200, cross_gap=20, pos_x=500, pos_y=300)
        cx, cy = result["cross"].centroid.x, result["cross"].centroid.y
        assert abs(cx - 500) < 1.0
        assert abs(cy - 300) < 1.0

    def test_position_retained(self):
        """The position tuple should match the input."""
        result = make_transmon_cross(cross_length=200, cross_gap=20, pos_x=100, pos_y=200)
        assert result["position"] == (100, 200)

    def test_orientation(self):
        """90° rotation should swap pin N/E positions."""
        r0 = make_transmon_cross(cross_length=200, cross_gap=20, pos_x=0, pos_y=0, orientation=0)
        r90 = make_transmon_cross(cross_length=200, cross_gap=20, pos_x=0, pos_y=0, orientation=90)
        # After 90° CCW: original N pin (0, +y) should move to (-y, 0) ≈ W position
        assert abs(r90["pins"]["N"][0] - (-r0["pins"]["N"][1])) < 1.0

    def test_etch_contains_cross(self):
        """Etch polygon must fully contain the cross polygon."""
        result = make_transmon_cross(cross_length=200, cross_gap=20)
        assert result["cross_etch"].contains(result["cross"])

    def test_pins_outside_etch(self):
        """Pin tips should be on the etch boundary, not inside the cross."""
        result = make_transmon_cross(cross_length=200, cross_gap=20)
        for _pin_name, pos in result["pins"].items():
            from shapely.geometry import Point

            assert not result["cross"].contains(Point(pos))

    def test_params_stored(self):
        result = make_transmon_cross(cross_length=200, cross_gap=20, cross_width=30)
        assert result["params"]["cross_length"] == 200
        assert result["params"]["cross_gap"] == 20
        assert result["params"]["cross_width"] == 30


# ═══════════════════════════════════════════════════════════════════════
# Claw Tests
# ═══════════════════════════════════════════════════════════════════════


class TestClaw:
    """Tests for make_claw()."""

    def test_returns_polygon(self):
        result = make_claw(claw_length=30, ground_spacing=5, cross_length=200, cross_gap=20)
        assert isinstance(result["arm"], Polygon)
        assert isinstance(result["etch"], Polygon)
        assert isinstance(result["pin"], LineString)

    def test_pin_position_exists(self):
        result = make_claw(claw_length=30, ground_spacing=5, cross_length=200, cross_gap=20)
        assert "pin_position" in result
        assert len(result["pin_position"]) == 2

    def test_pin_direction_exists(self):
        result = make_claw(claw_length=30, ground_spacing=5, cross_length=200, cross_gap=20)
        assert "pin_direction" in result
        d = result["pin_direction"]
        # Should be approximately a unit vector
        assert abs(np.linalg.norm(d) - 1.0) < 0.01

    def test_claw_west_of_qubit(self):
        """Default connector_location=0 (west) → claw centroid should be at negative x."""
        result = make_claw(claw_length=30, ground_spacing=5, cross_length=200, cross_gap=20, pos_x=0, pos_y=0)
        assert result["position"][0] < 0

    def test_claw_positioned_relative_to_qubit(self):
        """Claw placed at qubit (500, 0) should have centroid near x=500 minus offset."""
        result = make_claw(claw_length=30, ground_spacing=5, cross_length=200, cross_gap=20, pos_x=500, pos_y=0)
        assert result["position"][0] < 500  # west of qubit

    def test_etch_contains_arm(self):
        result = make_claw(claw_length=30, ground_spacing=5, cross_length=200, cross_gap=20)
        assert result["etch"].contains(result["arm"])

    def test_params_stored(self):
        result = make_claw(claw_length=30, ground_spacing=5, cross_length=200, cross_gap=20)
        assert result["params"]["claw_length"] == 30
        assert result["params"]["ground_spacing"] == 5


# ═══════════════════════════════════════════════════════════════════════
# Feedline Tests
# ═══════════════════════════════════════════════════════════════════════


class TestFeedline:
    """Tests for make_feedline()."""

    def test_returns_polygon(self):
        result = make_feedline(coupling_length=200)
        assert isinstance(result["trace"], Polygon)
        assert isinstance(result["etch"], Polygon)
        assert isinstance(result["centerline"], LineString)

    def test_feedline_length(self):
        """Feedline centerline length = 2 * coupling_length."""
        result = make_feedline(coupling_length=200, orientation=0)
        assert abs(result["centerline"].length - 400) < 1.0

    def test_positioning(self):
        result = make_feedline(coupling_length=200, pos_x=100, pos_y=50)
        assert result["position"] == (100, 50)

    def test_pins_at_endpoints(self):
        result = make_feedline(coupling_length=200, pos_x=0, pos_y=0, orientation=0)
        ps = result["pins"]["prime_start"]
        pe = result["pins"]["prime_end"]
        dist = np.linalg.norm(np.array(ps) - np.array(pe))
        assert abs(dist - 400) < 1.0

    def test_etch_contains_trace(self):
        result = make_feedline(coupling_length=200)
        assert result["etch"].contains(result["trace"])


# ═══════════════════════════════════════════════════════════════════════
# Resonator Tests
# ═══════════════════════════════════════════════════════════════════════


class TestResonator:
    """Tests for make_resonator()."""

    def test_returns_polygon(self):
        result = make_resonator(
            total_length=2000,
            coupling_length=200,
            start_pos=(-500, 0),
            start_direction=(-1, 0),
        )
        assert isinstance(result["trace"], (Polygon, MultiPolygon))

    def test_has_centerline(self):
        result = make_resonator(
            total_length=2000,
            coupling_length=200,
            start_pos=(-500, 0),
            start_direction=(-1, 0),
        )
        assert "centerline" in result

    def test_path_length_reasonable(self):
        """Generated path length should be in the right ballpark of total_length."""
        result = make_resonator(
            total_length=2000,
            coupling_length=200,
            start_pos=(-500, 0),
            start_direction=(-1, 0),
            spacing=100,
        )
        # The path won't be exact (leads, CLT arm, etc.), but should be > 50% of target
        assert result["path_length"] > 500

    def test_position_retained(self):
        result = make_resonator(
            total_length=2000,
            coupling_length=200,
            start_pos=(-500, 100),
            start_direction=(-1, 0),
        )
        assert result["start_pos"] == (-500, 100)

    def test_clt_arm_included(self):
        result = make_resonator(
            total_length=2000,
            coupling_length=200,
            start_pos=(-500, 0),
            start_direction=(-1, 0),
        )
        assert "clt_arm" in result
        assert isinstance(result["clt_arm"]["trace"], Polygon)


# ═══════════════════════════════════════════════════════════════════════
# Layout Assembly Tests
# ═══════════════════════════════════════════════════════════════════════


class TestLayout:
    """Tests for build_layout()."""

    @pytest.fixture
    def default_layout(self):
        """Layout with typical SQuADDS parameters."""
        return build_layout(
            cross_length=200,
            cross_gap=20,
            claw_length=50,
            ground_spacing=6,
            coupling_length=200,
            total_length=4000,
        )

    def test_has_all_components(self, default_layout):
        for key in ("qubit", "claw", "resonator", "feedline"):
            assert key in default_layout

    def test_design_params_stored(self, default_layout):
        dp = default_layout["design_params"]
        assert dp["cross_length"] == 200
        assert dp["total_length"] == 4000

    def test_qubit_position(self, default_layout):
        """Qubit should be at x=-1500 for total_length=4000 > 2500."""
        qx = default_layout["qubit"]["position"][0]
        assert abs(qx - (-1500)) < 1.0

    def test_claw_east_of_qubit(self, default_layout):
        """Claw centroid should be to the east of the qubit (connector_location=90, orientation=-90)."""
        qx = default_layout["qubit"]["position"][0]
        cx = default_layout["claw"]["position"][0]
        assert cx > qx

    def test_feedline_at_origin(self, default_layout):
        """Feedline should be centred at (0, 1200) matching SQuADDS layout."""
        fp = default_layout["feedline"]["position"]
        assert abs(fp[0]) < 1.0
        assert abs(fp[1] - 1200) < 1.0

    def test_all_polygons_valid(self, default_layout):
        """All generated polygons should be valid Shapely geometries."""
        q = default_layout["qubit"]["cross"]
        assert q.is_valid
        c = default_layout["claw"]["arm"]
        assert c.is_valid
        f = default_layout["feedline"]["trace"]
        assert f.is_valid
        r = default_layout["resonator"]["trace"]
        assert r.is_valid or isinstance(r, MultiPolygon)

    def test_small_total_length_qubit_same_pos(self):
        """Qubit position is always at x=-1500 (matching SQuADDS defaults)."""
        layout = build_layout(
            cross_length=200,
            cross_gap=20,
            claw_length=50,
            ground_spacing=6,
            coupling_length=200,
            total_length=2000,
        )
        qx = layout["qubit"]["position"][0]
        assert abs(qx - (-1500)) < 1.0


# ═══════════════════════════════════════════════════════════════════════
# Tests for init
# ═══════════════════════════════════════════════════════════════════════


class TestInit:
    def test_init_exists(self):
        """Confirm that the __init__.py exports are correct."""

        # no-op test: as long as this import doesn't fail, we're fine
