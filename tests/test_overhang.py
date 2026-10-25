"""Hand-calculation validation tests for overhanging beams.

Geometry: total length L, simple (vertical) supports at x = a and x = b
with 0 <= a < b <= L and not both at the beam ends. Loads on the
overhangs are fully included in the static equilibrium.
"""

import numpy as np
import pytest

from beamviz.beam import (
    AppliedMoment,
    DistributedLoad,
    PointLoad,
    SupportType,
    analyze,
    compute_reactions,
    section_forces,
)

OV = SupportType.OVERHANGING
SS = SupportType.SIMPLY_SUPPORTED


class TestCaseASymmetricOverhangs:
    """L = 6 m, supports at 1 m and 5 m, P = 10 kN down at x = 3 m.

    Hand calculation:
      R_A = R_B = 5 kN (symmetry).
      V: 0 on both overhangs; +5 between a and the load; -5 between the
         load and b; 0 after b.
      M: 0 on the overhangs; triangular between supports, Mmax = 10 kN·m
         at x = 3 m.
    """

    POS = (1.0, 5.0)
    LOADS = [PointLoad(3.0, 10.0)]

    def test_reactions(self):
        reactions = compute_reactions(6.0, OV, self.LOADS, support_positions=self.POS)
        assert reactions["R_left"] == pytest.approx(5.0)
        assert reactions["R_right"] == pytest.approx(5.0)

    def test_shear_at_stations(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v, _ = section_forces(6.0, OV, self.LOADS, x, support_positions=self.POS)
        np.testing.assert_allclose(v, [0.0, 5.0, 5.0, -5.0, -5.0, 0.0, 0.0])

    def test_moment_at_stations(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        _, m = section_forces(6.0, OV, self.LOADS, x, support_positions=self.POS)
        np.testing.assert_allclose(m, [0.0, 0.0, 5.0, 10.0, 5.0, 0.0, 0.0])

    def test_max_moment(self):
        result = analyze(6.0, OV, self.LOADS, support_positions=self.POS)
        x, val, side = result.extreme_details()["Max moment (+)"]
        assert x == pytest.approx(3.0)
        assert val == pytest.approx(10.0)
        assert side == "point value"

    def test_one_sided_limits_at_interior_supports(self):
        # Reactions are included in the right-continuous value and excluded
        # from the left-hand limit at each interior support.
        v_a_left, _ = section_forces(
            6.0, OV, self.LOADS, 1.0, strict_left=True, support_positions=self.POS
        )
        v_a, _ = section_forces(6.0, OV, self.LOADS, 1.0, support_positions=self.POS)
        assert v_a_left == pytest.approx(0.0)
        assert v_a == pytest.approx(5.0)
        v_b_left, _ = section_forces(
            6.0, OV, self.LOADS, 5.0, strict_left=True, support_positions=self.POS
        )
        v_b, _ = section_forces(6.0, OV, self.LOADS, 5.0, support_positions=self.POS)
        assert v_b_left == pytest.approx(-5.0)
        assert v_b == pytest.approx(0.0)


class TestCaseBLeftOverhang:
    """L = 5 m, supports at 1 m and 4 m, P = 4 kN down at x = 0 m.

    Hand calculation:
      R_B = [P*(0 - 1)] / (4 - 1) = -4/3 kN (downward reaction).
      R_A = 4 - (-4/3) = 16/3 kN.
      V: -4 on [0, 1); +4/3 on (1, 4); 0 on (4, 5].
      M: 0 at the free left end; -4x on [0, 1] (M(1) = -4, hogging over
         the support); (4x - 16)/3 on [1, 4] (M(4) = 0); 0 on [4, 5].
    """

    POS = (1.0, 4.0)
    LOADS = [PointLoad(0.0, 4.0)]

    def test_reactions(self):
        reactions = compute_reactions(5.0, OV, self.LOADS, support_positions=self.POS)
        assert reactions["R_left"] == pytest.approx(16.0 / 3.0)
        assert reactions["R_right"] == pytest.approx(-4.0 / 3.0)

    def test_shear_at_stations(self):
        x = np.array([0.0, 0.5, 1.0, 2.5, 4.0, 5.0])
        v, _ = section_forces(5.0, OV, self.LOADS, x, support_positions=self.POS)
        np.testing.assert_allclose(
            v, [-4.0, -4.0, 4.0 / 3.0, 4.0 / 3.0, 0.0, 0.0], atol=1e-12
        )

    def test_moment_at_stations(self):
        x = np.array([0.0, 0.5, 1.0, 2.5, 4.0, 5.0])
        _, m = section_forces(5.0, OV, self.LOADS, x, support_positions=self.POS)
        np.testing.assert_allclose(m, [0.0, -2.0, -4.0, -2.0, 0.0, 0.0], atol=1e-12)

    def test_free_end_moments_are_zero(self):
        result = analyze(5.0, OV, self.LOADS, support_positions=self.POS)
        assert result.moment[0] == pytest.approx(0.0, abs=1e-9)
        assert result.moment[-1] == pytest.approx(0.0, abs=1e-9)

    def test_negative_reaction_not_clamped(self):
        result = analyze(5.0, OV, self.LOADS, support_positions=self.POS)
        assert result.reactions["R_right"] == pytest.approx(-4.0 / 3.0)

    def test_extremes(self):
        result = analyze(5.0, OV, self.LOADS, support_positions=self.POS)
        details = result.extreme_details()
        x, val, side = details["Min shear (-)"]
        assert x == pytest.approx(0.0)
        assert val == pytest.approx(-4.0)
        x, val, side = details["Max shear (+)"]
        assert x == pytest.approx(1.0)
        assert val == pytest.approx(4.0 / 3.0)
        assert side == "right-hand limit"
        x, val, side = details["Min moment (-)"]
        assert x == pytest.approx(1.0)
        assert val == pytest.approx(-4.0)  # hogging over the left support


class TestCombinedLoadOverhang:
    """L = 6 m, supports at 1 m and 4 m, CW moment 3 kN·m at 0.5 m (left
    overhang), UDL 2 kN/m over [1, 4], point load 6 kN down at 5.5 m
    (right overhang).

    Hand calculation (moments about x = a = 1, CCW positive):
      R_B = [6*(5.5-1) + 2*3*(2.5-1) + 3] / 3 = (27 + 9 + 3)/3 = 13 kN
      R_A = (6 + 6) - 13 = -1 kN
    Cross-check (moments about x = b = 4):
      -R_A*3 - 6*(5.5-4) + 6*(4-2.5) - 3 = 3 - 9 + 9 - 3 = 0 ✓
    Shear: V = 0 up to the couple; -1 at a; -1 - 2(x-1) to -7 at b-;
           +6 after b; 0 after the 5.5 m load.
    Moment: +3 jump at 0.5; 3 on [0.5, 1]; 3 - (x-1) - (x-1)^2 on [1, 4]
           (M(4) = -9); back to 0 at 5.5; 0 to the free end.
    """

    POS = (1.0, 4.0)
    LOADS = [
        AppliedMoment(0.5, 3.0),
        DistributedLoad(1.0, 4.0, 2.0),
        PointLoad(5.5, 6.0),
    ]

    def test_reactions(self):
        reactions = compute_reactions(6.0, OV, self.LOADS, support_positions=self.POS)
        assert reactions["R_left"] == pytest.approx(-1.0)
        assert reactions["R_right"] == pytest.approx(13.0)

    def test_shear_at_stations(self):
        x = np.array([0.0, 1.0, 2.5, 4.0, 5.0, 5.5, 6.0])
        v, _ = section_forces(6.0, OV, self.LOADS, x, support_positions=self.POS)
        np.testing.assert_allclose(v, [0.0, -1.0, -4.0, 6.0, 6.0, 0.0, 0.0], atol=1e-12)

    def test_moment_at_stations(self):
        x = np.array([0.0, 0.5, 1.0, 2.5, 4.0, 5.5, 6.0])
        _, m = section_forces(6.0, OV, self.LOADS, x, support_positions=self.POS)
        expected = [
            0.0,
            3.0,               # +M_cw jump at x = 0.5
            3.0,               # constant couple region
            3.0 - 1.5 - 2.25,  # = -0.75
            -9.0,              # at the right support
            0.0,               # back to zero at the load
            0.0,               # free end
        ]
        np.testing.assert_allclose(m, expected, atol=1e-12)

    def test_extremes(self):
        result = analyze(6.0, OV, self.LOADS, support_positions=self.POS)
        details = result.extreme_details()
        x, val, side = details["Max moment (+)"]
        assert x == pytest.approx(0.5)
        assert val == pytest.approx(3.0)
        assert side == "right-hand limit"
        x, val, side = details["Min moment (-)"]
        assert x == pytest.approx(4.0)
        assert val == pytest.approx(-9.0)
        x, val, side = details["Min shear (-)"]
        assert x == pytest.approx(4.0)
        assert val == pytest.approx(-7.0)
        assert side == "left-hand limit"


class TestOverhangGeometryValidation:
    def test_missing_positions(self):
        with pytest.raises(ValueError, match="require support positions"):
            compute_reactions(6.0, OV, [])

    @pytest.mark.parametrize(
        "a,b",
        [
            (1.0, 1.0),   # a == b
            (2.0, 1.0),   # a > b
            (-0.5, 3.0),  # a < 0
            (1.0, 7.0),   # b > L
        ],
    )
    def test_invalid_geometry(self, a, b):
        with pytest.raises(ValueError, match="0 <= a < b"):
            compute_reactions(6.0, OV, [], support_positions=(a, b))

    def test_both_supports_at_ends_rejected(self):
        with pytest.raises(ValueError, match="simply supported"):
            compute_reactions(6.0, OV, [], support_positions=(0.0, 6.0))

    def test_support_at_one_end_is_allowed(self):
        # a = 0 with a right overhang is a valid configuration.
        # L = 6, a = 0, b = 4, P = 2 kN at the free end x = 6:
        # R_B = 2*6/4 = 3 kN, R_A = 2 - 3 = -1 kN.
        reactions = compute_reactions(
            6.0, OV, [PointLoad(6.0, 2.0)], support_positions=(0.0, 4.0)
        )
        assert reactions["R_left"] == pytest.approx(-1.0)
        assert reactions["R_right"] == pytest.approx(3.0)
        v, m = section_forces(
            6.0, OV, [PointLoad(6.0, 2.0)], np.array([0.0, 4.0, 6.0]),
            support_positions=(0.0, 4.0),
        )
        np.testing.assert_allclose(v, [-1.0, 2.0, 0.0], atol=1e-12)
        # M(0) = 0 (no lever arm at the free end of the load/reaction
        # contributions), M(4) = R_A*4 = -4 (hogging at the support),
        # M(6) = 0 (free end).
        np.testing.assert_allclose(m, [0.0, -4.0, 0.0], atol=1e-12)

    def test_empty_loads(self):
        result = analyze(6.0, OV, [], support_positions=(1.0, 5.0))
        assert all(r == 0.0 for r in result.reactions.values())
        np.testing.assert_allclose(result.shear, 0.0)
        np.testing.assert_allclose(result.moment, 0.0)

    def test_simply_supported_still_works_after_refactor(self):
        # Guard: the overhang refactor must not disturb the existing mode.
        v, m = section_forces(6.0, SS, [PointLoad(3.0, 10.0)], np.array([3.0]))
        assert v[0] == pytest.approx(-5.0)
        assert m[0] == pytest.approx(15.0)
