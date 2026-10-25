"""Unit tests for the BeamViz beam analysis engine."""

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

SS = SupportType.SIMPLY_SUPPORTED
FIX_L = SupportType.CANTILEVER_LEFT
FIX_R = SupportType.CANTILEVER_RIGHT


class TestSimplySupported:
    def test_center_point_load_reactions(self):
        reactions = compute_reactions(6.0, SS, [PointLoad(3.0, 10.0)])
        assert reactions["R_left"] == pytest.approx(5.0)
        assert reactions["R_right"] == pytest.approx(5.0)

    def test_center_point_load_shear_and_moment(self):
        x = np.array([0.0, 1.5, 3.0, 4.5, 6.0])
        v, m = section_forces(6.0, SS, [PointLoad(3.0, 10.0)], x)
        # Shear: +P/2 before the load, -P/2 after, zero at both supports.
        np.testing.assert_allclose(v, [5.0, 5.0, -5.0, -5.0, 0.0])
        # Moment: zero at supports, PL/4 = 15 kN·m at midspan.
        np.testing.assert_allclose(m, [0.0, 7.5, 15.0, 7.5, 0.0])

    def test_udl_full_span(self):
        loads = [DistributedLoad(0.0, 4.0, 2.0)]
        x = np.array([0.0, 2.0, 4.0])
        v, m = section_forces(4.0, SS, loads, x)
        assert v[0] == pytest.approx(4.0)  # wL/2 at the left support
        assert v[1] == pytest.approx(0.0, abs=1e-9)  # zero shear at midspan
        assert v[2] == pytest.approx(0.0)
        assert m[1] == pytest.approx(2.0 * 4.0**2 / 8.0)  # wL^2/8 = 4
        assert m[0] == pytest.approx(0.0)
        assert m[2] == pytest.approx(0.0)

    def test_partial_udl(self):
        loads = [DistributedLoad(1.0, 3.0, 3.0)]
        reactions = compute_reactions(6.0, SS, loads)
        # Total 6 kN acting at centroid x = 2 m.
        assert reactions["R_right"] == pytest.approx(2.0)
        assert reactions["R_left"] == pytest.approx(4.0)
        v, m = section_forces(6.0, SS, loads, np.array([2.0]))
        assert v[0] == pytest.approx(4.0 - 3.0 * (2.0 - 1.0))
        assert m[0] == pytest.approx(4.0 * 2.0 - 0.5 * 3.0 * 1.0**2)

    def test_clockwise_moment_reactions_and_jump(self):
        loads = [AppliedMoment(1.0, 8.0)]
        reactions = compute_reactions(4.0, SS, loads)
        assert reactions["R_left"] == pytest.approx(-2.0)
        assert reactions["R_right"] == pytest.approx(2.0)
        v, m = section_forces(4.0, SS, loads, np.array([0.5, 1.0, 4.0]))
        assert v[0] == pytest.approx(-2.0)
        assert m[0] == pytest.approx(-1.0)  # just before the moment
        assert m[1] == pytest.approx(-2.0 + 8.0)  # +M jump at x = 1
        assert m[2] == pytest.approx(0.0)  # diagram closes at the right end


class TestCantileverLeft:
    def test_tip_load(self):
        loads = [PointLoad(3.0, 5.0)]
        reactions = compute_reactions(3.0, FIX_L, loads)
        assert reactions["R_left"] == pytest.approx(5.0)
        assert reactions["M_left"] == pytest.approx(15.0)  # CCW positive
        x = np.array([0.0, 1.5, 2.999999, 3.0])
        v, m = section_forces(3.0, FIX_L, loads, x)
        np.testing.assert_allclose(v, [5.0, 5.0, 5.0, 0.0])
        np.testing.assert_allclose(m, [-15.0, -7.5, -0.000005, 0.0], atol=1e-4)

    def test_fixed_end_moment_hogging_at_support(self):
        result = analyze(3.0, FIX_L, [PointLoad(3.0, 5.0)])
        assert result.moment[0] == pytest.approx(-15.0)  # hogging at the wall
        assert result.moment.max() == pytest.approx(0.0, abs=1e-9)


class TestCantileverRight:
    def test_load_at_free_end(self):
        loads = [PointLoad(0.0, 4.0)]
        reactions = compute_reactions(3.0, FIX_R, loads)
        assert reactions["R_right"] == pytest.approx(4.0)
        assert reactions["M_right"] == pytest.approx(-12.0)  # CCW negative = CW
        x = np.array([0.0, 1.5, 3.0])
        v, m = section_forces(3.0, FIX_R, loads, x)
        np.testing.assert_allclose(v, [-4.0, -4.0, 0.0])
        np.testing.assert_allclose(m, [0.0, -6.0, -12.0])

    def test_full_udl(self):
        loads = [DistributedLoad(0.0, 3.0, 2.0)]
        x = np.array([0.0, 1.5, 3.0])
        v, m = section_forces(3.0, FIX_R, loads, x)
        np.testing.assert_allclose(v, [0.0, -3.0, 0.0])
        np.testing.assert_allclose(m, [0.0, -2.25, -9.0])  # -w*x^2/2 at the wall


class TestAnalysisResult:
    def test_extremes_center_point_load(self):
        result = analyze(6.0, SS, [PointLoad(3.0, 10.0)])
        loc, val = result.extremes()["Max moment (+)"]
        assert loc == pytest.approx(3.0)
        assert val == pytest.approx(15.0)
        assert result.shear.max() == pytest.approx(5.0)
        assert result.shear.min() == pytest.approx(-5.0)

    def test_diagrams_close_at_ends_for_simply_supported(self):
        loads = [PointLoad(2.0, 7.0), DistributedLoad(4.0, 6.0, 1.5), AppliedMoment(1.0, 3.0)]
        result = analyze(6.0, SS, loads)
        assert result.moment[0] == pytest.approx(0.0, abs=1e-9)
        assert result.moment[-1] == pytest.approx(0.0, abs=1e-9)
        assert result.shear[-1] == pytest.approx(0.0, abs=1e-9)

    def test_empty_loads_give_zero_forces(self):
        result = analyze(5.0, SS, [])
        assert all(r == 0.0 for r in result.reactions.values())
        np.testing.assert_allclose(result.shear, 0.0)
        np.testing.assert_allclose(result.moment, 0.0)


class TestValidation:
    def test_negative_length(self):
        with pytest.raises(ValueError, match="positive"):
            analyze(-1.0, SS, [])

    def test_load_outside_beam(self):
        with pytest.raises(ValueError, match="within"):
            analyze(6.0, SS, [PointLoad(7.0, 1.0)])

    def test_udl_bad_span(self):
        with pytest.raises(ValueError, match="UDL must satisfy"):
            analyze(6.0, SS, [DistributedLoad(2.0, 2.0, 5.0)])

    def test_unknown_load_type(self):
        with pytest.raises(TypeError, match="unknown load type"):
            analyze(6.0, SS, ["not a load"])


class TestBoundaryExtremes:
    """Regression tests: exact one-sided extrema at beam boundaries.

    Root cause fixed here: analyze() previously refined the grid only for
    discontinuities strictly inside (0, L) and took argmax/argmin over the
    sampled arrays, so a shear extremum exactly at a support (e.g. V(L-))
    was reported at the last interior grid point instead.
    """

    DEFAULT_LOADS = [PointLoad(3.0, 10.0), DistributedLoad(4.0, 6.0, 2.0)]

    def test_default_example_extremes_exact(self):
        result = analyze(6.0, SS, self.DEFAULT_LOADS)
        details = result.extreme_details()

        x, val, side = details["Max shear (+)"]
        assert x == pytest.approx(0.0)
        assert val == pytest.approx(17.0 / 3.0)

        x, val, side = details["Min shear (-)"]
        assert x == pytest.approx(6.0)
        assert val == pytest.approx(-25.0 / 3.0)
        assert side == "left-hand limit"  # V(L-), before the reaction

        x, val, side = details["Max moment (+)"]
        assert x == pytest.approx(3.0)
        assert val == pytest.approx(17.0)

        x, val, side = details["Min moment (-)"]
        assert val == pytest.approx(0.0)

    def test_one_sided_limits_differ_at_right_support(self):
        # V(L-) (before the reaction) and the point value V(L) must differ.
        v_left, _ = section_forces(6.0, SS, self.DEFAULT_LOADS, 6.0, strict_left=True)
        v_at, _ = section_forces(6.0, SS, self.DEFAULT_LOADS, 6.0)
        assert v_left == pytest.approx(-25.0 / 3.0)
        assert v_at == pytest.approx(0.0, abs=1e-9)

    def test_extremes_tuple_api_consistent_with_details(self):
        result = analyze(6.0, SS, self.DEFAULT_LOADS)
        loc, val = result.extremes()["Min shear (-)"]
        assert loc == pytest.approx(6.0)
        assert val == pytest.approx(-25.0 / 3.0)

    def test_grid_refines_boundary_discontinuities(self):
        # The sampled arrays must now contain an exact epsilon sample just
        # left of the support so the plotted jump is vertical.
        result = analyze(4.0, SS, [DistributedLoad(0.0, 4.0, 2.0)])
        eps = 4.0 * 1e-6
        assert result.x[-1] == pytest.approx(4.0)
        assert result.x[-2] == pytest.approx(4.0 - eps)
        assert result.shear[-2] == pytest.approx(-4.0, abs=1e-4)  # V(4-) = -wL/2
        assert result.shear[-1] == pytest.approx(0.0, abs=1e-9)

    def test_udl_moment_extreme_at_interior_stationary_point(self):
        result = analyze(4.0, SS, [DistributedLoad(0.0, 4.0, 2.0)])
        x, val, side = result.extreme_details()["Max moment (+)"]
        assert x == pytest.approx(2.0)
        assert val == pytest.approx(4.0)  # wL^2/8
        assert side == "point value"

    def test_partial_udl_stationary_point(self):
        # SS L = 6 m, UDL 2 kN/m over [0, 3]:
        # R_A = 4.5, R_B = 1.5, Mmax = 5.0625 kN·m at x = 2.25 m (V = 0).
        result = analyze(6.0, SS, [DistributedLoad(0.0, 3.0, 2.0)])
        x, val, side = result.extreme_details()["Max moment (+)"]
        assert x == pytest.approx(2.25)
        assert val == pytest.approx(5.0625)
        assert side == "point value"

    def test_point_load_exactly_at_supports(self):
        for pos in (0.0, 4.0):
            result = analyze(4.0, SS, [PointLoad(pos, 6.0)])
            np.testing.assert_allclose(result.shear, 0.0, atol=1e-9)
            np.testing.assert_allclose(result.moment, 0.0, atol=1e-9)

    def test_cantilever_wall_extremes(self):
        result = analyze(3.0, FIX_L, [PointLoad(3.0, 5.0)])
        details = result.extreme_details()
        x, val, side = details["Min moment (-)"]
        assert x == pytest.approx(0.0)
        assert val == pytest.approx(-15.0)
        x, val, side = details["Max shear (+)"]
        assert val == pytest.approx(5.0)
