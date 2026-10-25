"""Hand-calculation validation tests for linearly varying distributed loads.

Load model: w(x) = w1 + (w2 - w1) * (x - s) / (e - s), downward positive.
Exact resultant W = (w1 + w2)(e - s)/2 with centroid (from s)
xbar = (e - s)(w1 + 2*w2) / (3*(w1 + w2)) when w1 + w2 != 0.

Note on the section first moment: with u = x - s and k = (w2 - w1)/(e - s),
the accumulated load is Q(u) = w1*u + k*u^2/2 and the first moment of that
accumulated load ABOUT s is J(u) = w1*u^2/2 + k*u^3/3. The first moment
about the SECTION at x is u*Q(u) - J(u) = w1*u^2/2 + k*u^3/6 — only this
form is continuous with the resultant expression W*(x - c) at x = e.
"""

import math

import numpy as np
import pytest

from beamviz.beam import (
    AppliedMoment,
    DistributedLoad,
    PointLoad,
    SupportType,
    VaryingLoad,
    analyze,
    compute_reactions,
    section_forces,
)
from beamviz.plots import plot_diagrams

SS = SupportType.SIMPLY_SUPPORTED
FIX_L = SupportType.CANTILEVER_LEFT
OV = SupportType.OVERHANGING


class TestCaseAFullSpanTriangular:
    """SS beam, L = 6 m, triangular load 0 -> 6 kN/m over the full span.

    Hand calculation:
      W = (0 + 6)*6/2 = 18 kN, centroid xbar = 6*(0 + 12)/(3*6) = 4 m.
      R_B = 18*4/6 = 12 kN, R_A = 18 - 12 = 6 kN.
      V(x) = 6 - x^2/2 (quadratic), M(x) = 6x - x^3/6 (cubic).
      V = 0 at x = sqrt(12) = 2*sqrt(3) => Mmax = 6*(2r) - (2r)^3/6
          with r = sqrt(3): = 12r - 4r = 8*sqrt(3) ~= 13.8564 kN·m.
      (The task brief's "4*sqrt(2) at x = 2*sqrt(2)" is inconsistent with
      its own M(x) = 6x - x^3/6: the stationary point of that cubic is
      x = 2*sqrt(3) with Mmax = 8*sqrt(3); M(2*sqrt(2)) = 28*sqrt(2)/3
      is not a stationary value. The engine's analytic result is used.)
    """

    LOADS = [VaryingLoad(0.0, 6.0, 0.0, 6.0)]

    def test_reactions(self):
        reactions = compute_reactions(6.0, SS, self.LOADS)
        assert reactions["R_left"] == pytest.approx(6.0)
        assert reactions["R_right"] == pytest.approx(12.0)

    def test_resultant_and_centroid(self):
        load = self.LOADS[0]
        w, m0 = load.resultant_and_first_moment()
        assert w == pytest.approx(18.0)
        assert m0 == pytest.approx(72.0)  # W * c = 18 * 4
        assert load.centroid_position() == pytest.approx(4.0)

    def test_shear_at_stations(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        v, _ = section_forces(6.0, SS, self.LOADS, x)
        np.testing.assert_allclose(v, [6.0, 5.5, 4.0, 1.5, -2.0, -6.5, 0.0])
        # V(6-) one-sided limit (before the support reaction):
        v_left, _ = section_forces(6.0, SS, self.LOADS, 6.0, strict_left=True)
        assert v_left == pytest.approx(-12.0)

    def test_moment_follows_6x_minus_x_cubed_over_6(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        _, m = section_forces(6.0, SS, self.LOADS, x)
        np.testing.assert_allclose(
            m, [0.0, 35.0 / 6.0, 32.0 / 3.0, 13.5, 40.0 / 3.0, 55.0 / 6.0, 0.0]
        )

    def test_reported_maximum_moment(self):
        result = analyze(6.0, SS, self.LOADS)
        x, val, side = result.extreme_details()["Max moment (+)"]
        assert x == pytest.approx(2.0 * math.sqrt(3.0))
        assert val == pytest.approx(8.0 * math.sqrt(3.0))
        assert side == "point value"

    def test_zero_shear_at_reported_maximum(self):
        result = analyze(6.0, SS, self.LOADS)
        x, _val, _side = result.extreme_details()["Max moment (+)"]
        v_at_x, _ = section_forces(6.0, SS, self.LOADS, x)
        assert v_at_x == pytest.approx(0.0, abs=1e-9)


class TestCaseBConstantVaryingEqualsUdl:
    """SS beam, L = 4 m, w1 = w2 = 3 kN/m must equal a 3 kN/m UDL.

    Hand calculation: W = (3 + 3)*4/2 = 12 kN at midspan => RA = RB = 6 kN.
    """

    VARYING = [VaryingLoad(0.0, 4.0, 3.0, 3.0)]
    UDL = [DistributedLoad(0.0, 4.0, 3.0)]

    def test_reactions(self):
        reactions = compute_reactions(4.0, SS, self.VARYING)
        assert reactions["R_left"] == pytest.approx(6.0)
        assert reactions["R_right"] == pytest.approx(6.0)

    def test_internal_forces_match_udl_exactly(self):
        x = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0])
        v_var, m_var = section_forces(4.0, SS, self.VARYING, x)
        v_udl, m_udl = section_forces(4.0, SS, self.UDL, x)
        np.testing.assert_allclose(v_var, v_udl, atol=1e-9)
        np.testing.assert_allclose(m_var, m_udl, atol=1e-9)
        # Spot values from the UDL hand solution:
        np.testing.assert_allclose(m_var[3], 6.0, atol=1e-9)  # wL^2/8
        np.testing.assert_allclose(v_var[3], 0.0, atol=1e-9)

    def test_zero_gradient_uses_linear_extrema_path(self):
        result = analyze(4.0, SS, self.VARYING)
        x, val, _side = result.extreme_details()["Max moment (+)"]
        assert x == pytest.approx(2.0)
        assert val == pytest.approx(6.0)


class TestCaseCOverhangTriangular:
    """Overhang: L = 6 m, supports at 1 m and 5 m, triangular load on the
    left overhang from x = 0 (0 kN/m) to x = 1 m (6 kN/m).

    Hand calculation:
      W = (0 + 6)*1/2 = 3 kN; xbar = 1*(0 + 12)/(3*6) = 2/3 m => c = 2/3 m.
      Moments about x = a = 1 (CCW +):
        R_B*(5 - 1) + W*(c - a) = 0  =>  R_B = -3*(1/3)/4 = -0.25 kN
        R_A = 3 - (-0.25) = 3.25 kN.
      Shear: V = -3x^2 on [0, 1] (V(1-) = -3); +0.25 on (1, 5);
             0 after the right support reaction.
      Moment: M = -x^3 on [0, 1] (M(1) = -1, hogging over the support);
              M = -1 + 0.25*(x - 1) on [1, 5] => M(5) = 0; 0 to the free end.
    """

    POS = (1.0, 5.0)
    LOADS = [VaryingLoad(0.0, 1.0, 0.0, 6.0)]

    def test_reactions(self):
        reactions = compute_reactions(6.0, OV, self.LOADS, support_positions=self.POS)
        assert reactions["R_left"] == pytest.approx(3.25)
        assert reactions["R_right"] == pytest.approx(-0.25)

    def test_shear_at_stations(self):
        x = np.array([0.0, 0.5, 1.0, 3.0, 5.0, 6.0])
        v, _ = section_forces(6.0, OV, self.LOADS, x, support_positions=self.POS)
        np.testing.assert_allclose(v, [0.0, -0.75, 0.25, 0.25, 0.0, 0.0], atol=1e-12)
        v_left, _ = section_forces(
            6.0, OV, self.LOADS, 1.0, strict_left=True, support_positions=self.POS
        )
        assert v_left == pytest.approx(-3.0)  # V(1-) just before the reaction

    def test_moment_at_stations(self):
        x = np.array([0.0, 0.5, 1.0, 3.0, 5.0, 6.0])
        _, m = section_forces(6.0, OV, self.LOADS, x, support_positions=self.POS)
        np.testing.assert_allclose(
            m, [0.0, -0.125, -1.0, -0.5, 0.0, 0.0], atol=1e-12
        )

    def test_extremes(self):
        result = analyze(6.0, OV, self.LOADS, support_positions=self.POS)
        details = result.extreme_details()
        x, val, side = details["Min shear (-)"]
        assert x == pytest.approx(1.0)
        assert val == pytest.approx(-3.0)
        assert side == "left-hand limit"
        x, val, side = details["Max shear (+)"]
        assert x == pytest.approx(1.0)
        assert val == pytest.approx(0.25)
        assert side == "right-hand limit"
        x, val, side = details["Min moment (-)"]
        assert x == pytest.approx(1.0)
        assert val == pytest.approx(-1.0)


class TestPartialTrapezoidal:
    """SS beam, L = 6 m, trapezoidal load 2 -> 4 kN/m over [2, 5].

    Hand calculation:
      W = (2 + 4)*3/2 = 9 kN; xbar = 3*(2 + 8)/(3*6) = 5/3 => c = 11/3 m.
      R_B = 9*(11/3)/6 = 5.5 kN; R_A = 9 - 5.5 = 3.5 kN.
      (Cross-check: R_A*6 = 21 = W*(6 - 11/3) = 9*7/3 = 21.)
      With u = x - 2, k = 2/3: Q(u) = 2u + u^2/3, J_sec(u) = u*Q - (u^2 - (2/9)u^3).
      V(2.5) = 3.5 - Q(0.5) = 3.5 - 13/12 = 29/12.
      There is no reaction at x = 5 (supports at 0 and 6), so the load
      contribution is continuous at e: V(5-) = V(5) = -5.5, M(5-) = M(5) = 5.5.
      M(2) = 7; M(2.5) = 8.75 - 19/72; M(5) = 17.5 - 12 = 5.5; M(6) = 0.
      V = 0 inside (2, 5): (1/3)u^2 + 2u = 3.5 => 2u^2 + 12u - 21 = 0
      => u* = (sqrt(78) - 6)/2, x* = 2 + u* (V changes sign there).
      M(x*) = 3.5*x* - (u*^2 + u*^3/9).
    """

    LOADS = [VaryingLoad(2.0, 5.0, 2.0, 4.0)]

    def test_reactions(self):
        reactions = compute_reactions(6.0, SS, self.LOADS)
        assert reactions["R_left"] == pytest.approx(3.5)
        assert reactions["R_right"] == pytest.approx(5.5)

    def test_shear_and_moment_at_stations(self):
        x = np.array([0.0, 2.0, 2.5, 5.0, 6.0])
        v, m = section_forces(6.0, SS, self.LOADS, x)
        assert v[0] == pytest.approx(3.5)
        assert v[2] == pytest.approx(29.0 / 12.0)
        assert v[3] == pytest.approx(-5.5)  # continuous at e (no reaction here)
        np.testing.assert_allclose(
            m, [0.0, 7.0, 8.75 - 19.0 / 72.0, 5.5, 0.0], atol=1e-12
        )

    def test_quadratic_root_maximum_moment(self):
        u_star = (math.sqrt(78.0) - 6.0) / 2.0
        x_star = 2.0 + u_star
        m_expected = 3.5 * x_star - (u_star ** 2 + u_star ** 3 / 9.0)
        result = analyze(6.0, SS, self.LOADS)
        x, val, side = result.extreme_details()["Max moment (+)"]
        assert x == pytest.approx(x_star, rel=1e-9)
        assert val == pytest.approx(m_expected, rel=1e-9)
        assert side == "point value"
        # The engine's root must actually satisfy V = 0:
        v_at, _ = section_forces(6.0, SS, self.LOADS, x)
        assert v_at == pytest.approx(0.0, abs=1e-9)


class TestCantileverVarying:
    """Left-fixed cantilever, L = 4 m, triangular load 0 -> 8 kN/m.

    Hand calculation: w(x) = 2x; W = 16 kN at c = 8/3 m.
    R_A = 16 kN; M_A = integral(2x * x dx, 0, 4) = 128/3 kN·m (CCW +).
    V(x) = 16 - x^2 (quadratic); M(x) = -128/3 + 16x - x^3/3.
    M(0) = -128/3 (hogging at the wall), M(2) = -40/3, M(4) = 0.
    """

    LOADS = [VaryingLoad(0.0, 4.0, 0.0, 8.0)]

    def test_reactions(self):
        reactions = compute_reactions(4.0, FIX_L, self.LOADS)
        assert reactions["R_left"] == pytest.approx(16.0)
        assert reactions["M_left"] == pytest.approx(128.0 / 3.0)

    def test_shear_and_moment(self):
        x = np.array([0.0, 2.0, 4.0])
        v, m = section_forces(4.0, FIX_L, self.LOADS, x)
        np.testing.assert_allclose(v, [16.0, 12.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            m, [-128.0 / 3.0, -40.0 / 3.0, 0.0], atol=1e-12
        )

    def test_extremes(self):
        result = analyze(4.0, FIX_L, self.LOADS)
        details = result.extreme_details()
        x, val, _side = details["Min moment (-)"]
        assert x == pytest.approx(0.0)
        assert val == pytest.approx(-128.0 / 3.0)
        x, val, _side = details["Max shear (+)"]
        assert val == pytest.approx(16.0)


class TestZeroResultantPureCouple:
    """SS beam, L = 4 m, varying load over [1, 3] with w1 = 2, w2 = -2.

    The resultant is zero (w1 + w2 = 0) but the load is NOT inert: it acts
    as a pure couple on the equilibrium and still produces internal forces.

    Hand calculation (w(x) = 2 - 2(x - 1) on [1, 3], k = -2):
      M0 = integral(w*x dx) = -4/3 kN·m; W = 0 (pure couple of moment M0).
      Equilibrium: R_B = M0 / L = -1/3 kN; R_A = 0 - (-1/3) = +1/3 kN.
      V(x) = 1/3 - Q(x - 1) with Q(u) = 2u - u^2: V(2) = -2/3 (minimum,
      where w(x) = 0 at x = 2); V = 1/3 at x = 1 and x = 3; 0 beyond.
      M(x) = x/3 - [u*Q(u) - (u^2 - (2/3)u^3)], u = x - 1.
      M = 0 at x = 2, 3, 4; local extremes where V = 0 inside (1, 3):
      u^2 - 2u + 1/3 = 0 => x = 2 ± sqrt(2/3); M there from the exact
      hand formula (max ~= +0.3629, min ~= -0.3629).
    """

    LOADS = [VaryingLoad(1.0, 3.0, 2.0, -2.0)]

    @staticmethod
    def _m_exact(x: float) -> float:
        u = x - 1.0
        q = 2.0 * u - u * u                       # Q(u) = 2u - u^2
        j_about_s = u * u - (2.0 / 3.0) * u ** 3  # w1u^2/2 + k u^3/3
        return x / 3.0 - (u * q - j_about_s)      # R_A = 1/3 kN

    def test_reactions_include_the_couple(self):
        reactions = compute_reactions(4.0, SS, self.LOADS)
        assert reactions["R_left"] == pytest.approx(1.0 / 3.0)
        assert reactions["R_right"] == pytest.approx(-1.0 / 3.0)

    def test_shear_including_interior_stationary_point(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        v, _ = section_forces(4.0, SS, self.LOADS, x)
        np.testing.assert_allclose(
            v, [1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 0.0], atol=1e-12
        )

    def test_moment_at_stations(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        _, m = section_forces(4.0, SS, self.LOADS, x)
        # M(3) = R_A*3 - integral(w(t)*(3-t)dt) = 1 - 4/3 = -1/3 (the right
        # reaction at x = 4 lies to the RIGHT of the section, so it does
        # not contribute); M(4) closes to zero through R_B.
        np.testing.assert_allclose(
            m, [0.0, 1.0 / 3.0, 0.0, -1.0 / 3.0, 0.0], atol=1e-12
        )

    def test_extremes_from_quadratic_roots_and_v_stationary(self):
        result = analyze(4.0, SS, self.LOADS)
        details = result.extreme_details()
        r = math.sqrt(2.0 / 3.0)
        # Min shear at the interior point where w(x) = 0 (x = 2):
        x, val, side = details["Min shear (-)"]
        assert x == pytest.approx(2.0)
        assert val == pytest.approx(-2.0 / 3.0)
        assert side == "point value"
        # M extrema at the two roots of V = 0 inside (1, 3):
        x, val, _side = details["Max moment (+)"]
        assert x == pytest.approx(2.0 - r, rel=1e-9)
        assert val == pytest.approx(self._m_exact(2.0 - r), rel=1e-9)
        x, val, _side = details["Min moment (-)"]
        assert x == pytest.approx(2.0 + r, rel=1e-9)
        assert val == pytest.approx(self._m_exact(2.0 + r), rel=1e-9)


class TestCombinedVaryingWithPointLoad:
    """SS beam, L = 6 m: P = 10 kN at x = 3 plus the CASE A triangular load.

    Hand calculation:
      Total down = 10 + 18 = 28 kN; first moments about 0: 10*3 + 72 = 102.
      R_B = 102/6 = 17 kN; R_A = 28 - 17 = 11 kN.
      V(3-) = 11 - 4.5 = 6.5; V(3+) = -3.5; V(6-) = 11 - 10 - 18 = -17.
      M(3) = 11*3 - (3*4.5 - 9) - 10*0 = 33 - 4.5 = 28.5 kN·m (maximum:
      V = 11 - x^2/2 has no root in (0, 3) and 1 - x^2/2 has no root in
      (3, 6), so no interior stationary point exists).
      M(6) = 66 - 30 - (6*18 - 72) = 0 (closes).
    """

    LOADS = [PointLoad(3.0, 10.0), VaryingLoad(0.0, 6.0, 0.0, 6.0)]

    def test_reactions(self):
        reactions = compute_reactions(6.0, SS, self.LOADS)
        assert reactions["R_left"] == pytest.approx(11.0)
        assert reactions["R_right"] == pytest.approx(17.0)

    def test_shear_at_stations(self):
        x = np.array([0.0, 3.0, 6.0])
        v, _ = section_forces(6.0, SS, self.LOADS, x)
        np.testing.assert_allclose(v, [11.0, -3.5, 0.0])
        v_left, _ = section_forces(6.0, SS, self.LOADS, 6.0, strict_left=True)
        assert v_left == pytest.approx(-17.0)

    def test_maximum_moment_at_the_point_load(self):
        result = analyze(6.0, SS, self.LOADS)
        x, val, _side = result.extreme_details()["Max moment (+)"]
        assert x == pytest.approx(3.0)
        assert val == pytest.approx(28.5)
        _, m = section_forces(6.0, SS, self.LOADS, np.array([6.0]))
        assert m[0] == pytest.approx(0.0, abs=1e-9)


class TestContinuityAtLoadBoundaries:
    """V and M must be continuous at s and e of a varying load (no point
    force or applied moment there). Uses the partial trapezoid on [2, 5]
    of a simply supported beam whose supports are at 0 and 6."""

    LOADS = [VaryingLoad(2.0, 5.0, 2.0, 4.0)]
    EPS = 1e-9

    def _check_continuous(self, x0: float):
        x = np.array([x0 - self.EPS, x0, x0 + self.EPS])
        v, m = section_forces(6.0, SS, self.LOADS, x)
        np.testing.assert_allclose(v, np.repeat(v[1], 3), atol=1e-6)
        np.testing.assert_allclose(m, np.repeat(m[1], 3), atol=1e-6)
        return v[1], m[1]

    def test_continuous_at_start(self):
        v, m = self._check_continuous(2.0)
        assert v == pytest.approx(3.5)
        assert m == pytest.approx(7.0)

    def test_continuous_at_end(self):
        v, m = self._check_continuous(5.0)
        assert v == pytest.approx(-5.5)
        assert m == pytest.approx(5.5)


class TestVaryingLoadValidation:
    def test_degenerate_interval(self):
        with pytest.raises(ValueError, match="varying load must satisfy"):
            analyze(6.0, SS, [VaryingLoad(2.0, 2.0, 1.0, 1.0)])

    def test_interval_beyond_beam(self):
        with pytest.raises(ValueError, match="varying load must satisfy"):
            analyze(6.0, SS, [VaryingLoad(0.0, 7.0, 1.0, 1.0)])

    def test_non_finite_start_intensity(self):
        with pytest.raises(ValueError, match="finite"):
            analyze(6.0, SS, [VaryingLoad(0.0, 2.0, float("nan"), 1.0)])

    def test_non_finite_end_intensity(self):
        with pytest.raises(ValueError, match="finite"):
            analyze(6.0, SS, [VaryingLoad(0.0, 2.0, 1.0, float("inf"))])


def test_plot_smoke_with_varying_load():
    result = analyze(6.0, SS, [VaryingLoad(0.0, 6.0, 0.0, 6.0)])
    fig = plot_diagrams(result)
    assert fig is not None
    assert len(fig.axes) == 3
    import matplotlib.pyplot as plt

    plt.close(fig)
