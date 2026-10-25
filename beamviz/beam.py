"""Core beam analysis engine for BeamViz.

Computes support reactions and the shear-force / bending-moment
distributions for a slender 1-D beam subjected to point loads,
uniformly distributed loads (UDL) and concentrated moments.

Sign conventions
----------------
x axis          : 0 at the left end of the beam, positive to the right (m)
Applied loads   : positive magnitude acts DOWNWARDS (kN, kN/m)
Applied moments : positive magnitude acts CLOCKWISE (kN·m)
Shear V(x)      : positive when the left segment tends to move up (kN)
Moment M(x)     : positive = sagging (tension on the bottom fibre) (kN·m)
Support moments : reported counter-clockwise positive (labelled in the UI)
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from enum import Enum

import numpy as np

__all__ = [
    "SupportType",
    "PointLoad",
    "DistributedLoad",
    "VaryingLoad",
    "AppliedMoment",
    "BeamResult",
    "compute_reactions",
    "section_forces",
    "analyze",
]


class SupportType(str, Enum):
    """Supported boundary conditions."""

    SIMPLY_SUPPORTED = "Simply supported (pin + roller)"
    CANTILEVER_LEFT = "Cantilever (fixed at left end)"
    CANTILEVER_RIGHT = "Cantilever (fixed at right end)"
    OVERHANGING = "Overhanging (two simple supports)"


@dataclass(frozen=True)
class PointLoad:
    """Concentrated vertical force (kN). ``magnitude`` > 0 acts downwards."""

    position: float
    magnitude: float


@dataclass(frozen=True)
class DistributedLoad:
    """Uniformly distributed load over ``[start, end]`` (kN/m, + = downwards)."""

    start: float
    end: float
    intensity: float


@dataclass(frozen=True)
class VaryingLoad:
    """Linearly varying distributed load over ``[start, end]``.

    Intensity ``w(x) = w1 + (w2 - w1) * (x - start) / (end - start)``
    (kN/m, positive = downwards). ``w1 == w2`` reduces exactly to a UDL;
    a zero endpoint gives a triangular load; ``w1 + w2 == 0`` gives a
    zero-resultant load (a pure couple), which is handled exactly — it
    still produces internal forces and reaction moments.
    """

    start: float
    end: float
    start_intensity: float
    end_intensity: float

    def intensity_at(self, x: float) -> float:
        """Load intensity w(x) at position ``x`` (kN/m)."""
        span = self.end - self.start
        return self.start_intensity + (
            (self.end_intensity - self.start_intensity) * (x - self.start) / span
        )

    def resultant_and_first_moment(self) -> tuple:
        """Exact ``(W, M0)`` for equilibrium: resultant (kN) and first
        moment about x = 0 (kN·m), i.e. ``W = ∫w dx`` and ``M0 = ∫w·x dx``.

        Closed-form (no numerical integration). ``M0`` equals ``W * c``
        with ``c`` the resultant centroid, and stays valid when ``W = 0``
        (pure couple), so the zero-resultant case needs no special casing.
        """
        d = self.end - self.start
        w1, w2 = self.start_intensity, self.end_intensity
        k = (w2 - w1) / d
        s = self.start
        total = (w1 + w2) * d / 2.0
        m0 = (
            w1 * s * d
            + w1 * d * d / 2.0
            + k * s * d * d / 2.0
            + k * d ** 3 / 3.0
        )
        return total, m0

    def centroid_position(self):
        """Centroid of the resultant measured from x = 0, or ``None`` when
        the resultant is zero (``w1 + w2 == 0``)."""
        denom = self.start_intensity + self.end_intensity
        if denom == 0.0:
            return None
        span = self.end - self.start
        xbar = span * (self.start_intensity + 2.0 * self.end_intensity) / (3.0 * denom)
        return self.start + xbar


@dataclass(frozen=True)
class AppliedMoment:
    """Concentrated moment (kN·m, + = clockwise)."""

    position: float
    magnitude: float


Load = PointLoad | DistributedLoad | AppliedMoment | VaryingLoad


@dataclass
class BeamResult:
    """Complete analysis output for one beam configuration."""

    length: float
    support_type: SupportType
    reactions: dict
    loads: tuple
    x: np.ndarray
    shear: np.ndarray
    moment: np.ndarray
    support_positions: tuple = ()

    def extremes(self) -> dict:
        """Return peak values as ``{label: (x_location, value)}``.

        Extrema are computed from exact one-sided limits at every
        discontinuity and beam boundary (plus stationary points of M where
        V = 0), not from the plotted sample grid — so boundary extrema such
        as V(L-) are exact.
        """
        return {
            label: (x, value)
            for label, (x, value, _side) in self.extreme_details().items()
        }

    def extreme_details(self) -> dict:
        """Peak values together with the one-sided basis used for each.

        Returns ``{label: (x_location, value, basis)}`` where ``basis`` is
        ``"point value"``, ``"left-hand limit"`` or ``"right-hand limit"``.
        A left-hand limit at the right support, e.g. V(L-), is the shear
        immediately *before* the support reaction is included; the "point
        value" at x = L includes the reaction (V = 0 there). For overhanging
        beams the same distinction applies at each interior support x = a
        and x = b.
        """
        actions = _reaction_actions(
            self.support_type, self.length, self.reactions, self.support_positions
        )
        v_cands, m_cands = _extreme_candidates(self.length, actions, self.loads)
        return {
            "Max shear (+)": _pick(v_cands, want_max=True),
            "Min shear (-)": _pick(v_cands, want_max=False),
            "Max moment (+)": _pick(m_cands, want_max=True),
            "Min moment (-)": _pick(m_cands, want_max=False),
        }


def _validate(length: float, loads) -> None:
    """Raise ``ValueError`` with a user-friendly message for invalid input."""
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Beam length must be a positive, finite number.")

    for i, load in enumerate(loads):
        name = f"load #{i + 1}"
        if isinstance(load, PointLoad):
            if not np.isfinite(load.position) or not (0.0 <= load.position <= length):
                raise ValueError(
                    f"{name}: point load position must lie within [0, {length:g}] m."
                )
            if not np.isfinite(load.magnitude):
                raise ValueError(f"{name}: point load magnitude must be finite.")
        elif isinstance(load, DistributedLoad):
            if not (
                np.isfinite(load.start)
                and np.isfinite(load.end)
                and np.isfinite(load.intensity)
            ):
                raise ValueError(f"{name}: UDL inputs must be finite numbers.")
            if not (0.0 <= load.start < load.end <= length):
                raise ValueError(
                    f"{name}: UDL must satisfy 0 <= start < end <= {length:g} m."
                )
        elif isinstance(load, VaryingLoad):
            if not (
                np.isfinite(load.start)
                and np.isfinite(load.end)
                and np.isfinite(load.start_intensity)
                and np.isfinite(load.end_intensity)
            ):
                raise ValueError(f"{name}: varying load inputs must be finite numbers.")
            if not (0.0 <= load.start < load.end <= length):
                raise ValueError(
                    f"{name}: varying load must satisfy 0 <= start < end <= "
                    f"{length:g} m."
                )
        elif isinstance(load, AppliedMoment):
            if not np.isfinite(load.position) or not (0.0 <= load.position <= length):
                raise ValueError(
                    f"{name}: moment position must lie within [0, {length:g}] m."
                )
            if not np.isfinite(load.magnitude):
                raise ValueError(f"{name}: moment magnitude must be finite.")
        else:
            raise TypeError(f"{name}: unknown load type {type(load).__name__}.")


def _validate_support_geometry(
    length: float, support_type: SupportType, support_positions
) -> tuple:
    """Validate and normalise the support positions for the configuration.

    Returns the tuple of reaction-force positions. For simply supported
    beams that is ``(0, L)``; for cantilevers the single fixed end; for
    overhanging beams the user-supplied pair ``(a, b)``.
    """
    if support_type is SupportType.OVERHANGING:
        if support_positions is None:
            raise ValueError(
                "Overhanging beams require support positions (a, b) measured "
                "from the left beam end."
            )
        try:
            a, b = support_positions
        except (TypeError, ValueError):
            raise ValueError(
                "Support positions must be a pair (a, b) of numbers."
            ) from None
        a, b = float(a), float(b)
        if not (np.isfinite(a) and np.isfinite(b)):
            raise ValueError("Support positions must be finite numbers.")
        if not (0.0 <= a < b <= length):
            raise ValueError(
                f"Overhanging geometry must satisfy 0 <= a < b <= {length:g} "
                f"(got a = {a:g}, b = {b:g})."
            )
        if a == 0.0 and b == length:
            raise ValueError(
                "Supports at both beam ends (a = 0, b = L) make this a simply "
                "supported beam; select the simply supported configuration "
                "instead."
            )
        return (a, b)
    if support_type is SupportType.CANTILEVER_LEFT:
        return (0.0,)
    if support_type is SupportType.CANTILEVER_RIGHT:
        return (length,)
    if support_type is SupportType.SIMPLY_SUPPORTED:
        return (0.0, length)
    raise ValueError(f"Unsupported support type: {support_type!r}")


def compute_reactions(
    length: float,
    support_type: SupportType,
    loads,
    support_positions=None,
) -> dict:
    """Return support reactions from static equilibrium.

    Keys: ``R_left``/``R_right`` (kN, + = upward) and ``M_left``/``M_right``
    (kN·m, + = counter-clockwise). For overhanging beams ``R_left`` is the
    reaction at x = a and ``R_right`` at x = b; ``support_positions`` must
    then be the pair ``(a, b)``.
    """
    _validate(length, loads)
    positions = _validate_support_geometry(length, support_type, support_positions)

    point_total = sum(l.magnitude for l in loads if isinstance(l, PointLoad))
    udl_total = sum(
        l.intensity * (l.end - l.start) for l in loads if isinstance(l, DistributedLoad)
    )
    # First moment of downward loads about x = 0.
    point_first_moment = sum(
        l.magnitude * l.position for l in loads if isinstance(l, PointLoad)
    )
    udl_first_moment = sum(
        l.intensity * (l.end - l.start) * (l.start + l.end) / 2.0
        for l in loads
        if isinstance(l, DistributedLoad)
    )
    # Linearly varying loads: exact resultant W = ∫w dx and exact first
    # moment M0 = ∫w·x dx about x = 0 (M0 = W·c, valid even when W = 0).
    varying_total = 0.0
    varying_first_moment = 0.0        # Σ ∫w(t)·t dt
    varying_first_moment_right = 0.0  # Σ ∫w(t)·(L − t) dt
    for load in loads:
        if isinstance(load, VaryingLoad):
            w_total, m0 = load.resultant_and_first_moment()
            varying_total += w_total
            varying_first_moment += m0
            varying_first_moment_right += length * w_total - m0
    moment_total = sum(
        l.magnitude for l in loads if isinstance(l, AppliedMoment)
    )  # clockwise-positive input
    total_down = point_total + udl_total + varying_total

    if support_type is SupportType.SIMPLY_SUPPORTED:
        # Sum of moments about the left support (CCW positive):
        # R_right*L - sum(P*a) - sum(UDL) - sum(M_cw) = 0
        r_right = (
            point_first_moment
            + udl_first_moment
            + varying_first_moment
            + moment_total
        ) / length
        r_left = total_down - r_right
        return {"R_left": r_left, "R_right": r_right}

    if support_type is SupportType.CANTILEVER_LEFT:
        # Fixed-end moment (CCW positive) balancing all actions about x = 0.
        m_left = (
            point_first_moment
            + udl_first_moment
            + varying_first_moment
            + moment_total
        )
        return {"R_left": total_down, "M_left": m_left}

    if support_type is SupportType.CANTILEVER_RIGHT:
        point_first_moment_right = sum(
            l.magnitude * (length - l.position)
            for l in loads
            if isinstance(l, PointLoad)
        )
        udl_first_moment_right = sum(
            l.intensity * (l.end - l.start) * (length - (l.start + l.end) / 2.0)
            for l in loads
            if isinstance(l, DistributedLoad)
        )
        m_right = -(
            point_first_moment_right
            + udl_first_moment_right
            + varying_first_moment_right
        ) + moment_total
        return {"R_right": total_down, "M_right": m_right}

    if support_type is SupportType.OVERHANGING:
        # Two simple (vertical) supports at x = a and x = b; loads on the
        # overhangs are fully included. Sum of moments about the left
        # support (CCW positive):
        #   R_right*(b - a) - sum(P*(p - a)) - sum(W*(c - a)) - sum(M_cw) = 0
        # with p the load position, c the UDL centroid. A downward load on
        # the left overhang (p < a) therefore reduces R_right and can make
        # it negative (a downward reaction), which is statically admissible.
        a, b = positions
        lever = b - a
        # Varying loads: exact moment about a is ∫w(t)·(t − a)dt = M0 − a·W
        # (equals W·(c − a); also valid for zero-resultant pure couples).
        varying_about_a = 0.0
        for load in loads:
            if isinstance(load, VaryingLoad):
                w_total, m0 = load.resultant_and_first_moment()
                varying_about_a += m0 - a * w_total
        moment_about_a = (
            sum(
                l.magnitude * (l.position - a)
                for l in loads
                if isinstance(l, PointLoad)
            )
            + sum(
                l.intensity * (l.end - l.start) * ((l.start + l.end) / 2.0 - a)
                for l in loads
                if isinstance(l, DistributedLoad)
            )
            + varying_about_a
            + moment_total
        )
        r_right = moment_about_a / lever
        r_left = total_down - r_right
        return {"R_left": r_left, "R_right": r_right}

    raise ValueError(f"Unsupported support type: {support_type!r}")


def _reaction_actions(support_type: SupportType, length: float, reactions: dict, support_positions) -> list:
    """Normalise support reactions into internal-force actions.

    Returns a list of ``(kind, position, value)`` tuples where ``kind`` is
    ``"force"`` (upward reaction, kN) or ``"moment"`` (counter-clockwise
    support moment, kN·m). The right-end fixed-support moment is reported
    in ``reactions`` but is deliberately NOT applied inside ``[0, L]`` —
    the internal moment computed from the loads already carries it at the
    wall (behaviour preserved from the cantilever implementation).
    """
    actions = []
    if support_type is SupportType.SIMPLY_SUPPORTED:
        actions.append(("force", 0.0, reactions["R_left"]))
        actions.append(("force", length, reactions["R_right"]))
    elif support_type is SupportType.CANTILEVER_LEFT:
        actions.append(("force", 0.0, reactions["R_left"]))
        actions.append(("moment", 0.0, reactions["M_left"]))
    elif support_type is SupportType.CANTILEVER_RIGHT:
        actions.append(("force", length, reactions["R_right"]))
    elif support_type is SupportType.OVERHANGING:
        a, b = support_positions
        actions.append(("force", float(a), reactions["R_left"]))
        actions.append(("force", float(b), reactions["R_right"]))
    return actions


def _internal_forces(reaction_actions, loads, length: float, x, strict_left: bool = False) -> tuple:
    """Evaluate shear ``V`` and bending moment ``M`` at ``x`` (scalar/array).

    ``reaction_actions`` is the list produced by :func:`_reaction_actions`.
    The bending moment diagram uses the sagging-positive convention. A
    clockwise-positive applied moment therefore produces a jump of ``+M`` at
    its position; a counter-clockwise support moment produces a jump of
    ``-M`` at its position.

    With ``strict_left=True`` the evaluation returns the *left-hand limit*
    at ``x``: point loads, applied moments and reaction forces located
    exactly at ``x`` are excluded from the evaluation, yielding the exact
    one-sided values V(x-) and M(x-). UDL contributions are unchanged
    because V and M are continuous across UDL edges.
    """
    x_arr = np.asarray(x, dtype=float)
    scalar = x_arr.ndim == 0
    if scalar:
        x_arr = x_arr.reshape(1)

    v = np.zeros(x_arr.shape, dtype=float)
    m = np.zeros(x_arr.shape, dtype=float)

    for kind, pos, value in reaction_actions:
        if strict_left:
            at_pos = x_arr > pos
        else:
            at_pos = x_arr >= pos
        if kind == "force":
            v += np.where(at_pos, value, 0.0)
            m += np.where(at_pos, value * (x_arr - pos), 0.0)
        else:  # counter-clockwise support moment: jump of -M at pos
            m -= np.where(at_pos, value, 0.0)

    for load in loads:
        if isinstance(load, PointLoad):
            if strict_left:
                beyond = x_arr > load.position
            else:
                beyond = x_arr >= load.position
            v -= np.where(beyond, load.magnitude, 0.0)
            m -= np.where(beyond, load.magnitude * (x_arr - load.position), 0.0)
        elif isinstance(load, DistributedLoad):
            s, e, w = load.start, load.end, load.intensity
            v -= np.where(
                x_arr <= s,
                0.0,
                np.where(x_arr >= e, w * (e - s), w * (x_arr - s)),
            )
            moment_inner = 0.5 * w * (x_arr - s) ** 2
            moment_outer = w * (e - s) * (x_arr - (s + e) / 2.0)
            m -= np.where(
                x_arr <= s, 0.0, np.where(x_arr < e, moment_inner, moment_outer)
            )
        elif isinstance(load, VaryingLoad):
            # w(x) = w1 + k(x - s), k = (w2 - w1)/(e - s). With u = x - s
            # clamped to [0, span]:
            #   Q(u)  = w1*u + k*u^2/2      accumulated load from s to x
            #   J(u)  = w1*u^2/2 + k*u^3/3  first moment of that load about s
            #   M_sec = u*Q(u) - J(u)       = w1u^2/2 + k u^3/6, the first
            #                                 moment about the section at x
            # Beyond e the exact resultant/first-moment pair (W, M0) is used:
            #   V -= W and M -= x*W - M0, identical to -W*(x - c) when W != 0
            # and exact for the zero-resultant (pure couple) case too.
            s, e = load.start, load.end
            w1, w2 = load.start_intensity, load.end_intensity
            span = e - s
            k = (w2 - w1) / span
            w_total, m0 = load.resultant_and_first_moment()
            u = np.clip(x_arr - s, 0.0, span)
            q = w1 * u + 0.5 * k * u ** 2
            j_about_s = w1 * u ** 2 / 2.0 + k * u ** 3 / 3.0
            m_section = u * q - j_about_s
            v -= np.where(
                x_arr <= s, 0.0, np.where(x_arr >= e, w_total, q)
            )
            m -= np.where(
                x_arr <= s,
                0.0,
                np.where(x_arr >= e, x_arr * w_total - m0, m_section),
            )
        elif isinstance(load, AppliedMoment):
            if strict_left:
                beyond = x_arr > load.position
            else:
                beyond = x_arr >= load.position
            m += np.where(beyond, load.magnitude, 0.0)

    if scalar:
        return float(v[0]), float(m[0])
    return v, m


def section_forces(
    length,
    support_type,
    loads,
    x,
    strict_left: bool = False,
    support_positions=None,
):
    """Public helper: reactions + internal forces at ``x`` (scalar or array).

    With ``strict_left=True`` the one-sided left-hand limits V(x-) and
    M(x-) are returned instead of the right-continuous values. For
    overhanging beams pass ``support_positions=(a, b)``.
    """
    reactions = compute_reactions(
        length, support_type, loads, support_positions=support_positions
    )
    positions = _validate_support_geometry(length, support_type, support_positions)
    actions = _reaction_actions(support_type, length, reactions, positions)
    return _internal_forces(actions, loads, length, x, strict_left=strict_left)


def analyze(
    length: float,
    support_type: SupportType,
    loads,
    samples: int = 1201,
    support_positions=None,
) -> BeamResult:
    """Run the full analysis and return a :class:`BeamResult`.

    The sample grid is refined around every discontinuity (point loads,
    applied moments, UDL edges and support reactions) so that jumps are
    captured cleanly. For overhanging beams pass
    ``support_positions=(a, b)``.
    """
    _validate(length, loads)
    positions = _validate_support_geometry(length, support_type, support_positions)
    reactions = compute_reactions(
        length, support_type, loads, support_positions=positions
    )
    actions = _reaction_actions(support_type, length, reactions, positions)

    xs = np.linspace(0.0, length, max(int(samples), 2))
    eps = max(length * 1e-6, 1e-12)
    extras = []
    discontinuities = _discontinuity_positions(loads) + [
        pos for kind, pos, _value in actions
    ]
    for d in discontinuities:
        if 0.0 < d < length:
            extras.extend((d - eps, d, d + eps))
        elif d <= 0.0:
            # Discontinuity at the left end: refine just right of x = 0 so
            # the jump there (e.g. a point load sitting on the support) is
            # drawn vertically.
            extras.append(eps)
        elif d >= length:
            # Discontinuity at the right end: refine just left of x = L so
            # the one-sided limit V(L-) is represented in the sampled arrays.
            extras.append(length - eps)
    if extras:
        xs = np.unique(np.concatenate([xs, np.asarray(extras, dtype=float)]))

    shear, moment = _internal_forces(actions, loads, length, xs)

    return BeamResult(
        length=length,
        support_type=support_type,
        reactions=reactions,
        loads=tuple(loads),
        x=xs,
        shear=shear,
        moment=moment,
        support_positions=positions,
    )


def _discontinuity_positions(loads) -> list:
    positions = []
    for load in loads:
        if isinstance(load, (PointLoad, AppliedMoment)):
            positions.append(load.position)
        elif isinstance(load, (DistributedLoad, VaryingLoad)):
            positions.extend((load.start, load.end))
    return positions


_SIDE_VALUE = "point value"
_SIDE_LEFT = "left-hand limit"
_SIDE_RIGHT = "right-hand limit"


def _pick(candidates, want_max: bool) -> tuple:
    """Select the extreme candidate; first occurrence wins on exact ties
    (matching the previous numpy argmax/argmin behaviour)."""
    best = candidates[0]
    for cand in candidates[1:]:
        if want_max and cand[0] > best[0]:
            best = cand
        elif not want_max and cand[0] < best[0]:
            best = cand
    return (best[1], best[0], best[2])


def _extreme_candidates(length: float, reaction_actions, loads) -> tuple:
    """Exact one-sided candidate values for the V and M extrema.

    V is piecewise linear, so its extrema can only occur at one-sided
    limits around discontinuities (point loads, applied moments, UDL
    edges, support reactions — including interior overhang supports) and
    at the beam boundaries. M is piecewise quadratic; its interior
    stationary points (where V = 0 inside a uniformly loaded span) are
    added analytically. No reliance on the sample grid, so boundary
    extrema such as V(L-) are exact.
    """
    stations = {0.0, length}
    for load in loads:
        if isinstance(load, (PointLoad, AppliedMoment)):
            if 0.0 <= load.position <= length:
                stations.add(float(load.position))
        elif isinstance(load, (DistributedLoad, VaryingLoad)):
            if 0.0 <= load.start <= length:
                stations.add(float(load.start))
            if 0.0 <= load.end <= length:
                stations.add(float(load.end))
    for _kind, pos, _value in reaction_actions:
        if 0.0 <= pos <= length:
            stations.add(float(pos))
    stations = sorted(stations)

    v_cands, m_cands = [], []
    last = len(stations) - 1
    for i, d in enumerate(stations):
        v_r, m_r = _internal_forces(reaction_actions, loads, length, d)
        if i == 0:
            # Left end of the domain: only the right-hand value exists.
            v_cands.append((v_r, d, _SIDE_VALUE))
            m_cands.append((m_r, d, _SIDE_VALUE))
            continue
        v_l, m_l = _internal_forces(reaction_actions, loads, length, d, strict_left=True)
        if i == last:
            # Right end: the point value includes the support reaction,
            # which is a different quantity from the limit V(L-).
            v_cands.append((v_l, d, _SIDE_LEFT))
            m_cands.append((m_l, d, _SIDE_LEFT))
            v_cands.append((v_r, d, _SIDE_VALUE))
            m_cands.append((m_r, d, _SIDE_VALUE))
        else:
            # Judge the basis per quantity: at a point load V jumps but M
            # is continuous, so M must be labelled a plain point value.
            if v_l == v_r:
                v_cands.append((v_r, d, _SIDE_VALUE))
            else:
                v_cands.append((v_l, d, _SIDE_LEFT))
                v_cands.append((v_r, d, _SIDE_RIGHT))
            if m_l == m_r:
                m_cands.append((m_r, d, _SIDE_VALUE))
            else:
                m_cands.append((m_l, d, _SIDE_LEFT))
                m_cands.append((m_r, d, _SIDE_RIGHT))

    # Interior stationary points. Stations include every load edge, so the
    # set of distributed loads fully covering the open interval (a, b) is
    # constant there. On such an interval V(x) = v_a - sum of accumulated
    # distributed load, which is linear when only UDLs act and quadratic
    # when a linearly varying load acts (dV/dx = -w(x)).
    for a, b in zip(stations[:-1], stations[1:]):
        if b <= a:
            continue
        v_a, _ = _internal_forces(reaction_actions, loads, length, a)

        sum_w = 0.0        # Σ w (UDLs) + Σ w1 (varying loads)
        sum_k_half = 0.0   # Σ k/2 over varying loads  -> quadratic coeff
        sum_k_s = 0.0      # Σ k·s over varying loads
        sum_k_geom = 0.0   # Σ (k/2)(2·a·s − a²) over varying loads
        for load in loads:
            if isinstance(load, DistributedLoad):
                if load.start <= a and load.end >= b:
                    sum_w += load.intensity
            elif isinstance(load, VaryingLoad):
                if load.start <= a and load.end >= b:
                    span = load.end - load.start
                    k = (load.end_intensity - load.start_intensity) / span
                    w1 = load.start_intensity
                    sum_w += w1
                    sum_k_half += k / 2.0
                    sum_k_s += k * load.start
                    sum_k_geom += (k / 2.0) * (2.0 * a * load.start - a * a)
                    # V stationary point where the intensity vanishes:
                    # dV/dx = -w(x) = 0  =>  x = s - w1/k (if inside).
                    if k != 0.0:
                        x0 = load.start - w1 / k
                        if a < x0 < b:
                            v_x0, _ = _internal_forces(
                                reaction_actions, loads, length, x0
                            )
                            v_cands.append((v_x0, x0, _SIDE_VALUE))

        if abs(sum_k_half) <= 1e-12:
            # V linear on (a, b): M is stationary where V = 0.
            if abs(sum_w) <= 1e-12:
                continue  # V constant; M extrema lie at the interval ends
            x_star = a + v_a / sum_w
            if a < x_star < b:
                _, m_star = _internal_forces(reaction_actions, loads, length, x_star)
                m_cands.append((m_star, x_star, _SIDE_VALUE))
        else:
            # V quadratic on (a, b): V(x) = p2*x^2 + p1*x + p0; solve V = 0
            # exactly with the quadratic formula (robust, no sampling).
            p2 = -sum_k_half
            p1 = -sum_w + sum_k_s
            p0 = v_a + a * sum_w - sum_k_geom
            disc = p1 * p1 - 4.0 * p2 * p0
            if disc < 0.0:
                continue
            root = math.sqrt(disc)
            for x_root in ((-p1 - root) / (2.0 * p2), (-p1 + root) / (2.0 * p2)):
                if a < x_root < b:
                    _, m_star = _internal_forces(
                        reaction_actions, loads, length, x_root
                    )
                    m_cands.append((m_star, x_root, _SIDE_VALUE))

    return v_cands, m_cands
