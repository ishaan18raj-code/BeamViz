"""Matplotlib rendering of the beam schematic, shear-force and moment diagrams."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend for tests and Streamlit

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

from beamviz.beam import BeamResult, DistributedLoad, PointLoad, SupportType, VaryingLoad

# Restrained, consistent palette shared by the schematic and both diagrams.
_PALETTE = {
    "beam": "#37474f",        # structure, zero lines, support symbols
    "ink": "#212121",         # diagram curves and peak markers
    "guide": "#90a4ae",       # dimension lines, annotation frames, gridlines
    "shear_pos": "#2e7d32",
    "shear_neg": "#c62828",
    "moment_pos": "#f9a825",
    "moment_neg": "#1565c0",
    "point_load": "#c62828",
    "udl": "#ef6c00",
    "udl_text": "#e65100",
    "varying": "#8e24aa",
    "moment_symbol": "#6a1b9a",
}
SHEAR_POS, SHEAR_NEG = _PALETTE["shear_pos"], _PALETTE["shear_neg"]
MOMENT_POS, MOMENT_NEG = _PALETTE["moment_pos"], _PALETTE["moment_neg"]


def plot_diagrams(result: BeamResult, invert_bmd: bool = False, figsize=(10.5, 9.0)):
    """Return a Matplotlib figure with beam schematic, SFD and BMD."""
    fig, (ax_beam, ax_v, ax_m) = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=figsize,
        dpi=100,
        layout="constrained",
        gridspec_kw={"height_ratios": [1.15, 3.0, 3.0]},
    )
    fig.suptitle(
        f"BeamViz — {result.support_type.value} · L = {result.length:g} m",
        fontsize=12,
        fontweight="bold",
    )
    _draw_beam_schematic(ax_beam, result)
    _draw_diagram(
        ax_v, result, result.shear, "Shear force diagram — V(x)", "kN", SHEAR_POS, SHEAR_NEG
    )
    _draw_diagram(
        ax_m, result, result.moment, "Bending moment diagram — M(x)", "kN·m", MOMENT_POS, MOMENT_NEG
    )
    if invert_bmd:
        ax_m.invert_yaxis()
    ax_m.set_xlabel("Position along beam x (m)", fontsize=10)
    return fig


def _draw_wall(ax, x: float, at_left: bool, length: float):
    h = 0.7
    ax.plot([x, x], [-h, h], color=_PALETTE["beam"], lw=3, zorder=4)
    d = -0.03 * length if at_left else 0.03 * length
    for yv in np.linspace(-h, h, 8)[:-1]:
        ax.plot([x, x + d], [yv, yv + 0.1], color=_PALETTE["beam"], lw=1, zorder=4)


def _draw_beam_schematic(ax, result: BeamResult):
    L = result.length
    ax.set_xlim(-0.06 * L, 1.06 * L)
    ax.set_ylim(-1.2, 1.5)
    ax.set_yticks([])
    ax.axis("off")

    beam = _PALETTE["beam"]
    ax.plot([0, L], [0, 0], color=beam, lw=5, solid_capstyle="butt", zorder=2)

    support_captions = []
    if result.support_type is SupportType.SIMPLY_SUPPORTED:
        ax.plot([0], [0], marker="^", markersize=13, color=beam, zorder=4)
        ax.plot([L], [0], marker="o", markersize=9, color=beam, mfc="white", zorder=4)
        support_captions = [(0.0, "Pin"), (L, "Roller")]
    elif result.support_type is SupportType.OVERHANGING:
        a, b = result.support_positions
        ax.plot([a], [0], marker="^", markersize=13, color=beam, zorder=4)
        ax.plot([b], [0], marker="o", markersize=9, color=beam, mfc="white", zorder=4)
        support_captions = [(a, "Pin"), (b, "Roller")]
    elif result.support_type is SupportType.CANTILEVER_LEFT:
        _draw_wall(ax, 0.0, at_left=True, length=L)
        support_captions = [(0.0, "Fixed")]
    elif result.support_type is SupportType.CANTILEVER_RIGHT:
        _draw_wall(ax, L, at_left=False, length=L)
        support_captions = [(L, "Fixed")]

    for x_pos, text in support_captions:
        ax.text(x_pos, -0.55, text, ha="center", fontsize=9, color=beam)

    # Span dimension line with the total length.
    ax.annotate(
        "", xy=(0.0, -0.9), xytext=(L, -0.9),
        arrowprops=dict(arrowstyle="<->", color=_PALETTE["guide"], lw=1.0),
    )
    ax.text(
        L / 2.0, -1.05, f"L = {L:g} m", ha="center", va="center",
        fontsize=9, color=_PALETTE["guide"],
    )

    for load in result.loads:
        if isinstance(load, PointLoad):
            ax.annotate(
                "",
                xy=(load.position, 0.02),
                xytext=(load.position, 0.95),
                arrowprops=dict(arrowstyle="-|>", color=_PALETTE["point_load"], lw=2, mutation_scale=16),
            )
            ax.plot(
                [load.position], [0.0], marker="o", markersize=4,
                color=_PALETTE["point_load"], zorder=5,
            )
            ax.text(
                load.position, 1.05, f"{load.magnitude:g} kN",
                ha="center", fontsize=9, color=_PALETTE["point_load"],
            )
        elif isinstance(load, DistributedLoad):
            n = max(int(np.ceil((load.end - load.start) / L * 30)), 6)
            top = 0.5
            for x0 in np.linspace(load.start, load.end, n):
                ax.annotate(
                    "",
                    xy=(x0, 0.02),
                    xytext=(x0, top),
                    arrowprops=dict(arrowstyle="-|>", color=_PALETTE["udl"], lw=1.1, mutation_scale=9),
                )
            ax.plot([load.start, load.end], [top, top], color=_PALETTE["udl"], lw=1.6)
            ax.text(
                (load.start + load.end) / 2, top + 0.12, f"{load.intensity:g} kN/m",
                ha="center", fontsize=9, color=_PALETTE["udl_text"],
            )
        elif isinstance(load, VaryingLoad):
            # Sloping series of arrows whose heights follow the intensity
            # variation w(x) between the two endpoint intensities.
            w1, w2 = load.start_intensity, load.end_intensity
            w_ref = max(abs(w1), abs(w2))
            if w_ref > 0.0:
                n = max(int(np.ceil((load.end - load.start) / L * 30)), 6)
                xs = np.linspace(load.start, load.end, n)
                intensities = w1 + (w2 - w1) * (xs - load.start) / (load.end - load.start)
                tops = 0.02 + 0.48 * np.abs(intensities) / w_ref
                for x0, top in zip(xs, tops):
                    ax.annotate(
                        "",
                        xy=(x0, 0.02),
                        xytext=(x0, top),
                        arrowprops=dict(arrowstyle="-|>", color=_PALETTE["varying"], lw=1.1, mutation_scale=9),
                    )
                ax.plot(xs, tops, color=_PALETTE["varying"], lw=1.6)
                ax.text(
                    (load.start + load.end) / 2, 0.66,
                    f"{w1:g} → {w2:g} kN/m",
                    ha="center", fontsize=9, color=_PALETTE["moment_symbol"],
                )
        else:  # AppliedMoment
            symbol = "↻" if load.magnitude >= 0 else "↺"
            ax.text(load.position, 0.42, symbol, ha="center", va="center",
                    fontsize=17, color=_PALETTE["moment_symbol"])
            ax.text(load.position, 0.75, f"{abs(load.magnitude):g} kN·m",
                    ha="center", fontsize=9, color=_PALETTE["moment_symbol"])


def _draw_diagram(ax, result: BeamResult, y, title: str, unit: str, pos_color, neg_color):
    x = result.x
    ax.axhline(0.0, color=_PALETTE["beam"], lw=1.1, zorder=1)
    ax.fill_between(x, y, 0.0, where=y >= 0, color=pos_color, alpha=0.30, interpolate=True)
    ax.fill_between(x, y, 0.0, where=y <= 0, color=neg_color, alpha=0.30, interpolate=True)
    ax.plot(x, y, color=_PALETTE["ink"], lw=1.9)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=6)
    ax.set_ylabel(unit, fontsize=10)
    ax.grid(True, alpha=0.25, lw=0.6)
    ax.tick_params(labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    support_markers = {0.0, result.length} | set(result.support_positions or ())
    for pos in sorted(support_markers):
        ax.axvline(pos, color=_PALETTE["guide"], lw=0.8, ls="--", alpha=0.6, zorder=0)

    imax, imin = int(np.argmax(y)), int(np.argmin(y))
    value_box = dict(
        boxstyle="round,pad=0.25", fc="white", ec=_PALETTE["guide"], lw=0.6, alpha=0.9
    )
    if y[imax] != 0:
        ax.plot(x[imax], y[imax], "o", color=_PALETTE["ink"], ms=4)
        ax.annotate(
            f"{y[imax]:+,.2f}", (x[imax], y[imax]), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=8, bbox=value_box,
        )
    if y[imin] != 0 and imin != imax:
        ax.plot(x[imin], y[imin], "o", color=_PALETTE["ink"], ms=4)
        ax.annotate(
            f"{y[imin]:+,.2f}", (x[imin], y[imin]), xytext=(0, -15),
            textcoords="offset points", ha="center", fontsize=8, bbox=value_box,
        )
