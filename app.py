"""BeamViz — interactive Streamlit app for shear force and bending moment diagrams.

Run with:  python -m streamlit run app.py

Workflow: 1) beam setup · 2) define loads · 3) reactions · 4) diagrams · 5) extremes.
"""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from beamviz.beam import (
    AppliedMoment,
    DistributedLoad,
    PointLoad,
    SupportType,
    VaryingLoad,
    analyze,
)
from beamviz.plots import plot_diagrams

st.set_page_config(
    page_title="BeamViz — SFD & BMD tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal styling only: card-like metric tiles and slightly tighter spacing.
st.markdown(
    """
    <style>
        div[data-testid="stMetric"] {
            background: transparent;
            border: none;
            border-radius: 8px;
            padding: 10px 14px;
        }
        div[data-testid="stMetricLabel"] p { font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 BeamViz")
st.markdown(
    "Interactive **shear-force (SFD)** and **bending-moment (BMD)** diagrams "
    "for statically determinate beams"
)
st.caption(
    "Educational tool for coursework and demonstration — not suitable for "
    "certified structural design or real-world safety approval."
)

DEMO_LOADS = pd.DataFrame(
    {
        "type": ["Point load", "UDL"],
        "position": [3.0, 4.0],
        "value": [10.0, 2.0],
        "end": [0.0, 6.0],
        "w2": [0.0, 0.0],
    }
)

def reset_demo_loads() -> None:
    """Button callback: restore the preloaded demonstration load set."""
    st.session_state["loads_editor"] = DEMO_LOADS.copy()


with st.sidebar:
    st.header("1 · Beam setup")
    support_type = st.selectbox(
        "Beam type",
        options=list(SupportType),
        format_func=lambda s: s.value,
        help="Boundary conditions supported by the solver.",
    )
    length = st.number_input(
        "Beam length L (m)",
        min_value=0.1,
        max_value=100.0,
        value=6.0,
        step=0.1,
        help="Total physical length, measured from the left beam end.",
    )
    support_positions = None
    if support_type is SupportType.OVERHANGING:
        pos_a = st.number_input(
            "Left support position a (m)",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.1,
            key="overhang_a",
            help="First simple support, measured from the left beam end.",
        )
        pos_b = st.number_input(
            "Right support position b (m)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            key="overhang_b",
            help="Second simple support; the beam may overhang beyond it.",
        )
        support_positions = (pos_a, pos_b)
        if pos_a >= pos_b or pos_b > length:
            st.warning(
                "Overhanging geometry must satisfy 0 ≤ a < b ≤ L — the "
                "detailed error from the analysis is shown below until fixed."
            )
    samples = st.slider(
        "Curve resolution",
        min_value=201,
        max_value=4001,
        value=1201,
        step=200,
        help="Number of sample points used to draw the SFD/BMD curves.",
    )
    invert_bmd = st.checkbox(
        "Invert BMD axis (positive moment drawn downward)",
        value=False,
        help="Civil-engineering convention: sagging moment plotted downward.",
    )
    st.divider()
    st.button(
        "↺ Reset to demo loads",
        on_click=reset_demo_loads,
        use_container_width=True,
        help="Restore the preloaded demonstration example.",
    )
    st.divider()
    st.markdown(
        "**Sign conventions**\n"
        "- Applied loads: + = **downward**\n"
        "- Applied moments: + = **clockwise**\n"
        "- Shear V: + = left segment pushed up\n"
        "- Bending M: + = **sagging**\n"
        "- Support moments: reported **CCW positive**"
    )

st.subheader("2 · Define loads")
st.markdown(
    "Add rows as needed and pick the **Type** in each row:\n\n"
    "| Type | Position | Value | UDL end | w2 |\n"
    "|---|---|---|---|---|\n"
    "| **Point load** | application point (m) | magnitude (kN, + down) | — | — |\n"
    "| **UDL** | start (m) | intensity (kN/m, + down) | end (m) | — |\n"
    "| **Moment** | application point (m) | magnitude (kN·m, + clockwise) | — | — |\n"
    "| **Varying distributed** | start (m) | start intensity w1 (kN/m) | end (m) | end intensity w2 (kN/m) |"
)
loads_df = st.data_editor(
    DEMO_LOADS.copy(),
    key="loads_editor",
    num_rows="dynamic",
    column_config={
        "type": st.column_config.SelectboxColumn(
            "Type", options=["Point load", "UDL", "Moment", "Varying distributed load"]
        ),
        "position": st.column_config.NumberColumn("Position (m)", min_value=0.0, step=0.1),
        "value": st.column_config.NumberColumn("Magnitude / start intensity", step=0.1),
        "end": st.column_config.NumberColumn("UDL end (m)", min_value=0.0, step=0.1),
        "w2": st.column_config.NumberColumn("End intensity w2 (kN/m)", step=0.1),
    },
    hide_index=True,
)


def parse_loads(df: pd.DataFrame):
    """Convert the editor table into load objects plus a list of input errors."""
    loads, errors = [], []
    if df is None or len(df) == 0:
        return loads, errors
    for i, row in df.iterrows():
        raw_type = row["type"]
        if pd.isna(raw_type) or str(raw_type).strip() == "":
            continue
        ltype = str(raw_type).strip()
        try:
            position = float(row["position"])
            value = float(row["value"])
        except (TypeError, ValueError):
            errors.append(f"Row {i + 1}: 'Position' and 'Value' must be numbers.")
            continue
        if not (math.isfinite(position) and math.isfinite(value)):
            errors.append(f"Row {i + 1}: 'Position' and 'Value' must be finite numbers.")
            continue
        if ltype == "Point load":
            loads.append(PointLoad(position=position, magnitude=value))
        elif ltype == "UDL":
            try:
                end = float(row["end"])
            except (TypeError, ValueError):
                errors.append(f"Row {i + 1}: UDL needs a numeric 'UDL end'.")
                continue
            if not math.isfinite(end):
                errors.append(f"Row {i + 1}: UDL end must be a finite number.")
                continue
            loads.append(DistributedLoad(start=position, end=end, intensity=value))
        elif ltype == "Varying distributed load":
            try:
                end = float(row["end"])
                w2 = float(row["w2"])
            except (TypeError, ValueError):
                errors.append(
                    f"Row {i + 1}: Varying load needs a numeric 'UDL end' and 'w2'."
                )
                continue
            if not (math.isfinite(end) and math.isfinite(w2)):
                errors.append(
                    f"Row {i + 1}: Varying load 'UDL end' and 'w2' must be finite."
                )
                continue
            loads.append(
                VaryingLoad(
                    start=position,
                    end=end,
                    start_intensity=value,
                    end_intensity=w2,
                )
            )
        elif ltype == "Moment":
            loads.append(AppliedMoment(position=position, magnitude=value))
        else:
            errors.append(
                f"Row {i + 1}: unknown load type {ltype!r} "
                "(use Point load, UDL or Moment)."
            )
    return loads, errors


loads, errors = parse_loads(loads_df)

if errors:
    st.error("Please fix the load inputs:\n\n" + "\n".join(f"- {e}" for e in errors))
    st.stop()

try:
    result = analyze(
        length,
        support_type,
        loads,
        samples=samples,
        support_positions=support_positions,
    )
except (ValueError, TypeError) as exc:
    st.error(f"Invalid beam setup: {exc}")
    st.stop()

st.divider()
st.subheader("3 · Support reactions")
st.caption(
    "Signs: reactions are positive upward; fixed-end moments are reported "
    "counter-clockwise positive."
)
REACTION_BASE = {
    "R_left": "Left support reaction",
    "R_right": "Right support reaction",
    "M_left": "Left fixed-end moment",
    "M_right": "Right fixed-end moment",
}
reaction_positions = {"R_left": 0.0, "M_left": 0.0, "R_right": result.length, "M_right": result.length}
if result.support_type is SupportType.OVERHANGING and len(result.support_positions) == 2:
    reaction_positions["R_left"] = result.support_positions[0]
    reaction_positions["R_right"] = result.support_positions[1]
reaction_cols = st.columns(len(result.reactions))
for col, (key, value) in zip(reaction_cols, result.reactions.items()):
    unit = "kN" if key.startswith("R") else "kN·m"
    col.metric(
        f"{REACTION_BASE.get(key, key)} — x = {reaction_positions[key]:g} m ({unit})",
        f"{value:,.3f}",
        help="Positive = upward (forces) or counter-clockwise (moments).",
    )

st.divider()
st.subheader("4 · Shear-force and bending-moment diagrams")
st.pyplot(plot_diagrams(result, invert_bmd=invert_bmd))

st.divider()
st.subheader("5 · Extreme values")
extreme_cols = st.columns(4)
for col, (label, (loc, val, basis)) in zip(extreme_cols, result.extreme_details().items()):
    unit = "kN" if "shear" in label else "kN·m"
    col.metric(
        label,
        f"{val:,.3f} {unit}",
        delta=f"at x = {loc:.2f} m · {basis}",
        delta_color="off",
        help="The basis states whether the value is the point value or a "
        "one-sided limit (e.g. V(L−), taken just before a support reaction).",
    )
rows = [
    {
        "Quantity": label,
        "Location x (m)": loc,
        "Value": val,
        "Basis": basis,
    }
    for label, (loc, val, basis) in result.extreme_details().items()
]
st.table(pd.DataFrame(rows))
st.caption(
    "At a support the reported value may be a one-sided limit: e.g. V(L−) is "
    "the shear immediately before the support reaction is included — the "
    "point value at x = L itself is zero once the reaction is applied."
)

st.divider()
with st.expander("📘 How to read the diagrams — conventions and theory notes"):
    col_signs, col_theory = st.columns(2)
    with col_signs:
        st.markdown(
            "**Sign conventions**\n\n"
            "| Quantity | Positive means |\n"
            "|---|---|\n"
            "| Applied loads | Downward |\n"
            "| Applied moments | Clockwise |\n"
            "| Shear force V | Left segment pushed up |\n"
            "| Bending moment M | Sagging (tension on bottom fibre) |\n"
            "| Support moments | Counter-clockwise |"
        )
        st.markdown(
            "Bending moment is plotted sagging-positive; tick *Invert BMD axis* "
            "in the sidebar for the civil-engineering style (positive downward)."
        )
    with col_theory:
        st.markdown(
            "**How V and M are related**\n\n"
            "- dV/dx = −w(x): the SFD slope equals minus the load intensity.\n"
            "- dM/dx = V(x): the BMD slope equals the shear force.\n"
            "- M is extremal where V = 0 (or at a jump).\n"
            "- Point load → step in V, kink in M.\n"
            "- Applied moment → step in M.\n"
            "- UDL → linear V, parabolic M; linearly varying load → "
            "quadratic V, cubic M."
        )
    st.markdown(
        "**Model** — reactions come from ΣF = 0 and ΣM = 0; V(x) and M(x) are "
        "evaluated analytically (not numerically integrated), with the grid "
        "refined around every load discontinuity. Four boundary conditions are "
        "supported: simply supported, cantilever fixed at either end, and "
        "overhanging beams with two simple supports (0 ≤ a < b ≤ L, not both "
        "at the beam ends). Loads on overhangs are fully included; a support "
        "reaction may be negative (downward) when the loading demands it."
    )

st.caption(
    "BeamViz is an educational tool for statically determinate beams — it is "
    "not suitable for certified structural design or real-world safety approval."
)
