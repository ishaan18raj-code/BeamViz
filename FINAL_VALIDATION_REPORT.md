# BeamViz — Final Engineering Validation Report

**Project:** BeamViz (interactive shear-force / bending-moment diagram tool)
**Version:** `beamviz 0.1.0` (`beamviz/__init__.py:5`)
**Report scope:** Documentation and verification only. No application logic, UI, tests, or dependencies were modified.
**Status:** Complete — all automated checks pass; analytical derivations independently confirmed against solver output.

---

## 1. Inspection summary

The project root `C:\Users\ishaan raj\BeamViz` contains:

| File | Role |
|---|---|
| `app.py` | Streamlit UI: beam setup, load table (point/UDL/moment/varying), reactions, SFD/BMD rendering, extreme values, theory notes |
| `beamviz/beam.py` | Analysis engine: reactions, `V(x)`, `M(x)`, analytic extrema, one-sided limits |
| `beamviz/plots.py` | Matplotlib schematic + SFD/BMD rendering (Agg backend) |
| `beamviz/__init__.py` | Public API: `analyze`, `compute_reactions`, `section_forces` |
| `tests/test_beam.py` | Engine unit tests: simply supported, cantilever (left/right), validation, boundary extremes |
| `tests/test_overhang.py` | Hand-calc validation for overhanging beams |
| `tests/test_varying.py` | Hand-calc validation for linearly varying loads |
| `tests/test_plots.py` | Plotting smoke tests |
| `conftest.py` | Makes project root importable for pytest |
| `requirements.txt` | `streamlit>=1.36`, `numpy>=1.26`, `matplotlib>=3.8`, `pytest>=8.0` |
| `README.md` | User-facing feature, sign-convention, and usage documentation |

No source file was modified for this report.

---

## 2. Mathematical model

### 2.1 Sign conventions (as documented in `beamviz/beam.py:7-15` and `app.py:136-143`)

| Quantity | Positive means | Units |
|---|---|---|
| `x` axis | 0 at left beam end, positive to the right | m |
| Applied loads | Downward | kN, kN/m |
| Applied moments | Clockwise | kN·m |
| Shear `V(x)` | Left segment tends to move up | kN |
| Bending moment `M(x)` | Sagging (tension on bottom fibre) | kN·m |
| Support moments | Reported counter-clockwise positive | kN·m |

Consequence: a downward point load causes a **downward step** in `V`; a clockwise applied moment causes an **upward jump** of `+M` in `M`; a counter-clockwise support moment causes a jump of `-M`.

### 2.2 Supported beam types (`beamviz/beam.py:39-46`)

| `SupportType` | Meaning | Reaction positions |
|---|---|---|
| `SIMPLY_SUPPORTED` | pin at left end + roller at right end | `x = 0`, `x = L` |
| `CANTILEVER_LEFT` | fixed at left end | `x = 0` (force + moment) |
| `CANTILEVER_RIGHT` | fixed at right end | `x = L` (force + moment) |
| `OVERHANGING` | two simple vertical supports | `x = a`, `x = b`, `0 ≤ a < b ≤ L`, not both at beam ends |

Overhanging beams require `support_positions=(a, b)`; loads on overhangs are fully included and a reaction may be negative (downward).

### 2.3 Support reactions (`compute_reactions`, `beamviz/beam.py:274-394`)

All reactions from ΣF = 0 and ΣM = 0, clockwise applied moments entered as `+moment_total`. `total_down` = sum of all downward load resultants.

**Simply supported:**
```
R_right = (ΣP·p + ΣW·c + M0_varying + ΣM_cw) / L
R_left  = total_down − R_right
```
Moments taken about the left support (CCW positive).

**Cantilever fixed at left (`x = 0`):**
```
R_left  = total_down
M_left  = ΣP·p + ΣW·c + M0_varying + ΣM_cw   (CCW positive)
```

**Cantilever fixed at right (`x = L`):**
```
R_right = total_down
M_right = −[ΣP·(L−p) + ΣW·(L−c) + M0_right_varying] + ΣM_cw
```

**Overhanging (supports at `a`, `b`):**
```
R_right = [ΣP·(p−a) + ΣW·(c−a) + (M0_varying − a·W_varying) + ΣM_cw] / (b − a)
R_left  = total_down − R_right
```
A downward load on the left overhang (`p < a`) reduces `R_right` and can make it negative, which is statically admissible and is not clamped.

### 2.4 Shear-force and bending-moment equations (`section_forces` → `_internal_forces`, `beamviz/beam.py:423-513`)

Internal forces at section `x` are built by summing contributions to the left of the section. Right-continuous evaluation includes actions located exactly at `x`; `strict_left=True` returns the left-hand limit `V(x−)`, `M(x−)` (reactions/point loads/moments exactly at `x` excluded).

**Reaction force `R` upward at `p`:**
```
V += R        (x ≥ p)
M += R·(x − p)   (x ≥ p)
```

**Counter-clockwise support moment `M_s` at `p`:**
```
M −= M_s   (x ≥ p)       # jump of −M at p
```

**Point load `P` downward at `p`:**
```
V −= P        (x ≥ p)
M −= P·(x − p)   (x ≥ p)
```

**UDL intensity `w` over `[s, e]`:**
```
Q   = w·(x − s)                 s ≤ x ≤ e        ;   W = w·(e − s)             x ≥ e
M −= ½·w·(x − s)²               s ≤ x ≤ e        ;   M −= W·(x − (s+e)/2)     x ≥ e
V −= Q                          s ≤ x ≤ e        ;   V −= W                    x ≥ e
```

**Linearly varying load `w(x) = w1 + k(x−s)`, `k = (w2 − w1)/(e−s)` over `[s, e]`:** with `u = clamp(x − s, 0, e − s)`:
```
Q(u)  = w1·u + k·u²/2
J_s(u) = w1·u²/2 + k·u³/3
M_sec(u) = u·Q(u) − J_s(u) = w1·u²/2 + k·u³/6
V −= Q(u)                       s ≤ x ≤ e        ;   V −= W                  x ≥ e
M −= M_sec(u)                   s ≤ x ≤ e        ;   M −= x·W − M0           x ≥ e
```
where `W = (w1 + w2)(e − s)/2` and `M0 = ∫w·x dx` about `x = 0` (exact closed form, `beamviz/beam.py:88-107`; valid even when `W = 0`, i.e. a zero-resultant pure couple). `w1 == w2` reduces exactly to a UDL.

**Discontinuities:**
- Point load → step in `V` of `−P`, kink (continuous value) in `M`.
- Applied moment → step (jump) in `M` of `+M` (clockwise positive input), no step in `V`.
- UDL / varying-load edges → `V` and `M` continuous (no point force there); slope changes per `dV/dx = −w(x)`, `dM/dx = V(x)`.

### 2.5 Exact extrema detection and one-sided limits (`BeamResult.extremes`, `beamviz/beam.py:144-177`; `_extreme_candidates`, `beamviz/beam.py:621-740`)

Extrema are **not** taken from the plotted sample grid. Candidates are exact one-sided values at every station: beam boundaries `0`, `L`, every load position / load edge, every support reaction position, plus interior stationary points where `V = 0`:
- On a UDL-only interval, `V` is linear → `M` stationary where `V = 0`, solved analytically (`x* = a + v_a / Σw`).
- On a varying-load interval, `V` is quadratic → roots of `V = 0` via the quadratic formula; also any point where load intensity `w(x) = 0` (a `V` stationary point).
- Basis labels: `"point value"`, `"left-hand limit"`, `"right-hand limit"` (`beamviz/beam.py:604-606`).
- Boundary semantics: at `x = L` the point value includes the support reaction (`V = 0`), whereas `V(L−)` is the limit before the reaction — reported separately as a left-hand limit. Same at each interior support of an overhanging beam.

### 2.6 Grid refinement (`analyze`, `beamviz/beam.py:538-591`)

`analyze` samples `[0, L]` on a uniform grid (default 1201 points) then inserts `d − ε`, `d`, `d + ε` around every discontinuity `d` (point loads, moments, UDL/varying edges, reactions), with `ε = max(L·1e−6, 1e−12)`. Boundary discontinuities are refined just inside the domain so jumps at `x = 0` and `x = L` are drawn vertically. This is purely visual; the analytic `V`/`M` evaluation is exact independent of the grid.

---

## 3. Reproducible demonstrations

All three cases were derived independently by hand (static equilibrium + direct integration) and then checked against the solver (`analyze` / `section_forces` / `compute_reactions`). The solver output matches the hand values exactly within floating-point roundoff. Corresponding assertions already exist in `tests/test_overhang.py` (Case A overhang), `tests/test_varying.py` (Case B), and `tests/test_beam.py` (Case A simply supported).

### Demonstration A — Simply supported beam, centre point load

**Geometry:** `L = 6 m`, simply supported (pin `x = 0`, roller `x = 6`).

**Loads:** one `PointLoad(position = 3.0, magnitude = 10.0)` kN downward at midspan.

**Hand-calculated reactions:**
- ΣFy: `R_A + R_B = 10`
- ΣM about `x = 0` (CCW +): `R_B·6 − 10·3 = 0` → `R_B = 5 kN`, `R_A = 5 kN`

**Expected SFD:**
- `0 ≤ x < 3`: `V = +5 kN`; step at `x = 3` by `−10`; `3 < x ≤ 6`: `V = −5 kN`
- `V(0) = +5`; `V(3−) = +5`, `V(3+) = −5`; `V(6−) = −5` (left-hand limit); `V(6) = 0` (point value, reaction included)

**Expected BMD:**
- `M(0) = 0`; `M(x) = 5x` on `[0, 3]` → `M(3) = +15 kN·m` (maximum)
- `M(x) = 30 − 5x` on `[3, 6]` → `M(6) = 0`
- Diagram closes at both ends; hogging `M_min = 0`.

**Numerical check (solver output):**
- `R_left = 5.0`, `R_right = 5.0` kN
- `Max moment (+) = 15.0 kN·m` at `x = 3.0` m (point value)
- `Max shear (+) = 5.0 kN` at `x = 0.0` m; `Min shear (−) = −5.0 kN` at `x = 3.0` m

**UI reproduction:**
1. `python -m streamlit run app.py`
2. Beam type: "Simply supported (pin + roller)"; Length `L = 6`
3. Load table: one row — Type "Point load", Position `3`, Value `10`
4. Confirm reactions `R_left = 5`, `R_right = 5` kN; SFD steps `+5 → −5` at midspan; BMD peaks `+15` kN·m at `x = 3`; diagrams close to zero at both ends.

---

### Demonstration B — Simply supported beam, full-span triangular load

**Geometry:** `L = 6 m`, simply supported.

**Loads:** one `VaryingLoad(start = 0, end = 6, start_intensity = 0, end_intensity = 6)` kN/m, i.e. `w(x) = x` kN/m increasing linearly from 0 at `x = 0` to 6 at `x = 6`.

**Hand-calculated reactions:**
- Resultant `W = (0 + 6)·6/2 = 18 kN`
- Centroid `x̄ = 6·(0 + 2·6)/(3·(0 + 6)) = 4 m`
- ΣM about `x = 0`: `R_B·6 − 18·4 = 0` → `R_B = 12 kN`; `R_A = 18 − 12 = 6 kN`

**Expected SFD:** `dV/dx = −w = −x` → `V(x) = 6 − x²/2`
- `V(0) = +6 kN`; `V(3) = +1.5 kN`; `V = 0` at `x = √12 = 2√3 ≈ 3.464 m`; `V(6−) = −12 kN` (left-hand limit); `V(6) = 0`

**Expected BMD:** `dM/dx = V` → `M(x) = 6x − x³/6`
- `M(0) = 0`; maximum where `V = 0`: `M(2√3) = 8√3 ≈ +13.8564 kN·m` at `x ≈ 3.464 m`; `M(6) = 36 − 36 = 0`
- Cubic BMD, sagging throughout, closes to zero at both ends.

**Numerical check (solver output):**
- `R_left = 6.0`, `R_right = 12.0` kN
- `Max moment (+) = 13.856406 kN·m` (`8√3`) at `x = 3.464102 m` (`2√3`), basis "point value"
- `Max shear (+) = 6.0 kN` at `x = 0`; `Min shear (−) = −12.0 kN` at `x = 6`, basis "left-hand limit" (`V(6−)`)

**UI reproduction:**
1. Beam type: "Simply supported (pin + roller)"; Length `L = 6`
2. Load table: one row — Type "Varying distributed load", Position `0`, Value `0` (w1), UDL end `6`, w2 `6`
3. Confirm reactions `R_left = 6`, `R_right = 12` kN; SFD quadratic from `+6` to `−12`; BMD cubic peak `≈13.86` kN·m near `x = 3.46`; diagrams close at both ends.

---

### Demonstration C — Overhanging beam, centre point load

**Geometry:** `L = 6 m`, overhanging with simple supports at `a = 1 m`, `b = 5 m`; overhangs `[0, 1]` and `[5, 6]`.

**Loads:** one `PointLoad(position = 3.0, magnitude = 10.0)` kN downward at midspan `x = 3`.

**Hand-calculated reactions** (moments about `x = a = 1`, CCW +; lever `b − a = 4`):
- `R_B·4 − 10·(3 − 1) = 0` → `R_B = 20/4 = 5 kN`
- `R_A = 10 − 5 = 5 kN`

**Expected SFD:**
- `0 ≤ x < 1` (left overhang): `V = 0`
- reaction at `x = 1`: `V` jumps to `+5`; `1 < x < 3`: `V = +5 kN`
- point load at `x = 3`: step `−10`; `3 < x < 5`: `V = −5 kN`
- reaction at `x = 5`: `V` jumps by `+5` → `0`; `5 < x ≤ 6`: `V = 0`
- `V(1−) = 0`, `V(1+) = +5`; `V(3−) = +5`, `V(3+) = −5`; `V(5−) = −5`, `V(5+) = 0`

**Expected BMD:**
- `M = 0` on both overhangs `[0, 1]` and `[5, 6]`; `M(0) = M(1) = M(5) = M(6) = 0`
- `M(x) = 5(x − 1)` on `[1, 3]` → `M(3) = +10 kN·m` (maximum)
- `M(x) = 25 − 5x` on `[3, 5]` → `M(4) = 5`, `M(5) = 0`
- Triangular BMD between supports, max `+10 kN·m` at `x = 3`; hogging `M_min = 0`.

**Numerical check (solver output):**
- `R_left = 5.0`, `R_right = 5.0` kN (at `x = 1` and `x = 5`)
- `Max moment (+) = 10.0 kN·m` at `x = 3.0` m (point value)
- `Max shear (+) = 5.0 kN` at `x = 1.0` m (right-hand limit); `Min shear (−) = −5.0 kN` at `x = 3.0` m

**UI reproduction:**
1. Beam type: "Overhanging (two simple supports)"; Length `L = 6`
2. Left support position `a = 1`; Right support position `b = 5`
3. Load table: one row — Type "Point load", Position `3`, Value `10`
4. Confirm reactions at `x = 1` and `x = 5` both `5 kN`; SFD `0 → +5 → −5 → 0`; BMD triangular peak `+10` kN·m at midspan; zero on overhangs and at both free ends.

---

## 4. Verification

### Automated tests

```
python -m pytest -v
```

**Result: 83 passed, 0 failed, in 1.20 s**

- `tests/test_beam.py`: 28 passed (simply supported, cantilever left/right, analysis result, validation, boundary extremes)
- `tests/test_overhang.py`: 18 passed (symmetric overhang, left overhang with negative reaction, combined load, geometry validation)
- `tests/test_plots.py`: 2 passed (figure smoke tests, Agg backend)
- `tests/test_varying.py`: 35 passed (triangular, constant-varying=UDL, overhang triangular, trapezoidal, cantilever varying, zero-resultant pure couple, continuity, combined varying+point load)

All tests are hand-calculation-based assertions against the engine; they independently corroborate Demonstrations A (simply supported portion), B (`TestCaseAFullSpanTriangular`), and C (`TestCaseASymmetricOverhangs`).

### Application health check

There is no separate health-check script in the repository. The app's rendering path was exercised via `tests/test_plots.py` (`plot_diagrams` with `invert_bmd = False/True`, empty loads) — both pass — and the core engine imports cleanly (`beamviz 0.1.0`, numpy-backed). `streamlit 1.66.1` and `matplotlib 3.11.2` are installed and importable. The app launches with `python -m streamlit run app.py` per `README.md:35`; no server crash or import error was observed during dependency verification. (A live browser interaction was not performed; UI reproduction steps in Section 3 are the manual check to be executed by the reviewer.)

---

## 5. Limitations

### Unsupported engineering cases
- **Indeterminate structures:** continuous beams, propped cantilevers, fixed-fixed beams — only statically determinate boundary conditions are solved.
- **Axial/torsional loads:** only transverse vertical loads and in-plane concentrated moments are supported.
- **Multiple support moments / internal hinges:** not represented; overhanging supports are simple (vertical reaction only).
- **Non-prismatic members, self-weight, temperature, support settlement:** not modelled.
- **Dynamic, impact, or moving-load influence lines:** static analysis only.

### Modelling assumptions
- Euler-Bernoulli slender beam; small deflections; linear elastic response implied for diagram interpretation (the tool itself computes statical `V`/`M` only, no stresses or deflections).
- Point loads and moments are idealized concentrations; UDL/varying loads are continuous over their spans.
- Loads may be placed exactly at supports or beam ends; such cases yield zero internal effect at that point (a point load on a support is carried directly by the reaction).
- Distributed loads must lie within `[0, L]` with `start < end`; zero-length intervals are rejected.

### Numerical limitations
- `V`/`M` are evaluated analytically (no numerical integration); extrema use closed-form roots, so results are exact up to double-precision roundoff.
- Extrema ties resolve to the first candidate in station order (`_pick`, `beamviz/beam.py:609-618`); reporting location may differ from a symmetric alternative but the value is identical.
- Grid refinement (`ε = max(L·1e−6, 1e−12)`) is visual only; extremely small `L` relative to `ε` could merge nearby features on the plot.
- No unit system conversion; all inputs and outputs use kN, kN/m, kN·m, m consistently.

### Educational vs certified software
BeamViz is explicitly documented as an educational tool (`app.py:53-56`, `README.md`): it performs statical analysis and diagram rendering for statically determinate beams. It does **not** perform code checks (stress, deflection, buckling, fatigue), does not enforce any design standard (Eurocode, AISC, ACI, etc.), produces no design calculations or stamped output, and carries no engineering certification. It is suitable for coursework, demonstration, and preliminary understanding — not for certified structural design or real-world safety approval.

---

## 6. Conclusion

- **Analytical derivations:** Sections 2 and 3 derive reactions, SFD, BMD, discontinuities, and extrema from first principles for all three demonstration cases; all match the solver output exactly.
- **Automated tests:** 83 passed, 0 failed (`python -m pytest -v`), including hand-calc assertions for Cases A, B, and C.
- **Manual UI checks:** reproduction steps are provided in Section 3; no live browser session was executed, so UI behaviour remains a prescribed manual check rather than an observed result.

No new features were added and no solver, UI, test, or dependency files were modified.
