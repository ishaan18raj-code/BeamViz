# BeamViz — Interactive Shear Force and Bending Moment Diagram Tool

BeamViz is a lightweight web app for analysing 1-D beams and visualising
their **shear force (SFD)** and **bending moment (BMD)** diagrams.

## Features

- Support types: simply supported (pin + roller), cantilever fixed at the
  left end, cantilever fixed at the right end, and overhanging beams with
  two simple supports at arbitrary positions `a` and `b`
  (`0 <= a < b <= L`, not both at the beam ends).
- Loads: point loads (kN), uniformly distributed loads (kN/m) and
  concentrated moments (kN·m), combined in any mix.
- Analytic evaluation of V(x) and M(x) with automatic grid refinement at
  every discontinuity, so jumps and kinks are rendered exactly.
- Interactive UI (Streamlit): editable load table, reaction metrics,
  annotated diagrams, extreme-value summary, and an optional
  civil-engineering BMD orientation (positive drawn downward).

## Sign conventions

| Quantity          | Positive direction                          |
| ----------------- | ------------------------------------------- |
| Applied loads     | Downwards                                   |
| Applied moments   | Clockwise                                   |
| Shear force V     | Left segment pushed upwards                 |
| Bending moment M  | Sagging (tension on the bottom fibre)       |
| Support moments   | Reported counter-clockwise positive         |

## Getting started

```bash
# from the project folder
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser (default <http://localhost:8501>).
> Note: if the Python `Scripts` folder is not on `PATH`, always use the
> `python -m streamlit ...` form instead of the bare `streamlit` command.

## Running the tests

```bash
python -m pytest
```

## Project layout

```
BeamViz/
├── app.py               # Streamlit user interface
├── beamviz/
│   ├── beam.py          # analysis engine (reactions, V(x), M(x))
│   └── plots.py         # Matplotlib schematic + SFD/BMD rendering
├── tests/
│   ├── test_beam.py     # engine unit tests
│   └── test_plots.py    # plotting smoke tests
├── conftest.py          # makes the project root importable for pytest
└── requirements.txt
```
