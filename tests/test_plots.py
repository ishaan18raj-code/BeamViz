"""Smoke test for the plotting module (Agg backend, no display needed)."""

import matplotlib.pyplot as plt

from beamviz.beam import (
    AppliedMoment,
    DistributedLoad,
    PointLoad,
    SupportType,
    analyze,
)
from beamviz.plots import plot_diagrams


def test_plot_diagrams_returns_figure():
    result = analyze(
        6.0,
        SupportType.SIMPLY_SUPPORTED,
        [
            PointLoad(3.0, 10.0),
            DistributedLoad(4.0, 6.0, 2.0),
            AppliedMoment(1.0, 4.0),
        ],
    )
    for invert in (False, True):
        fig = plot_diagrams(result, invert_bmd=invert)
        assert fig is not None
        assert len(fig.axes) == 3
        plt.close(fig)


def test_plot_diagrams_empty_loads():
    result = analyze(4.0, SupportType.CANTILEVER_LEFT, [])
    fig = plot_diagrams(result)
    assert fig is not None
    plt.close(fig)
