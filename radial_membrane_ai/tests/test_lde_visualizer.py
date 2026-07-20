"""
Unit tests for the L.D.E. Visualizer and Workload.
"""

import matplotlib.pyplot as plt
from radial_membrane_ai.lde.visualizer import LDEVisualizer
from radial_membrane_ai.lde.workload import LDEWorkload
from radial_membrane_ai.lde.pipeline import lde_encode
from radial_membrane_ai.visualization.visualizer import MeshVisualizer


def test_lde_visualizer_rendering():
    raw_text = (
        "Read from left to right, the U.S. flag becomes a story — "
        "beginnings, grounding, and the horizon ahead."
    )
    state = lde_encode(raw_text)
    vis = LDEVisualizer()

    # Test individual render methods
    fig_strings = vis.render_strings(state)
    assert isinstance(fig_strings, plt.Figure)
    plt.close(fig_strings)

    fig_channels = vis.render_channels(state)
    assert isinstance(fig_channels, plt.Figure)
    plt.close(fig_channels)

    fig_boundary = vis.render_boundary(state)
    assert isinstance(fig_boundary, plt.Figure)
    plt.close(fig_boundary)

    fig_depth = vis.render_depth_distribution(state)
    assert isinstance(fig_depth, plt.Figure)
    plt.close(fig_depth)

    # Test full dashboard render method
    fig_dash = vis.render_lde_dashboard(state)
    assert isinstance(fig_dash, plt.Figure)
    plt.close(fig_dash)


def test_lde_visualizer_empty_rendering():
    state = lde_encode("!!!")  # Empty alphabetic text
    vis = LDEVisualizer()

    # Test full dashboard render method with empty state to cover "else" branch
    fig_dash = vis.render_lde_dashboard(state)
    assert isinstance(fig_dash, plt.Figure)
    plt.close(fig_dash)


def test_lde_workload_and_trace_rendering():
    wl = LDEWorkload()
    trace = wl.run("Hello JULES, letters form corridors of text!")

    # Verify trace structure
    assert len(trace.frames) == 1
    frame = trace.frames[0]
    assert hasattr(frame, "lde_state")
    assert frame.metrics.curvature >= 0.0

    # Test integrating rendering via standard MeshVisualizer
    vis = MeshVisualizer()
    report = vis.render_workload_trace(trace)

    assert report.rendered is not None
    assert report.summary.max_curvature >= 0.0
    assert report.summary.max_tension >= 0.0
    assert report.summary.coherence_stability >= 0.0
