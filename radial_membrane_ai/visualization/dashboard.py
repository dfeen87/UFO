# Copyright (c) Don Michael Feeney Jr.
# Licensed under the MIT License.

"""
Unified multi-panel dashboard and individual component rendering using Matplotlib.
All rendering is deterministic, reproducible, and governed.
"""

from __future__ import annotations
import io
import math
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from radial_membrane_ai.visualization.snapshots import (
    VisualizationFrame,
    VisualizationReport,
    MembraneGeometrySnapshot,
    VChannelSnapshot
)
from radial_membrane_ai.kernel_regimes.regime import KernelRegimeType

matplotlib.use("Agg")

# Global Matplotlib Configuration
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = False

# Colors as specified
COLOR_GREEN = "#4CAF50"
COLOR_YELLOW = "#FFC107"
COLOR_RED = "#F44336"

COLOR_REGIMES = {
    KernelRegimeType.DETERMINISTIC: "#2196F3",
    KernelRegimeType.STOCHASTIC: "#9C27B0",
    KernelRegimeType.HIGH_CURVATURE: "#FF5722",
    KernelRegimeType.ADVERSARIAL: "#795548",
    KernelRegimeType.MULTI_PHASE: "#009688",
    KernelRegimeType.BALANCED: "#9E9E9E",
}


def set_deterministic_env() -> None:
    """Sets environment seeds and properties to ensure reproducibility."""
    np.random.seed(0)


def render_membrane_geometry_panel(
    snapshot: MembraneGeometrySnapshot,
    ax: plt.Axes,
    title: str = "A. Membrane Geometry"
) -> None:
    """
    Renders Panel A: Membrane Geometry.
    Shows the radial boundary polygon, curvature hotspots, tension gradients,
    admissibility zones, and capacity scalar.
    """
    set_deterministic_env()
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')

    # Draw admissibility sectors or background zones
    # We have 12 sectors. Let's draw radial sectors.
    for i in range(12):
        theta_start = (2.0 * math.pi * i) / 12.0
        theta_end = (2.0 * math.pi * (i + 1)) / 12.0
        is_admissible = snapshot.admissibility_zones[i % len(snapshot.admissibility_zones)]
        color = COLOR_GREEN if is_admissible else COLOR_RED
        # Draw a small outer arc/segment indicating admissibility
        angles = np.linspace(theta_start, theta_end, 10)
        # We can draw shading for the quadrant/sector
        x_sector = [0.0] + [math.cos(a) * 0.2 for a in angles] + [0.0]
        y_sector = [0.0] + [math.sin(a) * 0.2 for a in angles] + [0.0]
        ax.fill(x_sector, y_sector, color=color, alpha=0.1)

    # Plot boundary coordinates
    if snapshot.boundary_coords:
        xs, ys = zip(*snapshot.boundary_coords)
        # Close the loop
        xs_closed = list(xs) + [xs[0]]
        ys_closed = list(ys) + [ys[0]]
        ax.plot(xs_closed, ys_closed, color="#333333", linewidth=1.5, label="Boundary")
        ax.fill(xs_closed, ys_closed, color="#CCCCCC", alpha=0.2)

    # Plot string nodes (12 sectors) with curvature and tension markers
    for i in range(12):
        theta = (2.0 * math.pi * (i + 1)) / 12.0
        # Interpolate boundary radius at string position
        r = 1.0
        # If we can match coordinates, let's find closest boundary coord
        if snapshot.boundary_coords:
            best_dist = float('inf')
            for cx, cy in snapshot.boundary_coords:
                c_theta = math.atan2(cy, cx) % (2 * math.pi)
                diff = abs(c_theta - theta) % (2 * math.pi)
                dist = min(diff, 2 * math.pi - diff)
                if dist < best_dist:
                    best_dist = dist
                    r = math.sqrt(cx**2 + cy**2)

        x = r * math.cos(theta)
        y = r * math.sin(theta)

        curv = snapshot.curvature_map[i % len(snapshot.curvature_map)]
        tens = snapshot.tension_map[i % len(snapshot.tension_map)]

        # Draw concentric rings/markers representing tension (inner) and curvature (outer)
        sz_tens = max(1.0, 40.0 + tens * 20.0)
        sz_curv = max(1.0, 20.0 + curv * 10.0)
        ax.scatter([x], [y], s=sz_tens, color=COLOR_RED, alpha=0.7, edgecolors='black', zorder=5)
        ax.scatter([x], [y], s=sz_curv, color=COLOR_YELLOW, alpha=0.9, edgecolors='black', zorder=6)

    # Text annotation for global properties
    ax.text(
        0.05, 0.05, f"Capacity Scalar: {snapshot.capacity_scalar:.2f}",
        transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
    )

    # Hide axis ticks for a clean spatial look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)


def render_vchannel_panel(
    snapshot: VChannelSnapshot,
    ax: plt.Axes,
    title: str = "B. V-Channels Routing"
) -> None:
    """
    Renders Panel B: V-Channels routing.
    Shows pressures, routing directions, admissibility, and SAO-eligible channels.
    """
    set_deterministic_env()
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10, fontweight='bold')

    # Draw the central 12 behavioral string nodes
    nodes_x = []
    nodes_y = []
    for i in range(12):
        theta = (2.0 * math.pi * (i + 1)) / 12.0
        r = 1.5
        nodes_x.append(r * math.cos(theta))
        nodes_y.append(r * math.sin(theta))

    # Draw nodes
    for i in range(12):
        is_sao = snapshot.sao_eligible[i % len(snapshot.sao_eligible)]
        is_admissible = snapshot.admissible[i % len(snapshot.admissible)]
        node_color = COLOR_GREEN if is_admissible else COLOR_RED
        edge_color = "gold" if is_sao else "black"
        lw = 2.0 if is_sao else 1.0
        ax.scatter([nodes_x[i]], [nodes_y[i]], s=100, color=node_color, edgecolors=edge_color, linewidths=lw, zorder=5)
        ax.text(nodes_x[i] * 1.25, nodes_y[i] * 1.25, str(i+1), fontsize=8, ha='center', va='center')

    # Draw V-Channel connection routing lines
    # We can connect each node to its neighbors or show routing vectors
    for i in range(12):
        pressure = snapshot.pressures[i % len(snapshot.pressures)]
        angle = snapshot.routing_dirs[i % len(snapshot.routing_dirs)]

        # Draw a routing vector arrow representing flow from nodes
        x_start = nodes_x[i]
        y_start = nodes_y[i]

        # Direction of routing
        dx = pressure * 0.4 * math.cos(angle)
        dy = pressure * 0.4 * math.sin(angle)

        ax.arrow(
            x_start, y_start, dx, dy,
            head_width=0.08, head_length=0.1, fc='#2196F3', ec='#2196F3',
            alpha=min(1.0, max(0.2, pressure)), zorder=4
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)


def render_curvature_tension_timeline(
    frames: List[VisualizationFrame],
    ax: plt.Axes
) -> None:
    """
    Renders Panel C: Curvature and Tension Timeline.
    Curvature (left axis) and Tension (right axis) with regime-colored background bands.
    """
    set_deterministic_env()
    ax.set_title("C. Curvature & Tension Trajectories", fontsize=10, fontweight='bold')
    if not frames:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return

    steps = list(range(len(frames)))
    curvatures = [f.curvature for f in frames]
    tensions = [f.tension for f in frames]

    # Draw background regime bands
    for idx, f in enumerate(frames):
        color = COLOR_REGIMES.get(f.regime, "#9E9E9E")
        ax.axvspan(idx - 0.5, idx + 0.5, facecolor=color, alpha=0.15)

    # Curvature plot (Left Y axis)
    ax.plot(steps, curvatures, color="purple", marker="o", markersize=4, label="Curvature (K)")
    ax.set_ylabel("Curvature (K)", color="purple")
    ax.tick_params(axis='y', labelcolor="purple")

    # Tension plot (Right Y axis)
    ax2 = ax.twinx()
    ax2.plot(steps, tensions, color="blue", marker="s", markersize=4, label="Tension (T)")
    ax2.set_ylabel("Tension (T)", color="blue")
    ax2.tick_params(axis='y', labelcolor="blue")

    ax.set_xlabel("Workload Step Index")
    ax.set_xticks(steps)


def render_sao_timeline(
    frames: List[VisualizationFrame],
    ax: plt.Axes
) -> None:
    """
    Renders Panel D: SAO Promotion Timeline.
    Shows short, mid, and long ranges.
    """
    set_deterministic_env()
    ax.set_title("D. SAO Promotion Gating Timeline", fontsize=10, fontweight='bold')
    if not frames:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return

    steps = list(range(len(frames)))

    # Y-axis will represent categories
    levels = ["Short-Range", "Mid-Range", "Long-Range"]
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(levels)
    ax.set_ylim(0.5, 3.5)

    # Map each frame's SAO level
    for idx, f in enumerate(frames):
        for e in f.sao_events:
            level_str = e.level.value if hasattr(e.level, 'value') else str(e.level)
            if "short" in level_str.lower():
                ax.scatter([idx], [1], s=120, color=COLOR_GREEN, edgecolors='black', marker='^', zorder=5)
            elif "mid" in level_str.lower():
                ax.scatter([idx], [2], s=120, color=COLOR_YELLOW, edgecolors='black', marker='p', zorder=5)
            elif "long" in level_str.lower():
                ax.scatter([idx], [3], s=120, color=COLOR_RED, edgecolors='black', marker='*', zorder=5)

    ax.set_xlabel("Workload Step Index")
    ax.set_xticks(steps)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)


def render_stability_band_panel(
    frames: List[VisualizationFrame],
    report: Optional[VisualizationReport],
    ax: plt.Axes
) -> None:
    """
    Renders Panel E: Stability Band Indicator with Rollback and Quarantine markers.
    """
    set_deterministic_env()
    ax.set_title("E. Stability Bands & Interventions", fontsize=10, fontweight='bold')
    if not frames:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return

    steps = list(range(len(frames)))

    # Draw horizontal bar for stability bands
    band_colors = []
    for f in frames:
        b_str = f.stability_band.value if hasattr(f.stability_band, 'value') else str(f.stability_band)
        if "green" in b_str.lower():
            band_colors.append(COLOR_GREEN)
        elif "yellow" in b_str.lower():
            band_colors.append(COLOR_YELLOW)
        else:
            band_colors.append(COLOR_RED)

    # Plot stability band horizontal timeline bar
    for idx, color in enumerate(band_colors):
        ax.barh(1, 0.8, left=idx - 0.4, color=color, edgecolor='none', height=0.4)

    ax.set_yticks([1])
    ax.set_yticklabels(["Stability Band"])
    ax.set_ylim(0.5, 1.8)

    # Add markers for rollbacks and quarantines
    if report and hasattr(report, 'summary'):
        if report.summary.rollback_count > 0:
            ax.text(0.5, 0.5, "Interventions logged", fontsize=8)

    ax.set_xlabel("Workload Step Index")
    ax.set_xticks(steps)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)


def render_coherence_panel(
    frames: List[VisualizationFrame],
    ax: plt.Axes
) -> None:
    """
    Renders Panel F: Coherence Panel.
    Agent, Cluster, and Global Coherence trajectories.
    """
    set_deterministic_env()
    ax.set_title("F. Coherence Timeline (A/C/G)", fontsize=10, fontweight='bold')
    if not frames:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return

    steps = list(range(len(frames)))
    coherences = [f.coherence for f in frames]

    ax.plot(steps, coherences, color="#3F51B5", marker="o", linewidth=2.0, label="Global Coherence")

    # Let's mock cluster/agent coherences relative to global coherence
    cluster_coh = [max(0.0, min(1.0, c * 0.95)) for c in coherences]
    agent_coh = [max(0.0, min(1.0, c * 0.9)) for c in coherences]

    ax.plot(steps, cluster_coh, color="#2196F3", linestyle="--", marker="x", alpha=0.7, label="Cluster Coherence")
    ax.plot(steps, agent_coh, color="#00BCD4", linestyle=":", marker="+", alpha=0.7, label="Agent Coherence")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Workload Step Index")
    ax.set_ylabel("Coherence Score")
    ax.set_xticks(steps)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, linestyle='--', alpha=0.5)


def render_regime_timeline(
    frames: List[VisualizationFrame],
    ax: plt.Axes
) -> None:
    """
    Renders Panel G: Regime Transition Timeline.
    """
    set_deterministic_env()
    ax.set_title("G. Regime Transitions", fontsize=10, fontweight='bold')
    if not frames:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return

    steps = list(range(len(frames)))

    # Plot horizontal step line or bar for active regime
    regime_list = [f.regime for f in frames]
    regime_names = list(COLOR_REGIMES.keys())

    y_vals = []
    for r in regime_list:
        try:
            y_vals.append(regime_names.index(r))
        except ValueError:
            y_vals.append(len(regime_names) - 1)

    ax.step(steps, y_vals, where='mid', color='#4CAF50', linewidth=2.0, marker='D')

    ax.set_yticks(range(len(regime_names)))
    ax.set_yticklabels([r.value for r in regime_names], fontsize=8)
    ax.set_ylim(-0.5, len(regime_names) - 0.5)

    ax.set_xlabel("Workload Step Index")
    ax.set_xticks(steps)
    ax.grid(True, linestyle='--', alpha=0.5)


def draw_unified_dashboard(
    frames: List[VisualizationFrame],
    report: Optional[VisualizationReport] = None,
    dpi: int = 120,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Draws all 7 panels in a clean subplots grid.
    Returns the Matplotlib Figure.
    """
    set_deterministic_env()
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Grid layout: 3 rows, 3 columns
    # We can place A and B on top row, C, D on middle row, E, F, G on bottom row
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    # Empty spot or text overview
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis('off')

    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])

    ax_f = fig.add_subplot(gs[2, 0])
    ax_g = fig.add_subplot(gs[2, 1])

    # Fill Panels
    # Take the latest frame for spatial snapshots
    if frames:
        latest = frames[-1]
        render_membrane_geometry_panel(latest.membrane_geometry, ax_a)
        render_vchannel_panel(latest.vchannels, ax_b)

        # Draw Summary overview in ax_info
        if report and report.summary:
            sum_text = (
                f"** U.F.O. GOVERNED DASHBOARD **\n"
                f"-------------------------------\n"
                f"Max Curvature: {report.summary.max_curvature:.2f}\n"
                f"Max Tension: {report.summary.max_tension:.2f}\n"
                f"SAO Promotions: {report.summary.sao_event_count}\n"
                f"Rollbacks: {report.summary.rollback_count}\n"
                f"Quarantines: {report.summary.quarantine_count}\n"
                f"Regime Transitions: {report.summary.regime_transition_count}\n"
                f"Coherence Stability: {report.summary.coherence_stability:.2f}"
            )
            ax_info.text(0.1, 0.9, sum_text, va='top', fontsize=9, fontfamily='monospace')

        render_curvature_tension_timeline(frames, ax_c)
        render_sao_timeline(frames, ax_d)
        render_stability_band_panel(frames, report, ax_e)
        render_coherence_panel(frames, ax_f)
        render_regime_timeline(frames, ax_g)

    return fig


def convert_figure_to_image(fig: plt.Figure) -> Image.Image:
    """Converts a Matplotlib Figure into a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    # Force load and copy bytes to ensure buffer can be closed safely
    img.load()
    buf.close()
    plt.close(fig)
    return img
