"""
Performance Visualization Plots.

Generates publication-quality plots for:
- Performance envelopes (L/D, sink rate vs velocity)
- Drag polar (CL vs CD)
- Optimization results (Pareto fronts)
- Energy balance timelines
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def check_matplotlib():
    """Ensure matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


def plot_performance_envelope(
    envelope: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Glider Performance Envelope",
    figsize: Tuple[int, int] = (12, 8),
) -> Optional[Figure]:
    """
    Plot glider performance envelope.

    Shows L/D ratio and sink rate vs velocity.

    Args:
        envelope: Dictionary with velocity, glide_ratio, sink_rate arrays
        save_path: Path to save figure (optional)
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    check_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    velocity = envelope["velocity"]
    glide_ratio = envelope["glide_ratio"]
    sink_rate = envelope["sink_rate"]
    cl = envelope.get("cl", np.zeros_like(velocity))
    cd = envelope.get("cd", np.zeros_like(velocity))

    # L/D vs Velocity
    ax1 = axes[0, 0]
    ax1.plot(velocity, glide_ratio, 'b-', linewidth=2, label='L/D')
    idx_max = np.argmax(glide_ratio)
    ax1.axvline(velocity[idx_max], color='r', linestyle='--', alpha=0.7, label=f'Best: {velocity[idx_max]:.1f} m/s')
    ax1.scatter([velocity[idx_max]], [glide_ratio[idx_max]], color='r', s=100, zorder=5)
    ax1.set_xlabel('Velocity (m/s)')
    ax1.set_ylabel('Glide Ratio (L/D)')
    ax1.set_title(f'Glide Ratio vs Velocity (Max L/D = {glide_ratio[idx_max]:.1f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Sink Rate vs Velocity
    ax2 = axes[0, 1]
    valid = sink_rate > 0
    ax2.plot(velocity[valid], sink_rate[valid], 'g-', linewidth=2, label='Sink Rate')
    if valid.any():
        idx_min = np.argmin(sink_rate[valid])
        valid_v = velocity[valid]
        valid_s = sink_rate[valid]
        ax2.axvline(valid_v[idx_min], color='r', linestyle='--', alpha=0.7, label=f'Min sink: {valid_v[idx_min]:.1f} m/s')
        ax2.scatter([valid_v[idx_min]], [valid_s[idx_min]], color='r', s=100, zorder=5)
    ax2.set_xlabel('Velocity (m/s)')
    ax2.set_ylabel('Sink Rate (m/s)')
    ax2.set_title('Sink Rate vs Velocity')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.invert_yaxis()  # Lower sink rate is better

    # Drag Polar (CL vs CD)
    ax3 = axes[1, 0]
    if cl.any() and cd.any():
        ax3.plot(cd * 1000, cl, 'purple', linewidth=2)
        ax3.set_xlabel('Drag Coefficient CD × 1000')
        ax3.set_ylabel('Lift Coefficient CL')
        ax3.set_title('Drag Polar')
        ax3.grid(True, alpha=0.3)

        # Add L/D lines
        for ld in [20, 30, 40, 50]:
            cd_line = np.linspace(0.001, max(cd), 50)
            cl_line = ld * cd_line
            ax3.plot(cd_line * 1000, cl_line, '--', alpha=0.3, label=f'L/D={ld}')

    # Speed polar (sink rate vs velocity, different format)
    ax4 = axes[1, 1]
    if valid.any():
        # Convert to typical glider polar format
        ax4.plot(velocity[valid], -sink_rate[valid], 'b-', linewidth=2)
        ax4.set_xlabel('Velocity (m/s)')
        ax4.set_ylabel('Climb/Sink Rate (m/s)')
        ax4.set_title('Speed Polar')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0, color='k', linestyle='-', alpha=0.3)

        # Add MacCready speeds
        for mc in [0.5, 1.0, 1.5]:
            ax4.axhline(-mc, color='orange', linestyle='--', alpha=0.5)
            ax4.text(velocity[-1], -mc, f' MC={mc}', va='center', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_drag_breakdown(
    drag_components: Dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Drag Breakdown",
) -> Optional[Figure]:
    """
    Plot pie chart of drag components.

    Args:
        drag_components: Dictionary with drag component values
        save_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    check_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 8))

    labels = list(drag_components.keys())
    sizes = list(drag_components.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.02] * len(labels)
    )

    ax.set_title(title, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_pareto_front(
    solutions: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Pareto Front - L/D vs Solar Area",
) -> Optional[Figure]:
    """
    Plot Pareto front from optimization results.

    Args:
        solutions: List of solution dictionaries with objectives
        save_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    check_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 8))

    glide_ratios = [s.get("glide_ratio", 0) for s in solutions]
    solar_areas = [s.get("solar_area", 0) for s in solutions]
    feasible = [s.get("is_feasible", True) for s in solutions]

    # Plot feasible solutions
    feas_ld = [ld for ld, f in zip(glide_ratios, feasible) if f]
    feas_sa = [sa for sa, f in zip(solar_areas, feasible) if f]
    infeas_ld = [ld for ld, f in zip(glide_ratios, feasible) if not f]
    infeas_sa = [sa for sa, f in zip(solar_areas, feasible) if not f]

    if feas_ld:
        ax.scatter(feas_ld, feas_sa, c='blue', s=50, alpha=0.7, label='Feasible')
    if infeas_ld:
        ax.scatter(infeas_ld, infeas_sa, c='red', s=30, alpha=0.3, label='Infeasible')

    # Highlight best solutions
    if feas_ld:
        idx_best_ld = np.argmax(feas_ld)
        idx_best_sa = np.argmax(feas_sa)
        ax.scatter([feas_ld[idx_best_ld]], [feas_sa[idx_best_ld]],
                   c='green', s=200, marker='*', label=f'Best L/D: {feas_ld[idx_best_ld]:.1f}')
        ax.scatter([feas_ld[idx_best_sa]], [feas_sa[idx_best_sa]],
                   c='orange', s=200, marker='*', label=f'Best Solar: {feas_sa[idx_best_sa]:.3f}m²')

    ax.set_xlabel('Glide Ratio (L/D)', fontsize=12)
    ax.set_ylabel('Solar Panel Area (m²)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_energy_balance(
    timeline: List,  # List of EnergyState
    save_path: Optional[str] = None,
    title: str = "Energy Balance Over Time",
) -> Optional[Figure]:
    """
    Plot energy balance timeline.

    Args:
        timeline: List of EnergyState objects
        save_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    check_matplotlib()

    times = [s.time for s in timeline]
    solar_power = [s.solar_power for s in timeline]
    consumption = [s.consumption for s in timeline]
    net_power = [s.net_power for s in timeline]
    battery_soc = [s.battery_soc * 100 for s in timeline]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Power plot
    ax1 = axes[0]
    ax1.fill_between(times, solar_power, alpha=0.3, color='orange', label='Solar')
    ax1.plot(times, solar_power, 'orange', linewidth=2)
    ax1.axhline(consumption[0], color='red', linestyle='--', label='Consumption')
    ax1.set_ylabel('Power (W)')
    ax1.set_title('Power Generation vs Consumption')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Net power plot
    ax2 = axes[1]
    colors = ['green' if n > 0 else 'red' for n in net_power]
    ax2.fill_between(times, net_power, alpha=0.3, color='blue')
    ax2.plot(times, net_power, 'b-', linewidth=2)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Net Power (W)')
    ax2.set_title('Net Power (Positive = Charging)')
    ax2.grid(True, alpha=0.3)

    # Battery SOC plot
    ax3 = axes[2]
    ax3.fill_between(times, battery_soc, alpha=0.3, color='green')
    ax3.plot(times, battery_soc, 'g-', linewidth=2)
    ax3.axhline(20, color='red', linestyle='--', alpha=0.7, label='Min SOC')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Battery SOC (%)')
    ax3.set_title('Battery State of Charge')
    ax3.set_ylim(0, 105)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_wing_planform(
    wing,  # WingGeometry object
    save_path: Optional[str] = None,
    show_solar: bool = True,
    title: str = "Wing Planform",
) -> Optional[Figure]:
    """
    Plot wing planform view.

    Args:
        wing: WingGeometry object
        save_path: Path to save figure
        show_solar: Show solar panel area
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    check_matplotlib()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Generate wing outline
    num_points = 50
    y = np.linspace(0, wing.semispan, num_points)
    chords = np.array([wing.chord_at_span(yi) for yi in y])
    le_x = np.array([wing.leading_edge_x(yi) for yi in y])
    te_x = le_x + chords

    # Plot both wings
    for sign in [-1, 1]:
        ax.plot(sign * y, le_x, 'b-', linewidth=2)
        ax.plot(sign * y, te_x, 'b-', linewidth=2)

    # Connect tips
    ax.plot([-wing.semispan, -wing.semispan], [le_x[-1], te_x[-1]], 'b-', linewidth=2)
    ax.plot([wing.semispan, wing.semispan], [le_x[-1], te_x[-1]], 'b-', linewidth=2)

    # Fill wing
    y_full = np.concatenate([y, y[::-1]])
    x_full = np.concatenate([le_x, te_x[::-1]])
    ax.fill(y_full, x_full, alpha=0.2, color='blue')
    ax.fill(-y_full, x_full, alpha=0.2, color='blue')

    # Show solar panel area
    if show_solar:
        solar_span_start = 0.05 * wing.semispan
        solar_span_end = 0.90 * wing.semispan
        y_solar = np.linspace(solar_span_start, solar_span_end, 30)

        for sign in [-1, 1]:
            for yi in y_solar:
                chord = wing.chord_at_span(yi)
                le = wing.leading_edge_x(yi)
                solar_le = le + 0.10 * chord
                solar_te = le + 0.70 * chord
                ax.plot([sign * yi, sign * yi], [solar_le, solar_te],
                       color='orange', linewidth=3, alpha=0.5)

    # Control surfaces
    for sign in [-1, 1]:
        # Ailerons
        ail_start = wing.aileron_span_start * wing.semispan
        ail_end = wing.aileron_span_end * wing.semispan
        for yi in [ail_start, ail_end]:
            chord = wing.chord_at_span(yi)
            le = wing.leading_edge_x(yi)
            ail_le = le + (1 - wing.aileron_chord_fraction) * chord
            ail_te = le + chord
            ax.plot([sign * yi, sign * yi], [ail_le, ail_te], 'r-', linewidth=3)

    ax.set_xlabel('Spanwise Position (m)')
    ax.set_ylabel('Chordwise Position (m)')
    ax.set_title(f'{title}\nSpan: {wing.wingspan:.2f}m, AR: {wing.aspect_ratio:.1f}, Area: {wing.wing_area:.3f}m²')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add legend
    patches = [
        mpatches.Patch(color='blue', alpha=0.2, label='Wing'),
        mpatches.Patch(color='orange', alpha=0.5, label='Solar Panels'),
        mpatches.Patch(color='red', label='Ailerons'),
    ]
    ax.legend(handles=patches)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_optimization_history(
    history: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Optimization Progress",
) -> Optional[Figure]:
    """
    Plot optimization convergence history.

    Args:
        history: List of generation statistics
        save_path: Path to save figure
        title: Plot title

    Returns:
        matplotlib Figure object
    """
    check_matplotlib()

    generations = [h["generation"] for h in history]
    pareto_sizes = [h.get("pareto_size", 0) for h in history]
    feasible_counts = [h.get("feasible_count", 0) for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1 = axes[0]
    ax1.plot(generations, pareto_sizes, 'b-o', linewidth=2, markersize=4)
    ax1.set_ylabel('Pareto Front Size')
    ax1.set_title('Pareto Front Evolution')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(generations, feasible_counts, 'g-o', linewidth=2, markersize=4)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Feasible Solutions')
    ax2.set_title('Feasible Solution Count')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
