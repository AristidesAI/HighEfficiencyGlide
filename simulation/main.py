#!/usr/bin/env python3
"""
HighEfficiencyGlide Simulation Entry Point.

This module provides the main interface for running glider simulations,
optimizations, and performance analyses.

Usage:
    python -m simulation.main [command] [options]

Commands:
    analyze     - Run aerodynamic analysis
    optimize    - Run wing optimization
    envelope    - Generate performance envelope
    summary     - Print glider summary

Examples:
    python -m simulation.main analyze --wingspan 4.0 --ar 20
    python -m simulation.main optimize --generations 100
    python -m simulation.main envelope --altitude 1000 --output plots/
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import yaml
import numpy as np

from .aerodynamics.glider_model import GliderModel, MassProperties, create_default_glider
from .aerodynamics.wing_geometry import WingGeometry, create_high_ar_glider_wing
from .aerodynamics.atmosphere import get_atmosphere


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}, using defaults")
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_glider_from_config(config: dict) -> GliderModel:
    """Create glider model from configuration dictionary."""
    wing_cfg = config.get("wing", {})
    mass_cfg = config.get("mass", {})

    wing = WingGeometry(
        wingspan=wing_cfg.get("wingspan", 4.0),
        root_chord=wing_cfg.get("root_chord", 0.35),
        taper_ratio=wing_cfg.get("taper_ratio", 0.45),
        sweep_angle=wing_cfg.get("sweep_angle", 0.0),
        dihedral_angle=wing_cfg.get("dihedral_angle", 3.0),
        twist=wing_cfg.get("twist", -3.0),
        airfoil_name=wing_cfg.get("airfoil", "HQW2512"),
    )

    mass = MassProperties(
        empty_mass=mass_cfg.get("structure", 8.0) + mass_cfg.get("avionics", 0.5),
        payload_mass=mass_cfg.get("payload", 2.0),
        battery_mass=mass_cfg.get("battery", 2.5),
        solar_cell_mass=mass_cfg.get("solar_cells", 1.5),
        cg_position=mass_cfg.get("cg_position", 0.28),
    )

    return GliderModel(wing=wing, mass=mass)


def cmd_analyze(args, config: dict):
    """Run aerodynamic analysis."""
    print("=" * 60)
    print("HighEfficiencyGlide - Aerodynamic Analysis")
    print("=" * 60)

    # Override config with command line args
    if args.wingspan:
        config.setdefault("wing", {})["wingspan"] = args.wingspan
    if args.ar:
        # Compute root chord from AR
        wingspan = config.get("wing", {}).get("wingspan", 4.0)
        taper = config.get("wing", {}).get("taper_ratio", 0.45)
        wing_area = wingspan**2 / args.ar
        root_chord = 2 * wing_area / (wingspan * (1 + taper))
        config.setdefault("wing", {})["root_chord"] = root_chord

    glider = create_glider_from_config(config)
    summary = glider.summary()

    print("\nGeometry:")
    print(f"  Wingspan:      {summary['geometry']['wingspan']:.2f} m")
    print(f"  Wing Area:     {summary['geometry']['wing_area']:.3f} m²")
    print(f"  Aspect Ratio:  {summary['geometry']['aspect_ratio']:.1f}")
    print(f"  MAC:           {summary['geometry']['mac']:.3f} m")
    print(f"  Taper Ratio:   {summary['geometry']['taper_ratio']:.2f}")

    print("\nMass:")
    print(f"  Total Mass:    {summary['mass']['total_mass']:.2f} kg")
    print(f"  Weight:        {summary['mass']['weight']:.1f} N")
    print(f"  Wing Loading:  {summary['mass']['wing_loading']:.1f} N/m²")

    print("\nAerodynamics:")
    print(f"  Oswald e:      {summary['aerodynamics']['oswald_efficiency']:.3f}")
    print(f"  CD0:           {summary['aerodynamics']['CD0']:.5f}")
    print(f"  CLα:           {summary['aerodynamics']['CL_alpha']:.4f} /deg")

    print("\nPerformance:")
    print(f"  Best L/D:      {summary['performance']['max_glide_ratio']:.1f}")
    print(f"  Best Glide V:  {summary['performance']['best_glide_speed']:.1f} m/s")
    print(f"  Min Sink V:    {summary['performance']['min_sink_speed']:.1f} m/s")
    print(f"  Min Sink Rate: {summary['performance']['min_sink_rate']:.2f} m/s")
    print(f"  Stall Speed:   {summary['performance']['stall_speed']:.1f} m/s")

    print("\nArduPilot Parameters:")
    print(f"  SOAR_POLAR_K:  {summary['ardupilot']['SOAR_POLAR_K']:.1f}")

    # Solar panel area
    solar_area = glider.wing.solar_panel_area()
    print(f"\nSolar Panel Area: {solar_area:.3f} m²")

    return glider


def cmd_envelope(args, config: dict):
    """Generate performance envelope."""
    print("=" * 60)
    print("HighEfficiencyGlide - Performance Envelope")
    print("=" * 60)

    glider = create_glider_from_config(config)
    altitude = args.altitude or config.get("simulation", {}).get("reference_altitude", 1000)

    print(f"\nComputing envelope at {altitude}m altitude...")
    envelope = glider.compute_performance_envelope(
        altitude=altitude,
        v_min=config.get("simulation", {}).get("v_min", 8.0),
        v_max=config.get("simulation", {}).get("v_max", 40.0),
        num_points=50,
    )

    # Find key points
    idx_ld_max = np.argmax(envelope["glide_ratio"])
    valid_sink = envelope["sink_rate"] > 0
    idx_min_sink = np.argmin(np.where(valid_sink, envelope["sink_rate"], np.inf))

    print(f"\nBest Glide (max L/D):")
    print(f"  Velocity:    {envelope['velocity'][idx_ld_max]:.1f} m/s")
    print(f"  L/D:         {envelope['glide_ratio'][idx_ld_max]:.1f}")
    print(f"  CL:          {envelope['cl'][idx_ld_max]:.3f}")
    print(f"  Alpha:       {envelope['alpha'][idx_ld_max]:.1f}°")

    print(f"\nMinimum Sink:")
    print(f"  Velocity:    {envelope['velocity'][idx_min_sink]:.1f} m/s")
    print(f"  Sink Rate:   {envelope['sink_rate'][idx_min_sink]:.2f} m/s")
    print(f"  L/D:         {envelope['glide_ratio'][idx_min_sink]:.1f}")

    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save data
        import json
        data_file = output_path / "envelope_data.json"
        with open(data_file, "w") as f:
            json.dump({k: v.tolist() for k, v in envelope.items()}, f, indent=2)
        print(f"\nData saved to: {data_file}")

        # Try to generate plots
        try:
            from .visualization.plots import plot_performance_envelope
            plot_file = output_path / "performance_envelope.png"
            plot_performance_envelope(envelope, save_path=str(plot_file))
            print(f"Plot saved to: {plot_file}")
        except ImportError:
            print("Visualization module not available, skipping plots")

    return envelope


def cmd_optimize(args, config: dict):
    """Run wing optimization."""
    print("=" * 60)
    print("HighEfficiencyGlide - Wing Optimization")
    print("=" * 60)

    try:
        from .optimization.genetic_algorithm import run_optimization
    except ImportError:
        print("Error: Optimization module not available")
        print("Run: pip install scipy")
        return None

    opt_config = config.get("optimization", {})
    generations = args.generations or opt_config.get("generations", 50)
    pop_size = args.population or opt_config.get("population_size", 100)

    print(f"\nRunning optimization:")
    print(f"  Algorithm:    {opt_config.get('algorithm', 'NSGA-II')}")
    print(f"  Population:   {pop_size}")
    print(f"  Generations:  {generations}")

    result = run_optimization(
        config=config,
        population_size=pop_size,
        generations=generations,
        verbose=True,
    )

    if result:
        print("\nOptimization complete!")
        print(f"Best design: {result}")

    return result


def cmd_summary(args, config: dict):
    """Print glider summary."""
    glider = create_glider_from_config(config)
    summary = glider.summary()

    print("\n" + "=" * 60)
    print("GLIDER SUMMARY")
    print("=" * 60)

    for category, values in summary.items():
        print(f"\n{category.upper()}:")
        for key, value in values.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4g}")
            else:
                print(f"  {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HighEfficiencyGlide - Solar Glider Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration YAML file",
        default=None,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run aerodynamic analysis")
    analyze_parser.add_argument("--wingspan", type=float, help="Wing span (m)")
    analyze_parser.add_argument("--ar", type=float, help="Aspect ratio")

    # Envelope command
    envelope_parser = subparsers.add_parser("envelope", help="Generate performance envelope")
    envelope_parser.add_argument("--altitude", type=float, help="Altitude (m)")
    envelope_parser.add_argument("--output", "-o", help="Output directory for plots")

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Run wing optimization")
    optimize_parser.add_argument("--generations", "-g", type=int, help="Number of generations")
    optimize_parser.add_argument("--population", "-p", type=int, help="Population size")

    # Summary command
    subparsers.add_parser("summary", help="Print glider summary")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command is None:
        # Default to summary
        cmd_summary(args, config)
    elif args.command == "analyze":
        cmd_analyze(args, config)
    elif args.command == "envelope":
        cmd_envelope(args, config)
    elif args.command == "optimize":
        cmd_optimize(args, config)
    elif args.command == "summary":
        cmd_summary(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
