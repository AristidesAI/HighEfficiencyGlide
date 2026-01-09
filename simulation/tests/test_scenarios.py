"""
Test Scenarios for Glider Simulation.

Validates glider performance under various real-world conditions:
- Calm air (baseline)
- Steady wind
- Turbulence
- Gusts
- Thermal soaring
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Generator
from enum import Enum
import random


class ConditionType(Enum):
    """Types of atmospheric conditions."""
    CALM = "calm"
    STEADY_WIND = "steady_wind"
    TURBULENCE = "turbulence"
    GUST = "gust"
    THERMAL = "thermal"


@dataclass
class WindState:
    """Current wind state at a point in time/space."""

    u: float  # East component (m/s)
    v: float  # North component (m/s)
    w: float  # Vertical component (m/s, positive = updraft)

    @property
    def speed(self) -> float:
        """Horizontal wind speed (m/s)."""
        return np.sqrt(self.u**2 + self.v**2)

    @property
    def direction(self) -> float:
        """Wind direction (degrees from north, direction wind is coming from)."""
        return np.degrees(np.arctan2(-self.u, -self.v)) % 360

    def relative_to_heading(self, heading: float) -> Tuple[float, float, float]:
        """
        Get wind components relative to aircraft heading.

        Args:
            heading: Aircraft heading (degrees from north)

        Returns:
            Tuple of (headwind, crosswind, updraft) components
        """
        heading_rad = np.radians(heading)

        # Rotate to aircraft frame
        headwind = -self.u * np.sin(heading_rad) - self.v * np.cos(heading_rad)
        crosswind = self.u * np.cos(heading_rad) - self.v * np.sin(heading_rad)

        return headwind, crosswind, self.w


@dataclass
class AtmosphericConditions:
    """Atmospheric conditions for a test scenario."""

    name: str
    condition_type: ConditionType

    # Steady wind parameters
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # degrees from north

    # Turbulence parameters
    turbulence_intensity: float = 0.0  # 0-1 scale
    turbulence_length_scale: float = 100.0  # m

    # Gust parameters
    gust_amplitude: float = 0.0  # m/s
    gust_frequency: float = 0.0  # Hz

    # Thermal parameters
    thermal_strength: float = 0.0  # m/s updraft
    thermal_diameter: float = 200.0  # m
    thermal_position: Tuple[float, float] = (0.0, 0.0)  # (x, y) in m


class WindModel:
    """
    Wind model for simulation.

    Generates wind vectors based on atmospheric conditions.
    """

    def __init__(self, conditions: AtmosphericConditions):
        self.conditions = conditions
        self._time = 0.0

        # Dryden turbulence model state
        self._turb_state = np.zeros(6)  # u, v, w and their derivatives

    def get_wind(
        self,
        position: Tuple[float, float, float],
        time: float
    ) -> WindState:
        """
        Get wind state at position and time.

        Args:
            position: (x, y, z) position in meters
            time: Simulation time in seconds

        Returns:
            WindState at the given point
        """
        x, y, z = position

        # Base steady wind
        wind_dir_rad = np.radians(self.conditions.wind_direction)
        u_base = -self.conditions.wind_speed * np.sin(wind_dir_rad)
        v_base = -self.conditions.wind_speed * np.cos(wind_dir_rad)
        w_base = 0.0

        # Add turbulence
        if self.conditions.turbulence_intensity > 0:
            u_turb, v_turb, w_turb = self._get_turbulence(time, z)
            u_base += u_turb
            v_base += v_turb
            w_base += w_turb

        # Add gusts
        if self.conditions.gust_amplitude > 0:
            gust = self._get_gust(time)
            u_base += gust * np.sin(wind_dir_rad)
            v_base += gust * np.cos(wind_dir_rad)

        # Add thermal
        if self.conditions.thermal_strength > 0:
            w_thermal = self._get_thermal(x, y)
            w_base += w_thermal

        return WindState(u=u_base, v=v_base, w=w_base)

    def _get_turbulence(self, time: float, altitude: float) -> Tuple[float, float, float]:
        """
        Generate turbulence using simplified Dryden model.

        Args:
            time: Current time (s)
            altitude: Altitude (m)

        Returns:
            (u, v, w) turbulence components
        """
        # Turbulence intensity scaling with altitude
        sigma_w = self.conditions.turbulence_intensity * 2.0  # m/s RMS
        sigma_uv = self.conditions.turbulence_intensity * 3.0  # m/s RMS

        # Length scales (increase with altitude)
        L_u = self.conditions.turbulence_length_scale * (altitude / 300) ** 0.33
        L_v = L_u
        L_w = L_u / 2

        # Simplified random process
        # In a full simulation, this would use proper filters
        dt = time - self._time
        self._time = time

        if dt > 0:
            # Correlated random walk
            decay = np.exp(-dt * 15 / self.conditions.turbulence_length_scale)
            noise = np.random.randn(3) * np.sqrt(1 - decay**2)

            self._turb_state[:3] = decay * self._turb_state[:3] + noise * [sigma_uv, sigma_uv, sigma_w]

        return tuple(self._turb_state[:3])

    def _get_gust(self, time: float) -> float:
        """
        Generate discrete gust.

        Uses 1-cosine gust profile.

        Args:
            time: Current time (s)

        Returns:
            Gust velocity (m/s)
        """
        if self.conditions.gust_frequency <= 0:
            return 0.0

        period = 1.0 / self.conditions.gust_frequency
        phase = (time % period) / period

        # 1-cosine gust during first quarter of period
        if phase < 0.25:
            gust_phase = phase * 4  # 0 to 1 over gust duration
            return self.conditions.gust_amplitude * 0.5 * (1 - np.cos(np.pi * gust_phase))
        else:
            return 0.0

    def _get_thermal(self, x: float, y: float) -> float:
        """
        Calculate thermal updraft at position.

        Uses Gaussian thermal model.

        Args:
            x: X position (m)
            y: Y position (m)

        Returns:
            Updraft velocity (m/s)
        """
        tx, ty = self.conditions.thermal_position
        r = np.sqrt((x - tx)**2 + (y - ty)**2)

        # Gaussian thermal profile
        sigma = self.conditions.thermal_diameter / 4
        updraft = self.conditions.thermal_strength * np.exp(-r**2 / (2 * sigma**2))

        return updraft


# Predefined test scenarios
SCENARIOS = {
    "calm": AtmosphericConditions(
        name="Calm conditions",
        condition_type=ConditionType.CALM,
        wind_speed=0.0,
        turbulence_intensity=0.0,
    ),

    "light_wind": AtmosphericConditions(
        name="Light wind from north",
        condition_type=ConditionType.STEADY_WIND,
        wind_speed=5.0,
        wind_direction=0.0,
        turbulence_intensity=0.05,
    ),

    "moderate_wind": AtmosphericConditions(
        name="Moderate wind from northwest",
        condition_type=ConditionType.STEADY_WIND,
        wind_speed=10.0,
        wind_direction=315.0,
        turbulence_intensity=0.1,
    ),

    "strong_wind": AtmosphericConditions(
        name="Strong headwind",
        condition_type=ConditionType.STEADY_WIND,
        wind_speed=15.0,
        wind_direction=180.0,
        turbulence_intensity=0.15,
    ),

    "light_turbulence": AtmosphericConditions(
        name="Light turbulence",
        condition_type=ConditionType.TURBULENCE,
        wind_speed=3.0,
        turbulence_intensity=0.1,
        turbulence_length_scale=100.0,
    ),

    "moderate_turbulence": AtmosphericConditions(
        name="Moderate turbulence",
        condition_type=ConditionType.TURBULENCE,
        wind_speed=5.0,
        turbulence_intensity=0.2,
        turbulence_length_scale=150.0,
    ),

    "severe_turbulence": AtmosphericConditions(
        name="Severe turbulence",
        condition_type=ConditionType.TURBULENCE,
        wind_speed=8.0,
        turbulence_intensity=0.35,
        turbulence_length_scale=200.0,
    ),

    "gusty": AtmosphericConditions(
        name="Gusty conditions",
        condition_type=ConditionType.GUST,
        wind_speed=8.0,
        gust_amplitude=5.0,
        gust_frequency=0.1,
        turbulence_intensity=0.15,
    ),

    "weak_thermal": AtmosphericConditions(
        name="Weak thermal",
        condition_type=ConditionType.THERMAL,
        wind_speed=2.0,
        thermal_strength=1.0,
        thermal_diameter=150.0,
    ),

    "moderate_thermal": AtmosphericConditions(
        name="Moderate thermal",
        condition_type=ConditionType.THERMAL,
        wind_speed=3.0,
        thermal_strength=2.5,
        thermal_diameter=200.0,
    ),

    "strong_thermal": AtmosphericConditions(
        name="Strong thermal",
        condition_type=ConditionType.THERMAL,
        wind_speed=4.0,
        thermal_strength=4.0,
        thermal_diameter=300.0,
    ),
}


def get_scenario(name: str) -> AtmosphericConditions:
    """Get predefined scenario by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]


def run_scenario_test(
    glider,  # GliderModel
    scenario_name: str,
    duration: float = 60.0,
    dt: float = 0.1,
    initial_velocity: float = 15.0,
    initial_altitude: float = 500.0,
) -> dict:
    """
    Run a test scenario and evaluate performance.

    Args:
        glider: GliderModel to test
        scenario_name: Name of scenario to run
        duration: Simulation duration (s)
        dt: Time step (s)
        initial_velocity: Starting airspeed (m/s)
        initial_altitude: Starting altitude (m)

    Returns:
        Dictionary with test results
    """
    scenario = get_scenario(scenario_name)
    wind_model = WindModel(scenario)

    # Initialize state
    position = np.array([0.0, 0.0, initial_altitude])
    velocity = initial_velocity
    heading = 0.0  # North

    # Recording
    results = {
        "scenario": scenario_name,
        "times": [],
        "altitudes": [],
        "airspeeds": [],
        "sink_rates": [],
        "glide_ratios": [],
        "wind_speeds": [],
    }

    time = 0.0
    while time < duration and position[2] > 0:
        # Get wind
        wind = wind_model.get_wind(tuple(position), time)
        headwind, crosswind, updraft = wind.relative_to_heading(heading)

        # Effective airspeed
        airspeed = velocity + headwind

        # Trim alpha for current conditions
        try:
            alpha = glider.find_trim_alpha(airspeed, position[2])
            from ..aerodynamics.glider_model import FlightCondition
            condition = FlightCondition(velocity=airspeed, altitude=position[2], alpha=alpha)
            state = glider.compute_forces(condition)

            # Net sink rate (glider sink - thermal updraft)
            net_sink = state.sink_rate - updraft

            # Record
            results["times"].append(time)
            results["altitudes"].append(position[2])
            results["airspeeds"].append(airspeed)
            results["sink_rates"].append(net_sink)
            results["glide_ratios"].append(state.glide_ratio)
            results["wind_speeds"].append(wind.speed)

            # Update position
            ground_speed = airspeed - headwind
            position[0] += ground_speed * np.cos(np.radians(heading)) * dt
            position[1] += ground_speed * np.sin(np.radians(heading)) * dt
            position[2] -= net_sink * dt

        except Exception:
            # Invalid flight condition
            break

        time += dt

    # Compute summary statistics
    if results["times"]:
        results["summary"] = {
            "final_altitude": position[2],
            "altitude_lost": initial_altitude - position[2],
            "distance_traveled": np.sqrt(position[0]**2 + position[1]**2),
            "avg_sink_rate": np.mean(results["sink_rates"]),
            "max_sink_rate": np.max(results["sink_rates"]),
            "min_sink_rate": np.min(results["sink_rates"]),
            "avg_glide_ratio": np.mean(results["glide_ratios"]),
            "avg_wind": np.mean(results["wind_speeds"]),
        }
    else:
        results["summary"] = {"error": "Simulation failed"}

    return results


def run_all_scenarios(glider) -> dict:
    """
    Run all predefined scenarios.

    Args:
        glider: GliderModel to test

    Returns:
        Dictionary mapping scenario names to results
    """
    all_results = {}
    for name in SCENARIOS:
        print(f"Running scenario: {name}")
        all_results[name] = run_scenario_test(glider, name)
    return all_results


def monte_carlo_analysis(
    glider,
    scenario_name: str,
    num_runs: int = 100,
) -> dict:
    """
    Monte Carlo analysis for robustness evaluation.

    Args:
        glider: GliderModel to test
        scenario_name: Base scenario name
        num_runs: Number of Monte Carlo runs

    Returns:
        Statistical summary of results
    """
    sink_rates = []
    glide_ratios = []
    altitudes_lost = []

    for _ in range(num_runs):
        results = run_scenario_test(glider, scenario_name)
        if "error" not in results.get("summary", {}):
            sink_rates.append(results["summary"]["avg_sink_rate"])
            glide_ratios.append(results["summary"]["avg_glide_ratio"])
            altitudes_lost.append(results["summary"]["altitude_lost"])

    return {
        "scenario": scenario_name,
        "num_runs": num_runs,
        "sink_rate": {
            "mean": np.mean(sink_rates),
            "std": np.std(sink_rates),
            "min": np.min(sink_rates),
            "max": np.max(sink_rates),
        },
        "glide_ratio": {
            "mean": np.mean(glide_ratios),
            "std": np.std(glide_ratios),
            "min": np.min(glide_ratios),
            "max": np.max(glide_ratios),
        },
        "altitude_lost": {
            "mean": np.mean(altitudes_lost),
            "std": np.std(altitudes_lost),
        },
    }
