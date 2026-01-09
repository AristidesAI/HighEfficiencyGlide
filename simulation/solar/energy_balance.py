"""
Energy Balance Analysis.

Computes power generation vs consumption to determine
if self-sustaining flight is achievable.

Components:
- Solar power generation (varies with sun, attitude)
- Power consumption (avionics, servos, compute)
- Battery state of charge
- Energy margin analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from ..aerodynamics.glider_model import GliderModel
from .cell_model import SolarCellModel, CellType, get_solar_model
from .placement_optimizer import SolarPlacementOptimizer, PlacementResult


@dataclass
class PowerConsumers:
    """Power consumption components."""

    flight_controller: float = 3.0  # Watts
    servos_idle: float = 2.0        # Watts (holding position)
    servos_active: float = 8.0      # Watts (during maneuvers)
    telemetry_radio: float = 1.5    # Watts
    gps_receiver: float = 0.5       # Watts
    sensors: float = 1.0            # Watts (airspeed, etc.)
    compute_idle: float = 5.0       # Watts (Raspberry Pi)
    compute_vision: float = 15.0    # Watts (Jetson Nano ML inference)
    camera: float = 2.0             # Watts
    lighting: float = 0.0           # Watts (navigation lights if any)

    def idle_power(self) -> float:
        """Total power consumption in cruise (idle)."""
        return (
            self.flight_controller +
            self.servos_idle +
            self.telemetry_radio +
            self.gps_receiver +
            self.sensors +
            self.compute_idle
        )

    def active_power(self) -> float:
        """Total power consumption during active operations."""
        return (
            self.flight_controller +
            self.servos_active +
            self.telemetry_radio +
            self.gps_receiver +
            self.sensors +
            self.compute_vision +
            self.camera
        )

    def average_power(self, active_fraction: float = 0.2) -> float:
        """
        Weighted average power consumption.

        Args:
            active_fraction: Fraction of time in active mode

        Returns:
            Average power consumption (Watts)
        """
        return (
            active_fraction * self.active_power() +
            (1 - active_fraction) * self.idle_power()
        )


@dataclass
class BatterySpecs:
    """Battery specifications."""

    capacity_wh: float = 50.0       # Watt-hours
    voltage_nominal: float = 11.1   # Volts (3S LiPo)
    discharge_rate_max: float = 2.0  # C-rate
    charge_rate_max: float = 1.0    # C-rate
    efficiency: float = 0.95        # Round-trip efficiency
    min_soc: float = 0.20           # Minimum state of charge (protect battery)
    mass: float = 0.4               # kg

    @property
    def capacity_ah(self) -> float:
        """Capacity in Amp-hours."""
        return self.capacity_wh / self.voltage_nominal

    @property
    def usable_capacity_wh(self) -> float:
        """Usable capacity (accounting for min SOC)."""
        return self.capacity_wh * (1 - self.min_soc)


@dataclass
class EnergyState:
    """Current energy state."""

    time: float  # Time in hours from start
    solar_power: float  # Current solar power generation (W)
    consumption: float  # Current power consumption (W)
    net_power: float  # Net power (positive = charging)
    battery_soc: float  # State of charge (0-1)
    battery_energy: float  # Current battery energy (Wh)
    is_sustainable: bool  # Net positive energy


@dataclass
class EnergyBalanceResult:
    """Results of energy balance analysis."""

    avg_solar_power: float  # Average solar generation (W)
    avg_consumption: float  # Average consumption (W)
    net_power: float  # Average net power (W)
    energy_margin: float  # Ratio of generation to consumption
    min_battery_soc: float  # Minimum SOC during mission
    is_sustainable: bool  # Can maintain flight indefinitely
    max_endurance_hours: float  # Maximum flight time if not sustainable
    timeline: List[EnergyState]  # Time series of energy states


class EnergyBalanceAnalyzer:
    """
    Analyzes energy balance for solar glider.

    Determines if the glider can achieve energy-neutral or
    energy-positive flight under given conditions.
    """

    def __init__(
        self,
        glider: GliderModel,
        solar_area: float,
        solar_model: Optional[SolarCellModel] = None,
        battery: Optional[BatterySpecs] = None,
        consumers: Optional[PowerConsumers] = None,
    ):
        """
        Initialize energy analyzer.

        Args:
            glider: Glider model
            solar_area: Total solar panel area (m²)
            solar_model: Solar cell model
            battery: Battery specifications
            consumers: Power consumption model
        """
        self.glider = glider
        self.solar_area = solar_area
        self.solar_model = solar_model or get_solar_model()
        self.battery = battery or BatterySpecs()
        self.consumers = consumers or PowerConsumers()

    def compute_solar_power(
        self,
        sun_elevation: float,
        ambient_temp: float = 20.0,
        airspeed: float = 15.0,
        incidence_angle: float = 0.0,
    ) -> float:
        """
        Compute solar power generation.

        Args:
            sun_elevation: Sun elevation angle (degrees)
            ambient_temp: Ambient temperature (°C)
            airspeed: Flight speed (m/s) - affects cooling
            incidence_angle: Panel incidence angle (degrees)

        Returns:
            Power generation (Watts)
        """
        if sun_elevation <= 0:
            return 0.0

        # Solar irradiance model
        # Direct normal irradiance at altitude
        air_mass = 1 / np.sin(np.radians(max(1, sun_elevation)))
        dni = 1361 * 0.7 ** (air_mass ** 0.678)  # Simplified atmospheric model

        # Global horizontal irradiance (simplified)
        ghi = dni * np.sin(np.radians(sun_elevation))

        # Irradiance on wing (approximately horizontal + tilt)
        wing_irradiance = ghi * np.cos(np.radians(incidence_angle))

        # Cell temperature (airspeed provides cooling)
        cell_temp = self.solar_model.cell_temperature(
            wing_irradiance,
            ambient_temp,
            wind_speed=airspeed
        )

        # Power output
        power = self.solar_model.power_output(
            self.solar_area,
            wing_irradiance,
            cell_temp,
            incidence_angle=incidence_angle
        )

        return power

    def simulate_day(
        self,
        latitude: float,
        day_of_year: int = 172,  # Summer solstice
        start_soc: float = 1.0,
        duration_hours: float = 12.0,
        time_step_minutes: float = 5.0,
        active_fraction: float = 0.2,
    ) -> EnergyBalanceResult:
        """
        Simulate energy balance over a day.

        Args:
            latitude: Flight latitude (degrees)
            day_of_year: Day of year (1-365)
            start_soc: Initial battery state of charge (0-1)
            duration_hours: Simulation duration (hours)
            time_step_minutes: Time step (minutes)
            active_fraction: Fraction of time in active mode

        Returns:
            EnergyBalanceResult with timeline and metrics
        """
        timeline = []
        dt = time_step_minutes / 60  # Convert to hours

        # Initialize state
        battery_energy = self.battery.capacity_wh * start_soc
        min_soc = start_soc

        # Solar parameters
        declination = 23.45 * np.sin(np.radians(360 / 365 * (day_of_year - 81)))

        total_solar = 0.0
        total_consumption = 0.0

        for step in range(int(duration_hours * 60 / time_step_minutes)):
            time_hours = step * dt

            # Solar hour angle (assume noon start)
            hour_angle = 15 * (time_hours - 6)  # Degrees per hour

            # Sun elevation
            sin_elevation = (
                np.sin(np.radians(latitude)) * np.sin(np.radians(declination)) +
                np.cos(np.radians(latitude)) * np.cos(np.radians(declination)) *
                np.cos(np.radians(hour_angle))
            )
            sun_elevation = np.degrees(np.arcsin(max(-1, min(1, sin_elevation))))

            # Solar power
            solar_power = self.compute_solar_power(sun_elevation)

            # Consumption (varies with activity)
            consumption = self.consumers.average_power(active_fraction)

            # Net power
            net_power = solar_power - consumption

            # Update battery
            if net_power > 0:
                # Charging (with efficiency loss)
                charge_power = min(
                    net_power * self.battery.efficiency,
                    self.battery.capacity_wh * self.battery.charge_rate_max
                )
                battery_energy += charge_power * dt
                battery_energy = min(battery_energy, self.battery.capacity_wh)
            else:
                # Discharging
                battery_energy += net_power * dt
                battery_energy = max(0, battery_energy)

            soc = battery_energy / self.battery.capacity_wh
            min_soc = min(min_soc, soc)

            # Record state
            state = EnergyState(
                time=time_hours,
                solar_power=solar_power,
                consumption=consumption,
                net_power=net_power,
                battery_soc=soc,
                battery_energy=battery_energy,
                is_sustainable=net_power > 0,
            )
            timeline.append(state)

            total_solar += solar_power * dt
            total_consumption += consumption * dt

        # Compute summary
        avg_solar = total_solar / duration_hours if duration_hours > 0 else 0
        avg_consumption = total_consumption / duration_hours if duration_hours > 0 else 0
        net_avg = avg_solar - avg_consumption
        energy_margin = avg_solar / avg_consumption if avg_consumption > 0 else 0

        # Endurance calculation
        if net_avg >= 0:
            max_endurance = float("inf")
        else:
            usable_energy = self.battery.usable_capacity_wh
            max_endurance = usable_energy / abs(net_avg)

        is_sustainable = energy_margin >= 1.0 and min_soc >= self.battery.min_soc

        return EnergyBalanceResult(
            avg_solar_power=avg_solar,
            avg_consumption=avg_consumption,
            net_power=net_avg,
            energy_margin=energy_margin,
            min_battery_soc=min_soc,
            is_sustainable=is_sustainable,
            max_endurance_hours=max_endurance,
            timeline=timeline,
        )

    def minimum_solar_area(
        self,
        target_margin: float = 1.2,
        latitude: float = 40.0,
    ) -> float:
        """
        Calculate minimum solar area for sustainable flight.

        Args:
            target_margin: Target energy margin (1.0 = break even)
            latitude: Flight latitude (degrees)

        Returns:
            Minimum required solar area (m²)
        """
        # Average daily consumption
        avg_consumption = self.consumers.average_power(0.2)

        # Estimate average solar generation per m²
        # Rough estimate for mid-latitudes, summer
        avg_daily_solar_hours = 6.0  # Effective full-sun hours
        peak_irradiance = 800.0  # W/m² effective

        generation_per_m2 = (
            peak_irradiance *
            self.solar_model.specs.efficiency_stc *
            avg_daily_solar_hours
        ) / 24  # Convert to average Watts

        # Required generation
        required_generation = avg_consumption * target_margin

        # Required area
        required_area = required_generation / generation_per_m2

        return required_area

    def summary(self) -> Dict:
        """Get summary of energy system."""
        # Quick analysis at noon
        noon_power = self.compute_solar_power(sun_elevation=60)
        idle_consumption = self.consumers.idle_power()
        active_consumption = self.consumers.active_power()

        return {
            "solar": {
                "area_m2": self.solar_area,
                "peak_power_w": noon_power,
                "efficiency": self.solar_model.specs.efficiency_stc,
            },
            "consumption": {
                "idle_w": idle_consumption,
                "active_w": active_consumption,
                "average_w": self.consumers.average_power(),
            },
            "battery": {
                "capacity_wh": self.battery.capacity_wh,
                "usable_wh": self.battery.usable_capacity_wh,
                "mass_kg": self.battery.mass,
            },
            "balance": {
                "peak_surplus_w": noon_power - idle_consumption,
                "margin_at_peak": noon_power / idle_consumption if idle_consumption > 0 else 0,
            },
        }


def quick_energy_check(
    solar_area: float,
    power_consumption: float = 15.0,
    cell_efficiency: float = 0.22,
    latitude: float = 40.0,
) -> Dict:
    """
    Quick check if energy balance is feasible.

    Args:
        solar_area: Solar panel area (m²)
        power_consumption: Average power consumption (W)
        cell_efficiency: Solar cell efficiency
        latitude: Mission latitude (degrees)

    Returns:
        Dictionary with feasibility assessment
    """
    # Rough estimates
    summer_sun_hours = 8.0 - abs(latitude) / 15  # Effective hours
    avg_irradiance = 600.0  # W/m² average over day

    daily_generation = solar_area * avg_irradiance * cell_efficiency * summer_sun_hours
    daily_consumption = power_consumption * 24

    return {
        "daily_generation_wh": daily_generation,
        "daily_consumption_wh": daily_consumption,
        "energy_margin": daily_generation / daily_consumption if daily_consumption > 0 else 0,
        "is_feasible": daily_generation >= daily_consumption,
        "surplus_wh": daily_generation - daily_consumption,
    }
