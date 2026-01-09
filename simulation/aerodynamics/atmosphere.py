"""
International Standard Atmosphere (ISA) Model.

Provides atmospheric properties as a function of altitude for aerodynamic calculations.
Based on the ISA model up to 11km (troposphere).

Reference: ICAO Standard Atmosphere (Doc 7488/3)
"""

import numpy as np
from dataclasses import dataclass
from typing import Union, Tuple

# ISA constants
SEA_LEVEL_TEMPERATURE = 288.15  # K (15°C)
SEA_LEVEL_PRESSURE = 101325.0  # Pa
SEA_LEVEL_DENSITY = 1.225  # kg/m³
TEMPERATURE_LAPSE_RATE = -0.0065  # K/m (troposphere)
GAS_CONSTANT = 287.058  # J/(kg·K)
GRAVITY = 9.80665  # m/s²
GAMMA = 1.4  # Ratio of specific heats for air
SUTHERLAND_CONSTANT = 110.4  # K
SUTHERLAND_REF_VISCOSITY = 1.716e-5  # Pa·s at 273.15 K


@dataclass
class AtmosphericState:
    """Atmospheric conditions at a given altitude."""

    altitude: float  # m
    temperature: float  # K
    pressure: float  # Pa
    density: float  # kg/m³
    dynamic_viscosity: float  # Pa·s
    kinematic_viscosity: float  # m²/s
    speed_of_sound: float  # m/s


class ISAAtmosphere:
    """
    International Standard Atmosphere model.

    Valid for altitudes from 0 to 11,000m (troposphere).
    Above 11km, temperature is assumed constant (isothermal stratosphere).
    """

    def __init__(self):
        self.T0 = SEA_LEVEL_TEMPERATURE
        self.P0 = SEA_LEVEL_PRESSURE
        self.rho0 = SEA_LEVEL_DENSITY
        self.L = TEMPERATURE_LAPSE_RATE
        self.R = GAS_CONSTANT
        self.g = GRAVITY
        self.gamma = GAMMA

        # Tropopause altitude
        self.h_tropopause = 11000.0  # m

    def temperature(self, altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate temperature at altitude.

        Args:
            altitude: Geometric altitude in meters

        Returns:
            Temperature in Kelvin
        """
        altitude = np.asarray(altitude)
        T = np.where(
            altitude <= self.h_tropopause,
            self.T0 + self.L * altitude,
            self.T0 + self.L * self.h_tropopause  # Isothermal above tropopause
        )
        return float(T) if T.ndim == 0 else T

    def pressure(self, altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate pressure at altitude using barometric formula.

        Args:
            altitude: Geometric altitude in meters

        Returns:
            Pressure in Pascals
        """
        altitude = np.asarray(altitude)
        T = self.temperature(altitude)

        # Troposphere (with temperature lapse)
        exponent = -self.g / (self.R * self.L)
        P_tropo = self.P0 * (T / self.T0) ** exponent

        # Above tropopause (isothermal)
        T_tropo = self.temperature(self.h_tropopause)
        P_tropo_base = self.P0 * (T_tropo / self.T0) ** exponent
        P_strato = P_tropo_base * np.exp(-self.g * (altitude - self.h_tropopause) / (self.R * T_tropo))

        P = np.where(altitude <= self.h_tropopause, P_tropo, P_strato)
        return float(P) if P.ndim == 0 else P

    def density(self, altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate air density at altitude using ideal gas law.

        Args:
            altitude: Geometric altitude in meters

        Returns:
            Density in kg/m³
        """
        T = self.temperature(altitude)
        P = self.pressure(altitude)
        rho = P / (self.R * T)
        return rho

    def dynamic_viscosity(self, altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate dynamic viscosity using Sutherland's formula.

        Args:
            altitude: Geometric altitude in meters

        Returns:
            Dynamic viscosity in Pa·s
        """
        T = self.temperature(altitude)
        T_ref = 273.15
        mu = SUTHERLAND_REF_VISCOSITY * (T / T_ref) ** 1.5 * (T_ref + SUTHERLAND_CONSTANT) / (T + SUTHERLAND_CONSTANT)
        return mu

    def kinematic_viscosity(self, altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate kinematic viscosity.

        Args:
            altitude: Geometric altitude in meters

        Returns:
            Kinematic viscosity in m²/s
        """
        mu = self.dynamic_viscosity(altitude)
        rho = self.density(altitude)
        return mu / rho

    def speed_of_sound(self, altitude: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate speed of sound.

        Args:
            altitude: Geometric altitude in meters

        Returns:
            Speed of sound in m/s
        """
        T = self.temperature(altitude)
        return np.sqrt(self.gamma * self.R * T)

    def get_state(self, altitude: float) -> AtmosphericState:
        """
        Get complete atmospheric state at altitude.

        Args:
            altitude: Geometric altitude in meters

        Returns:
            AtmosphericState dataclass with all properties
        """
        return AtmosphericState(
            altitude=altitude,
            temperature=self.temperature(altitude),
            pressure=self.pressure(altitude),
            density=self.density(altitude),
            dynamic_viscosity=self.dynamic_viscosity(altitude),
            kinematic_viscosity=self.kinematic_viscosity(altitude),
            speed_of_sound=self.speed_of_sound(altitude),
        )

    def reynolds_number(
        self,
        velocity: float,
        characteristic_length: float,
        altitude: float
    ) -> float:
        """
        Calculate Reynolds number.

        Args:
            velocity: Airspeed in m/s
            characteristic_length: Reference length (e.g., chord) in meters
            altitude: Altitude in meters

        Returns:
            Reynolds number (dimensionless)
        """
        nu = self.kinematic_viscosity(altitude)
        return velocity * characteristic_length / nu

    def mach_number(self, velocity: float, altitude: float) -> float:
        """
        Calculate Mach number.

        Args:
            velocity: Airspeed in m/s
            altitude: Altitude in meters

        Returns:
            Mach number (dimensionless)
        """
        a = self.speed_of_sound(altitude)
        return velocity / a

    def dynamic_pressure(self, velocity: float, altitude: float) -> float:
        """
        Calculate dynamic pressure (q = 0.5 * rho * V²).

        Args:
            velocity: Airspeed in m/s
            altitude: Altitude in meters

        Returns:
            Dynamic pressure in Pascals
        """
        rho = self.density(altitude)
        return 0.5 * rho * velocity ** 2

    def true_airspeed_from_eas(
        self,
        equivalent_airspeed: float,
        altitude: float
    ) -> float:
        """
        Convert Equivalent Airspeed (EAS) to True Airspeed (TAS).

        Args:
            equivalent_airspeed: EAS in m/s
            altitude: Altitude in meters

        Returns:
            True Airspeed in m/s
        """
        rho = self.density(altitude)
        return equivalent_airspeed * np.sqrt(self.rho0 / rho)

    def eas_from_true_airspeed(
        self,
        true_airspeed: float,
        altitude: float
    ) -> float:
        """
        Convert True Airspeed (TAS) to Equivalent Airspeed (EAS).

        Args:
            true_airspeed: TAS in m/s
            altitude: Altitude in meters

        Returns:
            Equivalent Airspeed in m/s
        """
        rho = self.density(altitude)
        return true_airspeed * np.sqrt(rho / self.rho0)


# Module-level singleton for convenience
_atmosphere = ISAAtmosphere()


def get_atmosphere() -> ISAAtmosphere:
    """Get the singleton atmosphere instance."""
    return _atmosphere
