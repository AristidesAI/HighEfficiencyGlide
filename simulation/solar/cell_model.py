"""
Solar Cell Performance Model.

Models photovoltaic cell behavior including:
- IV curve characteristics
- Temperature effects on efficiency
- Incidence angle effects
- Spectral considerations

Reference: Standard Test Conditions (STC)
- Irradiance: 1000 W/m²
- Cell temperature: 25°C
- Air mass: AM1.5
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class CellType(Enum):
    """Types of solar cells."""
    MONOCRYSTALLINE = "monocrystalline"  # Highest efficiency, ~22%
    POLYCRYSTALLINE = "polycrystalline"  # Lower cost, ~18%
    THIN_FILM = "thin_film"              # Flexible, ~12%
    PEROVSKITE = "perovskite"            # Emerging, ~25%+
    MULTIJUNCTION = "multijunction"      # Space-grade, ~30%+


@dataclass
class CellSpecifications:
    """Solar cell specifications."""

    cell_type: CellType
    efficiency_stc: float  # Efficiency at STC (0-1)
    temp_coefficient: float  # Power temperature coefficient (%/°C)
    weight_per_area: float  # kg/m²
    cost_per_watt: float  # $/W
    voc: float  # Open circuit voltage (V)
    isc: float  # Short circuit current (A)
    vmp: float  # Voltage at max power (V)
    imp: float  # Current at max power (A)
    fill_factor: float  # Fill factor (0-1)
    area: float  # Cell area (m²)


# Standard cell specifications
CELL_SPECS = {
    CellType.MONOCRYSTALLINE: CellSpecifications(
        cell_type=CellType.MONOCRYSTALLINE,
        efficiency_stc=0.22,
        temp_coefficient=-0.35,
        weight_per_area=0.7,
        cost_per_watt=0.30,
        voc=0.70,
        isc=9.5,
        vmp=0.58,
        imp=9.0,
        fill_factor=0.80,
        area=0.0156,  # 156mm x 100mm typical
    ),
    CellType.POLYCRYSTALLINE: CellSpecifications(
        cell_type=CellType.POLYCRYSTALLINE,
        efficiency_stc=0.18,
        temp_coefficient=-0.40,
        weight_per_area=0.75,
        cost_per_watt=0.25,
        voc=0.65,
        isc=9.0,
        vmp=0.53,
        imp=8.5,
        fill_factor=0.78,
        area=0.0156,
    ),
    CellType.THIN_FILM: CellSpecifications(
        cell_type=CellType.THIN_FILM,
        efficiency_stc=0.12,
        temp_coefficient=-0.20,
        weight_per_area=0.4,
        cost_per_watt=0.20,
        voc=0.90,
        isc=5.5,
        vmp=0.70,
        imp=5.0,
        fill_factor=0.70,
        area=0.01,
    ),
    CellType.PEROVSKITE: CellSpecifications(
        cell_type=CellType.PEROVSKITE,
        efficiency_stc=0.25,
        temp_coefficient=-0.30,
        weight_per_area=0.5,
        cost_per_watt=0.15,
        voc=1.10,
        isc=25.0,
        vmp=0.95,
        imp=23.0,
        fill_factor=0.82,
        area=0.01,
    ),
}


class SolarCellModel:
    """
    Photovoltaic cell performance model.

    Models power output as function of:
    - Solar irradiance
    - Cell temperature
    - Incidence angle
    """

    def __init__(self, specs: Optional[CellSpecifications] = None):
        """
        Initialize solar cell model.

        Args:
            specs: Cell specifications (default: monocrystalline)
        """
        self.specs = specs or CELL_SPECS[CellType.MONOCRYSTALLINE]

        # Reference conditions (STC)
        self.G_stc = 1000.0  # W/m²
        self.T_stc = 25.0    # °C

    def irradiance_factor(self, irradiance: float) -> float:
        """
        Power scaling factor for irradiance.

        Power is approximately linear with irradiance.

        Args:
            irradiance: Solar irradiance (W/m²)

        Returns:
            Power factor (0-1+)
        """
        return irradiance / self.G_stc

    def temperature_factor(self, cell_temp: float) -> float:
        """
        Power scaling factor for temperature.

        Cell efficiency decreases at higher temperatures.

        Args:
            cell_temp: Cell temperature (°C)

        Returns:
            Power factor (typically 0.8-1.1)
        """
        delta_t = cell_temp - self.T_stc
        factor = 1 + self.specs.temp_coefficient / 100 * delta_t
        return max(0.5, factor)  # Limit to reasonable range

    def cell_temperature(
        self,
        irradiance: float,
        ambient_temp: float,
        wind_speed: float = 5.0,
        noct: float = 45.0
    ) -> float:
        """
        Estimate cell temperature from operating conditions.

        Uses NOCT (Nominal Operating Cell Temperature) model.

        Args:
            irradiance: Solar irradiance (W/m²)
            ambient_temp: Ambient air temperature (°C)
            wind_speed: Wind speed (m/s)
            noct: Nominal Operating Cell Temperature (°C)

        Returns:
            Estimated cell temperature (°C)
        """
        # NOCT conditions: 800 W/m², 20°C ambient, 1 m/s wind
        G_noct = 800.0
        T_noct_ambient = 20.0
        wind_noct = 1.0

        # Temperature rise above ambient
        # Adjusted for actual wind speed
        wind_factor = (wind_noct / max(0.5, wind_speed)) ** 0.5
        delta_t = (noct - T_noct_ambient) * (irradiance / G_noct) * wind_factor

        return ambient_temp + delta_t

    def incidence_angle_factor(self, angle: float) -> float:
        """
        Power reduction factor for non-normal incidence.

        Uses cosine law with air-mass correction.

        Args:
            angle: Incidence angle from normal (degrees)

        Returns:
            Power factor (0-1)
        """
        angle_rad = np.radians(np.abs(angle))

        if angle > 85:
            return 0.0

        # Basic cosine law
        cos_factor = np.cos(angle_rad)

        # IAM (Incidence Angle Modifier) correction
        # Accounts for reflection losses at high angles
        b0 = 0.05  # IAM coefficient
        iam = 1 - b0 * (1 / cos_factor - 1)

        return max(0, cos_factor * iam)

    def power_output(
        self,
        area: float,
        irradiance: float,
        cell_temp: Optional[float] = None,
        ambient_temp: float = 25.0,
        incidence_angle: float = 0.0,
        wind_speed: float = 5.0,
    ) -> float:
        """
        Calculate power output from solar panel.

        Args:
            area: Panel area (m²)
            irradiance: Solar irradiance (W/m²)
            cell_temp: Cell temperature (°C), calculated if None
            ambient_temp: Ambient temperature (°C)
            incidence_angle: Sun incidence angle (degrees)
            wind_speed: Wind speed for cooling (m/s)

        Returns:
            Power output (Watts)
        """
        if irradiance <= 0:
            return 0.0

        # Calculate cell temperature if not provided
        if cell_temp is None:
            cell_temp = self.cell_temperature(irradiance, ambient_temp, wind_speed)

        # Apply all factors
        f_irr = self.irradiance_factor(irradiance)
        f_temp = self.temperature_factor(cell_temp)
        f_angle = self.incidence_angle_factor(incidence_angle)

        # Power at STC
        p_stc = area * self.G_stc * self.specs.efficiency_stc

        # Actual power
        power = p_stc * f_irr * f_temp * f_angle

        return max(0, power)

    def iv_curve(
        self,
        irradiance: float = 1000.0,
        cell_temp: float = 25.0,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate IV curve for the cell.

        Args:
            irradiance: Solar irradiance (W/m²)
            cell_temp: Cell temperature (°C)
            num_points: Number of points in curve

        Returns:
            Tuple of (voltage array, current array)
        """
        # Scale parameters for conditions
        f_irr = self.irradiance_factor(irradiance)
        f_temp = self.temperature_factor(cell_temp)

        isc = self.specs.isc * f_irr
        voc = self.specs.voc * (1 + 0.0005 * (cell_temp - self.T_stc))  # Slight Voc increase with temp

        # Single-diode model approximation
        v = np.linspace(0, voc, num_points)

        # Simplified IV curve using fill factor
        # I = Isc * (1 - (V/Voc)^n) where n adjusted for fill factor
        n = np.log(1 - self.specs.fill_factor) / np.log(1 - self.specs.vmp / self.specs.voc)
        i = isc * (1 - (v / voc) ** (1 / n))
        i = np.maximum(0, i)

        return v, i

    def mpp_tracking(
        self,
        irradiance: float,
        cell_temp: float
    ) -> Tuple[float, float, float]:
        """
        Find Maximum Power Point.

        Args:
            irradiance: Solar irradiance (W/m²)
            cell_temp: Cell temperature (°C)

        Returns:
            Tuple of (Vmp, Imp, Pmp)
        """
        v, i = self.iv_curve(irradiance, cell_temp)
        p = v * i
        idx = np.argmax(p)

        return v[idx], i[idx], p[idx]


def get_solar_model(cell_type: CellType = CellType.MONOCRYSTALLINE) -> SolarCellModel:
    """Get solar cell model for specified cell type."""
    return SolarCellModel(CELL_SPECS.get(cell_type))


def estimate_solar_mass(area: float, cell_type: CellType = CellType.MONOCRYSTALLINE) -> float:
    """
    Estimate mass of solar panels.

    Includes cells, encapsulation, and wiring.

    Args:
        area: Panel area (m²)
        cell_type: Type of solar cell

    Returns:
        Estimated mass (kg)
    """
    specs = CELL_SPECS.get(cell_type, CELL_SPECS[CellType.MONOCRYSTALLINE])

    # Cell mass
    cell_mass = area * specs.weight_per_area

    # Encapsulation overhead (~20%)
    encapsulation_factor = 1.2

    # Wiring and junction boxes (~5%)
    wiring_factor = 1.05

    return cell_mass * encapsulation_factor * wiring_factor
