"""
Core Glider Aerodynamic Model.

Provides complete aerodynamic analysis of the glider including:
- Lift, drag, and moment calculations
- Glide ratio optimization
- Stability analysis
- Performance envelope computation

Key equations:
- Lift: L = 0.5 * ρ * V² * S * CL
- Drag: D = D_parasite + D_induced = 0.5 * ρ * V² * S * (CD0 + CL²/(π*e*AR))
- Glide Ratio: L/D = CL / CD
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
from enum import Enum

from .atmosphere import ISAAtmosphere, get_atmosphere
from .wing_geometry import WingGeometry, oswald_efficiency_factor
from .airfoil_data import AirfoilDatabase, AirfoilPolar, get_airfoil_database


@dataclass
class MassProperties:
    """Glider mass and inertia properties."""

    empty_mass: float  # Structural mass without payload (kg)
    payload_mass: float = 0.0  # Payload mass (kg)
    battery_mass: float = 2.0  # Battery mass (kg)
    solar_cell_mass: float = 1.5  # Solar panel mass (kg)

    # Center of gravity position (fraction of MAC from LE)
    cg_position: float = 0.30

    @property
    def total_mass(self) -> float:
        """Total glider mass (kg)."""
        return self.empty_mass + self.payload_mass + self.battery_mass + self.solar_cell_mass

    @property
    def weight(self) -> float:
        """Total weight (N)."""
        return self.total_mass * 9.80665


@dataclass
class DragComponents:
    """Breakdown of drag contributions."""

    parasite: float  # Zero-lift drag (N)
    induced: float   # Lift-induced drag (N)
    profile: float   # Wing profile drag variation with CL (N)
    interference: float = 0.0  # Wing-fuselage interference (N)

    @property
    def total(self) -> float:
        """Total drag (N)."""
        return self.parasite + self.induced + self.profile + self.interference


@dataclass
class FlightCondition:
    """Flight state variables."""

    velocity: float  # True airspeed (m/s)
    altitude: float  # Geometric altitude (m)
    alpha: float = 0.0  # Angle of attack (degrees)
    bank_angle: float = 0.0  # Bank angle for turning (degrees)
    flap_deflection: float = 0.0  # Flap deflection (degrees)


@dataclass
class AerodynamicState:
    """Complete aerodynamic state at a flight condition."""

    condition: FlightCondition
    lift: float  # Lift force (N)
    drag: float  # Total drag (N)
    drag_components: DragComponents
    moment: float  # Pitching moment (N·m)

    cl: float  # Lift coefficient
    cd: float  # Total drag coefficient
    cm: float  # Moment coefficient

    glide_ratio: float  # L/D
    sink_rate: float  # Vertical sink rate (m/s)
    glide_angle: float  # Glide path angle (degrees)


class GliderModel:
    """
    Complete aerodynamic model for solar glider.

    Combines wing geometry, airfoil data, and atmospheric model
    to compute forces, moments, and performance metrics.
    """

    def __init__(
        self,
        wing: WingGeometry,
        mass: MassProperties,
        atmosphere: Optional[ISAAtmosphere] = None,
        airfoil_db: Optional[AirfoilDatabase] = None,
    ):
        """
        Initialize glider model.

        Args:
            wing: Wing geometry definition
            mass: Mass properties
            atmosphere: Atmosphere model (uses ISA if None)
            airfoil_db: Airfoil database (uses default if None)
        """
        self.wing = wing
        self.mass = mass
        self.atm = atmosphere or get_atmosphere()
        self.airfoil_db = airfoil_db or get_airfoil_database()

        # Get airfoil polar
        self.airfoil = self.airfoil_db.get_airfoil(wing.airfoil_name)
        if self.airfoil is None:
            raise ValueError(f"Airfoil '{wing.airfoil_name}' not found in database")

        # Compute derived quantities
        self._compute_derived_properties()

    def _compute_derived_properties(self):
        """Compute derived aerodynamic properties."""
        # Oswald efficiency
        self.e = oswald_efficiency_factor(self.wing.aspect_ratio, self.wing.taper_ratio)

        # Reference values
        self.S = self.wing.wing_area
        self.b = self.wing.wingspan
        self.AR = self.wing.aspect_ratio
        self.MAC = self.wing.mean_aerodynamic_chord

        # Zero-lift drag coefficient estimate
        # Based on wetted area and form factors
        self._cd0_wing = self._estimate_wing_cd0()
        self._cd0_fuselage = 0.004  # Streamlined fuselage assumption
        self._cd0_tail = 0.002  # Tail surfaces
        self._cd0_interference = 0.002  # Interference drag

        self.CD0 = self._cd0_wing + self._cd0_fuselage + self._cd0_tail + self._cd0_interference

        # Wing lift curve slope (3D correction)
        cl_alpha_2d = self.airfoil.cl_alpha  # per degree
        cl_alpha_2d_rad = cl_alpha_2d * 180 / np.pi
        self.CL_alpha = cl_alpha_2d_rad / (1 + cl_alpha_2d_rad / (np.pi * self.AR))
        self.CL_alpha_deg = self.CL_alpha * np.pi / 180

        # Zero-lift angle (3D)
        self.alpha_0 = self.airfoil.alpha_zero_lift

    def _estimate_wing_cd0(self) -> float:
        """Estimate wing zero-lift drag coefficient."""
        # Skin friction coefficient (turbulent)
        # Assume Re ~ 1e6 for estimation
        cf = 0.455 / (np.log10(1e6))**2.58

        # Form factor for wing
        t_c = 0.12  # Assume 12% thickness ratio
        x_c_max = 0.3  # Location of max thickness
        ff = (1 + 0.6 / x_c_max * t_c + 100 * t_c**4) * 1.34  # With compressibility

        # Wetted area ratio (both sides of wing)
        swet_sref = 2.0 * (1 + 0.2 * t_c)

        return cf * ff * swet_sref

    def compute_cl(self, alpha: float, flap: float = 0.0) -> float:
        """
        Compute lift coefficient.

        Args:
            alpha: Angle of attack (degrees)
            flap: Flap deflection (degrees)

        Returns:
            Lift coefficient CL
        """
        # Basic lift
        cl = self.CL_alpha_deg * (alpha - self.alpha_0)

        # Flap effect (simplified)
        if flap != 0:
            cl += 0.5 * flap / 10  # Approximate 0.05 CL per degree of flap

        # Stall limit
        cl_max = self.airfoil.cl_max * 0.9  # 3D reduction
        cl = np.clip(cl, -cl_max, cl_max)

        return cl

    def compute_cd(self, cl: float, alpha: float = 0.0) -> Tuple[float, DragComponents]:
        """
        Compute drag coefficient and components.

        Args:
            cl: Lift coefficient
            alpha: Angle of attack (degrees)

        Returns:
            Tuple of (total CD, DragComponents)
        """
        # Induced drag: CDi = CL² / (π * e * AR)
        cd_induced = cl**2 / (np.pi * self.e * self.AR)

        # Profile drag variation with CL (bucket drag polar)
        cl_minD = 0.4  # CL for minimum drag
        k_profile = 0.005  # Profile drag sensitivity
        cd_profile = k_profile * (cl - cl_minD)**2

        # Parasite drag
        cd_parasite = self.CD0

        # Total
        cd_total = cd_parasite + cd_induced + cd_profile

        components = DragComponents(
            parasite=cd_parasite,
            induced=cd_induced,
            profile=cd_profile,
            interference=0.0,
        )

        return cd_total, components

    def compute_forces(
        self,
        condition: FlightCondition
    ) -> AerodynamicState:
        """
        Compute all aerodynamic forces at flight condition.

        Args:
            condition: Flight condition

        Returns:
            Complete aerodynamic state
        """
        # Atmospheric properties
        rho = self.atm.density(condition.altitude)
        q = 0.5 * rho * condition.velocity**2  # Dynamic pressure

        # Coefficients
        cl = self.compute_cl(condition.alpha, condition.flap_deflection)
        cd, drag_components = self.compute_cd(cl, condition.alpha)

        # Moment coefficient (simplified)
        cm_ac = -0.05  # Airfoil moment about AC
        cm = cm_ac + cl * (self.mass.cg_position - 0.25)  # Transfer to CG

        # Forces
        lift = q * self.S * cl
        drag = q * self.S * cd
        moment = q * self.S * self.MAC * cm

        # Scale drag components to forces
        drag_components_n = DragComponents(
            parasite=q * self.S * drag_components.parasite,
            induced=q * self.S * drag_components.induced,
            profile=q * self.S * drag_components.profile,
            interference=q * self.S * drag_components.interference,
        )

        # Performance
        glide_ratio = cl / cd if cd > 0 else 0
        glide_angle = np.degrees(np.arctan(1 / glide_ratio)) if glide_ratio > 0 else 90
        sink_rate = condition.velocity * np.sin(np.radians(glide_angle))

        return AerodynamicState(
            condition=condition,
            lift=lift,
            drag=drag,
            drag_components=drag_components_n,
            moment=moment,
            cl=cl,
            cd=cd,
            cm=cm,
            glide_ratio=glide_ratio,
            sink_rate=sink_rate,
            glide_angle=glide_angle,
        )

    def find_trim_alpha(
        self,
        velocity: float,
        altitude: float,
        bank_angle: float = 0.0
    ) -> float:
        """
        Find angle of attack for trimmed (L = W) flight.

        Args:
            velocity: True airspeed (m/s)
            altitude: Altitude (m)
            bank_angle: Bank angle (degrees)

        Returns:
            Trim angle of attack (degrees)
        """
        rho = self.atm.density(altitude)
        q = 0.5 * rho * velocity**2

        # Required CL for level flight (accounting for bank)
        load_factor = 1 / np.cos(np.radians(bank_angle))
        cl_required = (self.mass.weight * load_factor) / (q * self.S)

        # Invert lift equation
        alpha_trim = cl_required / self.CL_alpha_deg + self.alpha_0

        return alpha_trim

    def compute_performance_envelope(
        self,
        altitude: float = 1000.0,
        v_min: float = 10.0,
        v_max: float = 50.0,
        num_points: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Compute performance envelope at altitude.

        Args:
            altitude: Flight altitude (m)
            v_min: Minimum velocity (m/s)
            v_max: Maximum velocity (m/s)
            num_points: Number of velocity points

        Returns:
            Dictionary with velocity, glide_ratio, sink_rate, cl, cd arrays
        """
        velocities = np.linspace(v_min, v_max, num_points)
        results = {
            "velocity": velocities,
            "glide_ratio": np.zeros(num_points),
            "sink_rate": np.zeros(num_points),
            "cl": np.zeros(num_points),
            "cd": np.zeros(num_points),
            "alpha": np.zeros(num_points),
        }

        for i, v in enumerate(velocities):
            try:
                alpha = self.find_trim_alpha(v, altitude)
                condition = FlightCondition(velocity=v, altitude=altitude, alpha=alpha)
                state = self.compute_forces(condition)

                results["glide_ratio"][i] = state.glide_ratio
                results["sink_rate"][i] = state.sink_rate
                results["cl"][i] = state.cl
                results["cd"][i] = state.cd
                results["alpha"][i] = alpha
            except Exception:
                results["glide_ratio"][i] = 0
                results["sink_rate"][i] = 0

        return results

    def best_glide_speed(self, altitude: float = 1000.0) -> Tuple[float, float]:
        """
        Find speed for maximum glide ratio.

        Args:
            altitude: Flight altitude (m)

        Returns:
            Tuple of (best_glide_speed, max_glide_ratio)
        """
        envelope = self.compute_performance_envelope(altitude)
        idx = np.argmax(envelope["glide_ratio"])
        return envelope["velocity"][idx], envelope["glide_ratio"][idx]

    def min_sink_speed(self, altitude: float = 1000.0) -> Tuple[float, float]:
        """
        Find speed for minimum sink rate.

        Args:
            altitude: Flight altitude (m)

        Returns:
            Tuple of (min_sink_speed, min_sink_rate)
        """
        envelope = self.compute_performance_envelope(altitude)
        # Filter out zero values
        valid = envelope["sink_rate"] > 0
        if not np.any(valid):
            return 0, 0
        idx = np.argmin(envelope["sink_rate"][valid])
        valid_indices = np.where(valid)[0]
        return envelope["velocity"][valid_indices[idx]], envelope["sink_rate"][valid_indices[idx]]

    def wing_loading(self) -> float:
        """Get wing loading (N/m²)."""
        return self.mass.weight / self.S

    def stall_speed(self, altitude: float = 0.0) -> float:
        """
        Compute stall speed at altitude.

        Args:
            altitude: Flight altitude (m)

        Returns:
            Stall speed (m/s)
        """
        rho = self.atm.density(altitude)
        cl_max = self.airfoil.cl_max * 0.9  # 3D reduction
        return np.sqrt(2 * self.mass.weight / (rho * self.S * cl_max))

    def soaring_polar_k(self) -> float:
        """
        Compute ArduPilot SOAR_POLAR_K parameter.

        K = 16 * mass / wing_area

        Returns:
            SOAR_POLAR_K value
        """
        return 16 * self.mass.total_mass / self.S

    def summary(self) -> Dict:
        """Get summary of glider properties and performance."""
        v_bg, ld_max = self.best_glide_speed()
        v_ms, sr_min = self.min_sink_speed()

        return {
            "geometry": {
                "wingspan": self.b,
                "wing_area": self.S,
                "aspect_ratio": self.AR,
                "mac": self.MAC,
                "taper_ratio": self.wing.taper_ratio,
            },
            "mass": {
                "total_mass": self.mass.total_mass,
                "weight": self.mass.weight,
                "wing_loading": self.wing_loading(),
            },
            "aerodynamics": {
                "oswald_efficiency": self.e,
                "CD0": self.CD0,
                "CL_alpha": self.CL_alpha_deg,
            },
            "performance": {
                "best_glide_speed": v_bg,
                "max_glide_ratio": ld_max,
                "min_sink_speed": v_ms,
                "min_sink_rate": sr_min,
                "stall_speed": self.stall_speed(),
            },
            "ardupilot": {
                "SOAR_POLAR_K": self.soaring_polar_k(),
            },
        }


def create_default_glider() -> GliderModel:
    """
    Create a default high-efficiency solar glider.

    Specifications:
    - 4m wingspan
    - ~15kg total mass
    - AR ~20 for high efficiency
    """
    from .wing_geometry import create_high_ar_glider_wing

    wing = create_high_ar_glider_wing(
        wingspan=4.0,
        aspect_ratio=20.0,
        taper_ratio=0.45,
    )

    mass = MassProperties(
        empty_mass=8.0,
        payload_mass=2.0,
        battery_mass=2.5,
        solar_cell_mass=1.5,
        cg_position=0.28,
    )

    return GliderModel(wing=wing, mass=mass)
