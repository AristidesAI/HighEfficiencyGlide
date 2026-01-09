"""
Airfoil Aerodynamic Data.

Provides lift, drag, and moment coefficients for various airfoil profiles.
Includes NACA 4-digit and 5-digit series generation, as well as
tabulated data for high-performance glider airfoils.

Key airfoils for solar gliders:
- NACA 63-412: Good lift, moderate thickness
- Eppler E387: Popular for sailplanes
- Selig S1223: High-lift, low Reynolds number
- HQ/W 2.5/12: High-performance sailplane
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from scipy.interpolate import interp1d


@dataclass
class AirfoilCoefficients:
    """Aerodynamic coefficients at a specific angle of attack."""

    alpha: float  # Angle of attack (degrees)
    cl: float  # Lift coefficient
    cd: float  # Drag coefficient
    cm: float  # Pitching moment coefficient (about quarter chord)


@dataclass
class AirfoilPolar:
    """
    Complete aerodynamic polar for an airfoil.

    Contains coefficient data across angle of attack range
    and provides interpolation methods.
    """

    name: str
    alpha: np.ndarray  # Angles of attack (degrees)
    cl: np.ndarray  # Lift coefficients
    cd: np.ndarray  # Drag coefficients
    cm: np.ndarray  # Moment coefficients
    reynolds: float  # Reynolds number for this polar

    # Interpolation functions (created lazily)
    _cl_interp: Optional[Callable] = field(default=None, repr=False)
    _cd_interp: Optional[Callable] = field(default=None, repr=False)
    _cm_interp: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self):
        """Create interpolation functions."""
        self._cl_interp = interp1d(
            self.alpha, self.cl, kind='cubic', fill_value='extrapolate'
        )
        self._cd_interp = interp1d(
            self.alpha, self.cd, kind='cubic', fill_value='extrapolate'
        )
        self._cm_interp = interp1d(
            self.alpha, self.cm, kind='cubic', fill_value='extrapolate'
        )

    def get_cl(self, alpha: float) -> float:
        """Get lift coefficient at angle of attack."""
        return float(self._cl_interp(alpha))

    def get_cd(self, alpha: float) -> float:
        """Get drag coefficient at angle of attack."""
        return float(self._cd_interp(alpha))

    def get_cm(self, alpha: float) -> float:
        """Get moment coefficient at angle of attack."""
        return float(self._cm_interp(alpha))

    def get_coefficients(self, alpha: float) -> AirfoilCoefficients:
        """Get all coefficients at angle of attack."""
        return AirfoilCoefficients(
            alpha=alpha,
            cl=self.get_cl(alpha),
            cd=self.get_cd(alpha),
            cm=self.get_cm(alpha),
        )

    @property
    def cl_max(self) -> float:
        """Maximum lift coefficient."""
        return float(np.max(self.cl))

    @property
    def alpha_stall(self) -> float:
        """Stall angle of attack (degrees)."""
        return float(self.alpha[np.argmax(self.cl)])

    @property
    def cl_alpha(self) -> float:
        """Lift curve slope (per degree) in linear region."""
        # Use linear region (-5 to 8 degrees typically)
        mask = (self.alpha >= -5) & (self.alpha <= 8)
        if np.sum(mask) < 2:
            mask = np.ones_like(self.alpha, dtype=bool)
        coeffs = np.polyfit(self.alpha[mask], self.cl[mask], 1)
        return float(coeffs[0])

    @property
    def alpha_zero_lift(self) -> float:
        """Zero-lift angle of attack (degrees)."""
        # Find where Cl crosses zero
        idx = np.argmin(np.abs(self.cl))
        return float(self.alpha[idx])

    @property
    def max_lift_to_drag(self) -> Tuple[float, float]:
        """
        Maximum lift-to-drag ratio and corresponding angle of attack.

        Returns:
            Tuple of (L/D_max, alpha at L/D_max)
        """
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            ld = np.where(self.cd > 0, self.cl / self.cd, 0)
        idx = np.argmax(ld)
        return float(ld[idx]), float(self.alpha[idx])

    @property
    def cd_min(self) -> float:
        """Minimum drag coefficient."""
        return float(np.min(self.cd))


class AirfoilDatabase:
    """
    Database of airfoil polars.

    Provides access to predefined airfoils and methods for
    generating NACA airfoil characteristics.
    """

    def __init__(self):
        self._airfoils: Dict[str, AirfoilPolar] = {}
        self._load_default_airfoils()

    def _load_default_airfoils(self):
        """Load commonly used glider airfoils."""

        # NACA 2412 - General purpose, widely used
        # Data approximate for Re = 1e6
        alpha_2412 = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16])
        cl_2412 = np.array([-0.6, -0.4, -0.15, 0.1, 0.35, 0.55, 0.75, 0.95, 1.1, 1.2, 1.25, 1.15, 1.0])
        cd_2412 = np.array([0.018, 0.012, 0.009, 0.008, 0.008, 0.009, 0.010, 0.012, 0.015, 0.020, 0.030, 0.050, 0.080])
        cm_2412 = np.array([-0.04, -0.04, -0.045, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05])

        self._airfoils["NACA2412"] = AirfoilPolar(
            name="NACA2412",
            alpha=alpha_2412,
            cl=cl_2412,
            cd=cd_2412,
            cm=cm_2412,
            reynolds=1e6,
        )

        # Eppler E387 - Popular sailplane airfoil
        # Data approximate for Re = 300,000
        alpha_e387 = np.array([-6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16])
        cl_e387 = np.array([-0.2, 0.0, 0.25, 0.5, 0.75, 1.0, 1.2, 1.35, 1.45, 1.45, 1.35, 1.15])
        cd_e387 = np.array([0.015, 0.010, 0.008, 0.007, 0.008, 0.010, 0.013, 0.018, 0.025, 0.040, 0.065, 0.100])
        cm_e387 = np.array([-0.08, -0.08, -0.08, -0.08, -0.085, -0.09, -0.095, -0.10, -0.10, -0.10, -0.10, -0.10])

        self._airfoils["E387"] = AirfoilPolar(
            name="E387",
            alpha=alpha_e387,
            cl=cl_e387,
            cd=cd_e387,
            cm=cm_e387,
            reynolds=3e5,
        )

        # Selig S1223 - High lift, low Reynolds number
        # Data approximate for Re = 200,000
        alpha_s1223 = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
        cl_s1223 = np.array([0.4, 0.7, 1.0, 1.3, 1.55, 1.8, 2.0, 2.15, 2.2, 2.1, 1.9, 1.6])
        cd_s1223 = np.array([0.025, 0.020, 0.018, 0.018, 0.020, 0.025, 0.032, 0.045, 0.065, 0.095, 0.140, 0.200])
        cm_s1223 = np.array([-0.25, -0.25, -0.26, -0.27, -0.28, -0.29, -0.30, -0.30, -0.30, -0.30, -0.30, -0.30])

        self._airfoils["S1223"] = AirfoilPolar(
            name="S1223",
            alpha=alpha_s1223,
            cl=cl_s1223,
            cd=cd_s1223,
            cm=cm_s1223,
            reynolds=2e5,
        )

        # SD7037 - Popular RC sailplane airfoil
        # Data approximate for Re = 200,000
        alpha_sd7037 = np.array([-6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14])
        cl_sd7037 = np.array([-0.15, 0.1, 0.35, 0.6, 0.85, 1.05, 1.2, 1.3, 1.35, 1.3, 1.15])
        cd_sd7037 = np.array([0.014, 0.010, 0.008, 0.007, 0.008, 0.010, 0.013, 0.018, 0.028, 0.045, 0.075])
        cm_sd7037 = np.array([-0.10, -0.10, -0.10, -0.10, -0.10, -0.105, -0.11, -0.11, -0.11, -0.11, -0.11])

        self._airfoils["SD7037"] = AirfoilPolar(
            name="SD7037",
            alpha=alpha_sd7037,
            cl=cl_sd7037,
            cd=cd_sd7037,
            cm=cm_sd7037,
            reynolds=2e5,
        )

        # HQ/W 2.5/12 - High-performance competition sailplane
        # Data approximate for Re = 1e6
        alpha_hqw = np.array([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14])
        cl_hqw = np.array([-0.1, 0.15, 0.4, 0.65, 0.9, 1.1, 1.25, 1.35, 1.4, 1.35])
        cd_hqw = np.array([0.008, 0.006, 0.0055, 0.006, 0.007, 0.009, 0.012, 0.017, 0.025, 0.040])
        cm_hqw = np.array([-0.06, -0.06, -0.06, -0.065, -0.07, -0.075, -0.08, -0.08, -0.08, -0.08])

        self._airfoils["HQW2512"] = AirfoilPolar(
            name="HQW2512",
            alpha=alpha_hqw,
            cl=cl_hqw,
            cd=cd_hqw,
            cm=cm_hqw,
            reynolds=1e6,
        )

    def get_airfoil(self, name: str) -> Optional[AirfoilPolar]:
        """Get airfoil polar by name."""
        return self._airfoils.get(name)

    def list_airfoils(self) -> List[str]:
        """List available airfoil names."""
        return list(self._airfoils.keys())

    def add_airfoil(self, polar: AirfoilPolar):
        """Add a custom airfoil to the database."""
        self._airfoils[polar.name] = polar


def generate_naca_4digit_geometry(
    designation: str,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate NACA 4-digit airfoil coordinates.

    Args:
        designation: 4-digit NACA designation (e.g., "2412")
        num_points: Number of points per surface

    Returns:
        Tuple of (x, y_upper, y_lower) coordinates normalized to chord
    """
    if len(designation) != 4:
        raise ValueError("NACA designation must be 4 digits")

    m = int(designation[0]) / 100  # Maximum camber
    p = int(designation[1]) / 10   # Location of maximum camber
    t = int(designation[2:4]) / 100  # Thickness

    # Generate x coordinates (cosine spacing for better resolution at LE/TE)
    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))

    # Thickness distribution
    yt = 5 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1015 * x**4
    )

    # Camber line
    if p == 0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(
            x < p,
            m / p**2 * (2 * p * x - x**2),
            m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2)
        )
        dyc_dx = np.where(
            x < p,
            2 * m / p**2 * (p - x),
            2 * m / (1 - p)**2 * (p - x)
        )

    # Rotate thickness by camber slope
    theta = np.arctan(dyc_dx)

    y_upper = yc + yt * np.cos(theta)
    y_lower = yc - yt * np.cos(theta)

    return x, y_upper, y_lower


def estimate_naca_4digit_coefficients(
    designation: str,
    alpha: np.ndarray,
    reynolds: float = 1e6
) -> AirfoilPolar:
    """
    Estimate aerodynamic coefficients for NACA 4-digit airfoil.

    Uses thin airfoil theory with empirical corrections.

    Args:
        designation: 4-digit NACA designation
        alpha: Array of angles of attack (degrees)
        reynolds: Reynolds number

    Returns:
        AirfoilPolar with estimated coefficients
    """
    m = int(designation[0]) / 100
    p = int(designation[1]) / 10
    t = int(designation[2:4]) / 100

    # Zero-lift angle (thin airfoil theory)
    if p > 0:
        alpha_0 = -np.degrees(2 * m * (1 - 2 * p))
    else:
        alpha_0 = 0

    # Lift curve slope (corrected for finite thickness)
    cl_alpha = 2 * np.pi * (1 + 0.77 * t)  # per radian
    cl_alpha_deg = cl_alpha * np.pi / 180

    # Lift coefficient
    alpha_eff = alpha - alpha_0
    cl = cl_alpha_deg * alpha_eff

    # Apply stall model
    cl_max = 1.4 + 0.7 * m  # Approximate max Cl
    alpha_stall = alpha_0 + cl_max / cl_alpha_deg

    # Smooth stall transition
    stall_factor = 1 / (1 + np.exp(2 * (alpha - alpha_stall)))
    cl = cl * stall_factor + cl_max * np.exp(-0.5 * ((alpha - alpha_stall) / 5)**2) * (1 - stall_factor)

    # Drag coefficient
    # Minimum drag (skin friction approximation)
    cf = 0.074 / reynolds**0.2  # Turbulent flat plate
    cd_min = 2 * cf * (1 + 2 * t)

    # Induced drag contribution (2D approximation)
    cd_induced = cl**2 / (np.pi * 10)  # Approximate e*AR effect

    # Profile drag increase at high alpha
    cd_pressure = 0.01 * (alpha - alpha_0)**2 / 100

    cd = cd_min + cd_induced + cd_pressure

    # Post-stall drag increase
    cd = cd + 0.1 * np.maximum(0, alpha - alpha_stall)**2

    # Moment coefficient (thin airfoil theory)
    if p > 0:
        cm_ac = -np.pi / 4 * cl_alpha_deg * (alpha_0)  # Simplified
        cm = np.full_like(alpha, -0.05 - 0.1 * m)
    else:
        cm = np.full_like(alpha, -0.01)

    return AirfoilPolar(
        name=f"NACA{designation}",
        alpha=alpha,
        cl=cl,
        cd=cd,
        cm=cm,
        reynolds=reynolds,
    )


# Module-level singleton database
_database = AirfoilDatabase()


def get_airfoil_database() -> AirfoilDatabase:
    """Get the singleton airfoil database instance."""
    return _database
