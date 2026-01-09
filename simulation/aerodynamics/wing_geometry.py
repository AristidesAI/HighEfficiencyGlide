"""
Parametric Wing Geometry Module.

Defines wing planform geometry with support for:
- Rectangular, tapered, and elliptical planforms
- Sweep and dihedral
- Twist distribution
- Multiple wing sections with different airfoils

Key parameters for glide ratio optimization:
- Aspect Ratio (AR): Higher AR reduces induced drag
- Taper Ratio (λ): Affects lift distribution efficiency
- Sweep: Affects stability and high-speed performance
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class PlanformType(Enum):
    """Wing planform shape types."""
    RECTANGULAR = "rectangular"
    TAPERED = "tapered"
    ELLIPTICAL = "elliptical"
    DOUBLE_TAPERED = "double_tapered"


@dataclass
class WingSection:
    """
    Definition of a wing section (panel).

    Used for wings with multiple taper ratios or airfoil transitions.
    """

    span_fraction_start: float  # Fraction of semispan where section starts
    span_fraction_end: float    # Fraction of semispan where section ends
    chord_start: float          # Chord at section start (m)
    chord_end: float            # Chord at section end (m)
    airfoil_name: str = "E387"  # Airfoil for this section
    twist_start: float = 0.0   # Twist at start (degrees, positive = washout)
    twist_end: float = 0.0     # Twist at end (degrees)


@dataclass
class WingGeometry:
    """
    Complete wing geometry definition.

    Supports both simple parametric wings and complex multi-section wings.
    """

    # Primary dimensions
    wingspan: float  # Total wingspan (m)
    root_chord: float  # Chord at wing root (m)
    taper_ratio: float = 1.0  # Tip chord / Root chord (0 < λ ≤ 1)

    # Angular parameters
    sweep_angle: float = 0.0  # Leading edge sweep (degrees)
    dihedral_angle: float = 0.0  # Dihedral angle (degrees)
    twist: float = 0.0  # Washout at tip (degrees, positive = nose down)

    # Planform type
    planform: PlanformType = PlanformType.TAPERED

    # Multi-section definition (optional, overrides simple parameters)
    sections: Optional[List[WingSection]] = None

    # Airfoil
    airfoil_name: str = "E387"

    # Control surfaces
    aileron_span_start: float = 0.6  # Fraction of semispan
    aileron_span_end: float = 0.95   # Fraction of semispan
    aileron_chord_fraction: float = 0.25  # Fraction of local chord

    # Flap configuration (optional)
    flap_span_start: float = 0.1
    flap_span_end: float = 0.6
    flap_chord_fraction: float = 0.30

    def __post_init__(self):
        """Validate geometry parameters."""
        if self.wingspan <= 0:
            raise ValueError("Wingspan must be positive")
        if self.root_chord <= 0:
            raise ValueError("Root chord must be positive")
        if not 0 < self.taper_ratio <= 1:
            raise ValueError("Taper ratio must be in range (0, 1]")

    @property
    def semispan(self) -> float:
        """Half wingspan (m)."""
        return self.wingspan / 2

    @property
    def tip_chord(self) -> float:
        """Chord at wing tip (m)."""
        return self.root_chord * self.taper_ratio

    @property
    def mean_aerodynamic_chord(self) -> float:
        """
        Mean Aerodynamic Chord (MAC).

        For trapezoidal wing: MAC = (2/3) * c_r * (1 + λ + λ²) / (1 + λ)
        """
        lam = self.taper_ratio
        return (2 / 3) * self.root_chord * (1 + lam + lam**2) / (1 + lam)

    @property
    def wing_area(self) -> float:
        """
        Reference wing area (m²).

        For trapezoidal wing: S = (b/2) * (c_r + c_t)
        """
        return self.semispan * (self.root_chord + self.tip_chord)

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio AR = b² / S."""
        return self.wingspan**2 / self.wing_area

    @property
    def mac_position(self) -> float:
        """
        Spanwise position of MAC from root (m).

        For trapezoidal wing: y_mac = (b/6) * (1 + 2λ) / (1 + λ)
        """
        lam = self.taper_ratio
        return (self.semispan / 3) * (1 + 2 * lam) / (1 + lam)

    def chord_at_span(self, y: float) -> float:
        """
        Get chord length at spanwise position.

        Args:
            y: Spanwise position from root (m), 0 ≤ y ≤ semispan

        Returns:
            Local chord length (m)
        """
        y = np.clip(y, 0, self.semispan)
        eta = y / self.semispan  # Normalized span position

        if self.planform == PlanformType.RECTANGULAR:
            return self.root_chord

        elif self.planform == PlanformType.ELLIPTICAL:
            return self.root_chord * np.sqrt(1 - eta**2)

        else:  # TAPERED or DOUBLE_TAPERED
            if self.sections is not None:
                # Multi-section wing
                for section in self.sections:
                    if section.span_fraction_start <= eta <= section.span_fraction_end:
                        local_eta = (eta - section.span_fraction_start) / (
                            section.span_fraction_end - section.span_fraction_start
                        )
                        return section.chord_start + local_eta * (
                            section.chord_end - section.chord_start
                        )
                # Default to last section's tip chord
                return self.sections[-1].chord_end
            else:
                # Simple linear taper
                return self.root_chord * (1 - eta * (1 - self.taper_ratio))

    def twist_at_span(self, y: float) -> float:
        """
        Get twist angle at spanwise position.

        Args:
            y: Spanwise position from root (m)

        Returns:
            Local twist angle (degrees), positive = washout
        """
        eta = np.clip(y / self.semispan, 0, 1)

        if self.sections is not None:
            for section in self.sections:
                if section.span_fraction_start <= eta <= section.span_fraction_end:
                    local_eta = (eta - section.span_fraction_start) / (
                        section.span_fraction_end - section.span_fraction_start
                    )
                    return section.twist_start + local_eta * (
                        section.twist_end - section.twist_start
                    )
            return self.sections[-1].twist_end

        # Linear twist distribution
        return self.twist * eta

    def leading_edge_x(self, y: float) -> float:
        """
        X-position of leading edge at spanwise location.

        Args:
            y: Spanwise position from root (m)

        Returns:
            X-position of LE (m), relative to root LE
        """
        return y * np.tan(np.radians(self.sweep_angle))

    def vertical_position(self, y: float) -> float:
        """
        Vertical (Z) position due to dihedral.

        Args:
            y: Spanwise position from root (m)

        Returns:
            Z-position (m), relative to root
        """
        return y * np.tan(np.radians(self.dihedral_angle))

    def get_panel_geometry(
        self,
        num_panels: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate panel geometry for VLM or other analyses.

        Args:
            num_panels: Number of spanwise panels per semispan

        Returns:
            Tuple of (y_positions, chords, twists, le_x_positions)
        """
        # Use cosine spacing for better tip resolution
        theta = np.linspace(0, np.pi / 2, num_panels + 1)
        y = self.semispan * np.sin(theta)

        chords = np.array([self.chord_at_span(yi) for yi in y])
        twists = np.array([self.twist_at_span(yi) for yi in y])
        le_x = np.array([self.leading_edge_x(yi) for yi in y])

        return y, chords, twists, le_x

    def solar_panel_area(
        self,
        span_start: float = 0.05,
        span_end: float = 0.90,
        chord_start: float = 0.10,
        chord_end: float = 0.70
    ) -> float:
        """
        Calculate available area for solar panels on upper surface.

        Args:
            span_start: Start of solar panel region (fraction of semispan)
            span_end: End of solar panel region (fraction of semispan)
            chord_start: Start of panel region (fraction of local chord from LE)
            chord_end: End of panel region (fraction of local chord from LE)

        Returns:
            Total solar panel area on both wings (m²)
        """
        # Numerical integration
        num_points = 50
        y = np.linspace(span_start * self.semispan, span_end * self.semispan, num_points)
        chords = np.array([self.chord_at_span(yi) for yi in y])
        panel_widths = chords * (chord_end - chord_start)

        # Trapezoidal integration
        area_semispan = np.trapz(panel_widths, y)

        # Both wings
        return 2 * area_semispan

    def to_dict(self) -> dict:
        """Export geometry to dictionary."""
        return {
            "wingspan": self.wingspan,
            "root_chord": self.root_chord,
            "taper_ratio": self.taper_ratio,
            "sweep_angle": self.sweep_angle,
            "dihedral_angle": self.dihedral_angle,
            "twist": self.twist,
            "planform": self.planform.value,
            "airfoil_name": self.airfoil_name,
            "wing_area": self.wing_area,
            "aspect_ratio": self.aspect_ratio,
            "mac": self.mean_aerodynamic_chord,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WingGeometry":
        """Create geometry from dictionary."""
        planform = PlanformType(data.get("planform", "tapered"))
        return cls(
            wingspan=data["wingspan"],
            root_chord=data["root_chord"],
            taper_ratio=data.get("taper_ratio", 1.0),
            sweep_angle=data.get("sweep_angle", 0.0),
            dihedral_angle=data.get("dihedral_angle", 0.0),
            twist=data.get("twist", 0.0),
            planform=planform,
            airfoil_name=data.get("airfoil_name", "E387"),
        )


def create_high_ar_glider_wing(
    wingspan: float = 4.0,
    aspect_ratio: float = 20.0,
    taper_ratio: float = 0.4,
) -> WingGeometry:
    """
    Create a high aspect ratio glider wing optimized for efficiency.

    Args:
        wingspan: Total wingspan (m)
        aspect_ratio: Target aspect ratio
        taper_ratio: Wing taper ratio

    Returns:
        WingGeometry configured for high L/D
    """
    # Calculate root chord from AR and wingspan
    wing_area = wingspan**2 / aspect_ratio
    root_chord = 2 * wing_area / (wingspan * (1 + taper_ratio))

    return WingGeometry(
        wingspan=wingspan,
        root_chord=root_chord,
        taper_ratio=taper_ratio,
        sweep_angle=0.0,  # No sweep for subsonic efficiency
        dihedral_angle=3.0,  # Moderate dihedral for stability
        twist=-3.0,  # Washout for stall characteristics
        planform=PlanformType.TAPERED,
        airfoil_name="HQW2512",  # High-performance sailplane airfoil
    )


def oswald_efficiency_factor(aspect_ratio: float, taper_ratio: float) -> float:
    """
    Estimate Oswald efficiency factor for induced drag.

    Based on empirical correlation for unswept tapered wings.

    Args:
        aspect_ratio: Wing aspect ratio
        taper_ratio: Wing taper ratio

    Returns:
        Oswald efficiency factor e (0 < e ≤ 1)
    """
    # Optimal taper ratio for elliptical loading is ~0.4
    # Efficiency decreases for rectangular (λ=1) or highly tapered wings
    taper_penalty = 0.95 - 0.1 * abs(taper_ratio - 0.4)

    # High AR wings have slightly lower e due to viscous effects
    ar_factor = 1 / (1 + 0.01 * aspect_ratio)

    # Base efficiency
    e = 0.98 * taper_penalty * (1 - 0.045 * aspect_ratio**0.68) * ar_factor

    return np.clip(e, 0.7, 0.98)
