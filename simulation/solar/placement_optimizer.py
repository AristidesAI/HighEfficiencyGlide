"""
Solar Cell Placement Optimizer.

Determines optimal placement of solar cells on wing surface
considering:
- Available wing area (excluding control surfaces)
- Cell orientation and tilt
- Shading from fuselage/tail
- Weight distribution effects on CG
- Structural considerations
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

from ..aerodynamics.wing_geometry import WingGeometry


class PlacementStrategy(Enum):
    """Solar cell placement strategies."""
    UNIFORM = "uniform"           # Even distribution
    ROOT_WEIGHTED = "root"        # More cells near root (stability)
    TIP_WEIGHTED = "tip"          # More cells near tip (area)
    OPTIMAL_CG = "optimal_cg"     # Optimize for CG location


@dataclass
class CellPlacement:
    """Definition of a single solar cell placement."""

    span_position: float  # Fraction of semispan
    chord_position: float  # Fraction of local chord (from LE)
    area: float  # Cell area (m²)
    tilt_angle: float = 0.0  # Tilt relative to wing surface (degrees)
    efficiency_factor: float = 1.0  # Local efficiency factor (shading, etc.)


@dataclass
class PlacementResult:
    """Results of placement optimization."""

    placements: List[CellPlacement]
    total_area: float  # m²
    total_cells: int
    estimated_power: float  # Watts at 1000 W/m² irradiance
    cg_shift: float  # CG shift due to solar cells (m)
    mass: float  # Total solar cell mass (kg)


class SolarPlacementOptimizer:
    """
    Optimizes solar cell placement on glider wing.

    Considers wing geometry, control surface locations,
    structural limits, and CG effects.
    """

    def __init__(
        self,
        wing: WingGeometry,
        cell_size: Tuple[float, float] = (0.156, 0.10),  # 156mm x 100mm
        cell_efficiency: float = 0.22,
        cell_mass_per_area: float = 0.8,  # kg/m²
    ):
        """
        Initialize placement optimizer.

        Args:
            wing: Wing geometry
            cell_size: Cell dimensions (span, chord) in meters
            cell_efficiency: Cell efficiency at STC
            cell_mass_per_area: Cell mass per unit area
        """
        self.wing = wing
        self.cell_width = cell_size[0]
        self.cell_length = cell_size[1]
        self.cell_area = self.cell_width * self.cell_length
        self.efficiency = cell_efficiency
        self.mass_per_area = cell_mass_per_area

        # Exclusion zones (fractions)
        self.exclude_le = 0.05  # Leading edge (structural)
        self.exclude_te = 0.25  # Trailing edge (control surfaces)
        self.exclude_root = 0.05  # Root (fuselage junction)
        self.exclude_tip = 0.05  # Tip (vortex effects)

    def available_chord_range(self, span_fraction: float) -> Tuple[float, float]:
        """
        Get available chord range for cells at span position.

        Args:
            span_fraction: Spanwise position (0-1)

        Returns:
            Tuple of (start, end) chord fractions
        """
        # Check for control surface overlap
        in_aileron = (
            self.wing.aileron_span_start <= span_fraction <= self.wing.aileron_span_end
        )
        in_flap = (
            self.wing.flap_span_start <= span_fraction <= self.wing.flap_span_end
        )

        chord_start = self.exclude_le
        chord_end = 1.0 - self.exclude_te

        if in_aileron:
            chord_end = 1.0 - self.wing.aileron_chord_fraction
        if in_flap:
            chord_end = min(chord_end, 1.0 - self.wing.flap_chord_fraction)

        return chord_start, chord_end

    def compute_placements(
        self,
        strategy: PlacementStrategy = PlacementStrategy.UNIFORM,
        packing_factor: float = 0.85,
    ) -> PlacementResult:
        """
        Compute optimal cell placements.

        Args:
            strategy: Placement strategy
            packing_factor: Cell packing density (0-1)

        Returns:
            PlacementResult with cell positions and metrics
        """
        placements = []
        total_area = 0.0
        total_mass = 0.0
        cg_moment = 0.0

        # Span range for placement
        span_start = self.exclude_root
        span_end = 1.0 - self.exclude_tip

        # Number of spanwise rows
        available_span = (span_end - span_start) * self.wing.semispan
        num_rows = int(available_span / self.cell_width * packing_factor)

        for row in range(num_rows):
            span_frac = span_start + (row + 0.5) / num_rows * (span_end - span_start)
            span_pos = span_frac * self.wing.semispan

            # Get local chord
            local_chord = self.wing.chord_at_span(span_pos)

            # Available chord range
            chord_start, chord_end = self.available_chord_range(span_frac)
            available_chord = (chord_end - chord_start) * local_chord

            # Number of cells in this row
            num_cells = int(available_chord / self.cell_length * packing_factor)

            # Weight factor based on strategy
            if strategy == PlacementStrategy.ROOT_WEIGHTED:
                weight = 1.5 - span_frac
            elif strategy == PlacementStrategy.TIP_WEIGHTED:
                weight = 0.5 + span_frac
            elif strategy == PlacementStrategy.OPTIMAL_CG:
                # More cells inboard to minimize CG shift
                weight = max(0.3, 1.2 - span_frac)
            else:
                weight = 1.0

            # Efficiency factor (decreases slightly toward tip due to turbulence)
            eff_factor = 1.0 - 0.05 * span_frac

            for col in range(num_cells):
                chord_frac = chord_start + (col + 0.5) / num_cells * (chord_end - chord_start)

                placement = CellPlacement(
                    span_position=span_frac,
                    chord_position=chord_frac,
                    area=self.cell_area * weight,
                    tilt_angle=0.0,
                    efficiency_factor=eff_factor,
                )
                placements.append(placement)

                # Accumulate totals
                cell_mass = self.cell_area * weight * self.mass_per_area
                total_area += self.cell_area * weight
                total_mass += cell_mass
                cg_moment += cell_mass * span_pos

        # Calculate CG shift
        cg_shift = cg_moment / total_mass if total_mass > 0 else 0.0

        # Estimated power (both wings)
        estimated_power = total_area * 2 * 1000 * self.efficiency  # At 1000 W/m²

        return PlacementResult(
            placements=placements,
            total_area=total_area * 2,  # Both wings
            total_cells=len(placements) * 2,
            estimated_power=estimated_power,
            cg_shift=cg_shift,
            mass=total_mass * 2,
        )

    def compute_power_vs_sun_angle(
        self,
        placements: List[CellPlacement],
        sun_elevation: float,
        sun_azimuth: float,
        aircraft_heading: float = 0.0,
        bank_angle: float = 0.0,
        pitch_angle: float = 0.0,
    ) -> float:
        """
        Compute total power output for given sun position and aircraft attitude.

        Args:
            placements: List of cell placements
            sun_elevation: Sun elevation angle (degrees, 0 = horizon)
            sun_azimuth: Sun azimuth (degrees from north)
            aircraft_heading: Aircraft heading (degrees from north)
            bank_angle: Aircraft bank angle (degrees)
            pitch_angle: Aircraft pitch angle (degrees)

        Returns:
            Total power output (Watts)
        """
        if sun_elevation <= 0:
            return 0.0

        # Solar irradiance (simplified model)
        # Reduced by atmosphere at low elevations
        air_mass = 1 / np.sin(np.radians(max(1, sun_elevation)))
        irradiance = 1000 * 0.7 ** (air_mass - 1)

        total_power = 0.0

        for placement in placements:
            # Compute local surface normal considering aircraft attitude
            # Simplified: assume wing is horizontal when aircraft is level
            span_pos = placement.span_position * self.wing.semispan

            # Local dihedral effect
            local_dihedral = self.wing.dihedral_angle + bank_angle

            # Incidence angle calculation (simplified)
            # Normal vector of wing surface
            wing_tilt = np.radians(local_dihedral + placement.tilt_angle)

            # Sun vector (simplified spherical coordinates)
            sun_elev_rad = np.radians(sun_elevation)
            relative_azimuth = np.radians(sun_azimuth - aircraft_heading)

            # Cosine of incidence angle
            cos_inc = (
                np.sin(sun_elev_rad) * np.cos(wing_tilt) +
                np.cos(sun_elev_rad) * np.sin(wing_tilt) * np.cos(relative_azimuth)
            )
            cos_inc = max(0, cos_inc)

            # Power from this cell
            cell_power = (
                placement.area *
                irradiance *
                self.efficiency *
                placement.efficiency_factor *
                cos_inc
            )
            total_power += cell_power

        # Both wings
        return total_power * 2

    def optimize_for_mission(
        self,
        mission_latitude: float,
        mission_time: str = "noon",
        heading_distribution: Optional[np.ndarray] = None,
    ) -> PlacementResult:
        """
        Optimize placement for specific mission profile.

        Args:
            mission_latitude: Mission latitude (degrees)
            mission_time: Time of day ("morning", "noon", "afternoon")
            heading_distribution: Probability distribution of headings

        Returns:
            Optimized PlacementResult
        """
        # Determine dominant sun position
        if mission_time == "morning":
            sun_azimuth = 90  # East
            sun_elevation = 30 + mission_latitude * 0.3
        elif mission_time == "afternoon":
            sun_azimuth = 270  # West
            sun_elevation = 30 + mission_latitude * 0.3
        else:  # noon
            sun_azimuth = 180  # South (northern hemisphere)
            sun_elevation = 90 - abs(mission_latitude) + 23.5  # Summer approximation

        # For uniform heading distribution, use UNIFORM strategy
        # For specific headings, could tilt cells
        return self.compute_placements(PlacementStrategy.OPTIMAL_CG)


def compute_shading_factor(
    wing: WingGeometry,
    span_position: float,
    fuselage_width: float = 0.15,
    tail_shadow_angle: float = 30.0,
) -> float:
    """
    Compute shading factor at wing position.

    Args:
        wing: Wing geometry
        span_position: Spanwise position (fraction)
        fuselage_width: Fuselage width at wing junction (m)
        tail_shadow_angle: Sun angle below which tail shadows wing (degrees)

    Returns:
        Shading factor (0 = fully shaded, 1 = no shading)
    """
    span_dist = span_position * wing.semispan

    # Fuselage shading (affects root area)
    fuselage_factor = min(1.0, 2 * span_dist / fuselage_width)

    # Simplified model: full sunlight beyond fuselage
    return fuselage_factor
