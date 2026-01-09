"""
Optimization Constraints.

Defines constraints that feasible designs must satisfy:
1. Wing loading limits (structural and performance)
2. Static stability margin
3. Total weight limit
4. Stall speed requirements
5. Structural feasibility

Constraints are formulated as g(x) <= 0 where feasible designs
have non-positive constraint values.
"""

from dataclasses import dataclass
from typing import Optional

from ..aerodynamics.glider_model import GliderModel


@dataclass
class ConstraintResult:
    """Results of constraint evaluation."""

    wing_loading_violation: float = 0.0  # (actual - max), negative if satisfied
    stability_violation: float = 0.0     # (min_required - actual), negative if satisfied
    weight_violation: float = 0.0        # (actual - max), negative if satisfied
    stall_speed_violation: float = 0.0   # (actual - max), negative if satisfied
    structural_violation: float = 0.0    # Combined structural feasibility

    is_feasible: bool = True

    @property
    def total_violation(self) -> float:
        """Sum of constraint violations (for penalty methods)."""
        return max(0, self.wing_loading_violation) + \
               max(0, self.stability_violation) + \
               max(0, self.weight_violation) + \
               max(0, self.stall_speed_violation) + \
               max(0, self.structural_violation)

    def __post_init__(self):
        """Determine overall feasibility."""
        self.is_feasible = self.total_violation == 0


def evaluate_constraints(
    glider: GliderModel,
    max_wing_loading: float = 100.0,  # N/m²
    min_stability_margin: float = 0.05,  # fraction of MAC
    max_weight: float = 20.0,  # kg
    max_stall_speed: float = 12.0,  # m/s
) -> ConstraintResult:
    """
    Evaluate all constraints for a glider design.

    Args:
        glider: GliderModel instance to evaluate
        max_wing_loading: Maximum allowable wing loading (N/m²)
        min_stability_margin: Minimum static margin (fraction of MAC)
        max_weight: Maximum total weight (kg)
        max_stall_speed: Maximum allowable stall speed (m/s)

    Returns:
        ConstraintResult with violation values
    """
    result = ConstraintResult()

    # Wing loading constraint
    wing_loading = glider.wing_loading()
    result.wing_loading_violation = wing_loading - max_wing_loading

    # Stability constraint
    # Static margin = (x_np - x_cg) / MAC
    # x_np typically at 25% MAC for neutral point
    # Require CG ahead of NP by min_stability_margin * MAC
    neutral_point = 0.25  # fraction of MAC
    stability_margin = neutral_point - glider.mass.cg_position
    result.stability_violation = min_stability_margin - stability_margin

    # Weight constraint
    result.weight_violation = glider.mass.total_mass - max_weight

    # Stall speed constraint
    stall_speed = glider.stall_speed()
    result.stall_speed_violation = stall_speed - max_stall_speed

    # Structural feasibility (simplified check)
    result.structural_violation = check_structural_feasibility(glider)

    # Update overall feasibility
    result.is_feasible = (
        result.wing_loading_violation <= 0 and
        result.stability_violation <= 0 and
        result.weight_violation <= 0 and
        result.stall_speed_violation <= 0 and
        result.structural_violation <= 0
    )

    return result


def check_structural_feasibility(glider: GliderModel) -> float:
    """
    Check structural feasibility of the design.

    Simplified check based on:
    - Aspect ratio limits (high AR = structural challenges)
    - Wing loading (affects spar sizing)
    - Taper ratio (affects load distribution)

    Args:
        glider: GliderModel to evaluate

    Returns:
        Constraint violation value (negative if feasible)
    """
    AR = glider.wing.aspect_ratio
    taper = glider.wing.taper_ratio
    wing_loading = glider.wing_loading()

    # Aspect ratio limit based on wing loading
    # Higher wing loading can support higher AR
    # Empirical relationship for composite structures
    ar_limit = 15 + 0.1 * wing_loading  # Approximate

    ar_violation = AR - ar_limit

    # Taper ratio check
    # Very low taper ratio concentrates load at root
    # Very high taper ratio reduces structural efficiency
    taper_violation = 0.0
    if taper < 0.25:
        taper_violation = 0.25 - taper
    elif taper > 0.7:
        taper_violation = taper - 0.7

    # Root bending moment check
    # M_root ~ W * b / 4 for elliptical loading
    # Spar stress ~ M / I, where I ~ c³
    # Simplified: check root chord is adequate for bending
    root_chord = glider.wing.root_chord
    min_root_chord = 0.1 + 0.02 * AR  # Approximate minimum

    chord_violation = min_root_chord - root_chord

    # Combined violation
    total_violation = max(0, ar_violation) + max(0, taper_violation) + max(0, chord_violation)

    return total_violation if total_violation > 0 else -0.01


def wing_loading_constraint(
    glider: GliderModel,
    max_loading: float = 100.0
) -> float:
    """
    Wing loading constraint.

    Wing loading W/S affects:
    - Stall speed (higher loading = faster stall)
    - Gust response (lower loading = more sensitive)
    - Structural weight

    Args:
        glider: GliderModel to evaluate
        max_loading: Maximum allowable wing loading (N/m²)

    Returns:
        Constraint value g(x) = W/S - max (feasible if <= 0)
    """
    return glider.wing_loading() - max_loading


def stability_constraint(
    glider: GliderModel,
    min_margin: float = 0.05
) -> float:
    """
    Static stability margin constraint.

    Ensures CG is ahead of neutral point by at least min_margin * MAC.

    Args:
        glider: GliderModel to evaluate
        min_margin: Minimum stability margin (fraction of MAC)

    Returns:
        Constraint value (feasible if <= 0)
    """
    # Simplified neutral point estimation
    # NP ~ 0.25 for conventional wing
    neutral_point = 0.25
    stability_margin = neutral_point - glider.mass.cg_position

    return min_margin - stability_margin


def weight_constraint(
    glider: GliderModel,
    max_weight: float = 20.0
) -> float:
    """
    Total weight constraint.

    Args:
        glider: GliderModel to evaluate
        max_weight: Maximum allowable weight (kg)

    Returns:
        Constraint value (feasible if <= 0)
    """
    return glider.mass.total_mass - max_weight


def stall_speed_constraint(
    glider: GliderModel,
    max_stall: float = 12.0,
    altitude: float = 0.0
) -> float:
    """
    Stall speed constraint.

    Lower stall speed improves low-speed handling and
    allows thermalling in weak lift.

    Args:
        glider: GliderModel to evaluate
        max_stall: Maximum allowable stall speed (m/s)
        altitude: Reference altitude (m)

    Returns:
        Constraint value (feasible if <= 0)
    """
    stall_speed = glider.stall_speed(altitude)
    return stall_speed - max_stall


def aspect_ratio_constraint(
    glider: GliderModel,
    max_ar: float = 25.0
) -> float:
    """
    Aspect ratio constraint for structural feasibility.

    Very high AR wings are structurally challenging
    and may flutter.

    Args:
        glider: GliderModel to evaluate
        max_ar: Maximum aspect ratio

    Returns:
        Constraint value (feasible if <= 0)
    """
    return glider.wing.aspect_ratio - max_ar


def reynolds_number_constraint(
    glider: GliderModel,
    min_reynolds: float = 100000,
    reference_velocity: float = 15.0,
    altitude: float = 1000.0
) -> float:
    """
    Reynolds number constraint for airfoil validity.

    Airfoil data may not be valid below certain Reynolds numbers.

    Args:
        glider: GliderModel to evaluate
        min_reynolds: Minimum Reynolds number
        reference_velocity: Reference flight velocity (m/s)
        altitude: Reference altitude (m)

    Returns:
        Constraint value (feasible if <= 0)
    """
    Re = glider.atm.reynolds_number(
        reference_velocity,
        glider.wing.mean_aerodynamic_chord,
        altitude
    )
    return min_reynolds - Re


def solar_power_constraint(
    glider: GliderModel,
    min_power: float = 50.0,  # Watts
    solar_efficiency: float = 0.22,
    solar_irradiance: float = 800.0,  # W/m² (accounting for angle, weather)
    packing_factor: float = 0.85
) -> float:
    """
    Minimum solar power generation constraint.

    Ensures enough solar area for minimum power requirements.

    Args:
        glider: GliderModel to evaluate
        min_power: Minimum required power (W)
        solar_efficiency: Solar cell efficiency
        solar_irradiance: Expected irradiance (W/m²)
        packing_factor: Cell packing factor

    Returns:
        Constraint value (feasible if <= 0)
    """
    solar_area = glider.wing.solar_panel_area()
    power_generated = solar_area * packing_factor * solar_irradiance * solar_efficiency

    return min_power - power_generated
