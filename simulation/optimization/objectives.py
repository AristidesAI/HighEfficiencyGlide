"""
Optimization Objective Functions.

Defines the objectives to optimize for the solar glider:
1. Maximize glide ratio (L/D)
2. Maximize solar panel area
3. Minimize sink rate

These are multi-objective optimization targets that form a Pareto front.
"""

from dataclasses import dataclass
from typing import Tuple

from ..aerodynamics.glider_model import GliderModel


@dataclass
class ObjectiveResult:
    """Results of objective function evaluation."""

    glide_ratio: float  # L/D ratio (maximize)
    solar_area: float   # Solar panel area in m² (maximize)
    min_sink_rate: float  # Minimum sink rate in m/s (minimize)

    # Derived metrics
    endurance_factor: float = 0.0  # Combined metric for endurance

    def __post_init__(self):
        """Compute derived metrics."""
        # Endurance factor combines L/D with solar area
        # Higher is better
        if self.min_sink_rate > 0:
            self.endurance_factor = self.glide_ratio * self.solar_area / self.min_sink_rate
        else:
            self.endurance_factor = 0.0


def evaluate_objectives(glider: GliderModel) -> ObjectiveResult:
    """
    Evaluate all objectives for a glider design.

    Args:
        glider: GliderModel instance to evaluate

    Returns:
        ObjectiveResult with computed objectives
    """
    # Get performance envelope
    envelope = glider.compute_performance_envelope()

    # Glide ratio objective
    glide_ratio = float(max(envelope["glide_ratio"]))

    # Minimum sink rate
    valid_sink = envelope["sink_rate"] > 0
    if valid_sink.any():
        min_sink_rate = float(min(envelope["sink_rate"][valid_sink]))
    else:
        min_sink_rate = float("inf")

    # Solar panel area
    solar_area = glider.wing.solar_panel_area()

    return ObjectiveResult(
        glide_ratio=glide_ratio,
        solar_area=solar_area,
        min_sink_rate=min_sink_rate,
    )


def glide_ratio_objective(glider: GliderModel) -> float:
    """
    Compute glide ratio objective.

    Glide ratio L/D = CL/CD determines how far the glider
    can travel per unit altitude lost.

    Higher is better.

    Args:
        glider: GliderModel to evaluate

    Returns:
        Maximum L/D ratio
    """
    _, max_ld = glider.best_glide_speed()
    return max_ld


def solar_area_objective(glider: GliderModel) -> float:
    """
    Compute solar panel area objective.

    Larger area = more power generation capacity.

    Args:
        glider: GliderModel to evaluate

    Returns:
        Available solar panel area in m²
    """
    return glider.wing.solar_panel_area()


def sink_rate_objective(glider: GliderModel) -> float:
    """
    Compute minimum sink rate objective.

    Lower sink rate = longer time aloft for given altitude.

    Args:
        glider: GliderModel to evaluate

    Returns:
        Minimum sink rate in m/s (positive value)
    """
    _, min_sink = glider.min_sink_speed()
    return min_sink


def energy_balance_objective(
    glider: GliderModel,
    solar_efficiency: float = 0.22,
    power_consumption: float = 15.0,  # Watts
    solar_irradiance: float = 1000.0,  # W/m²
) -> float:
    """
    Compute energy balance objective.

    Positive value means net energy gain (self-sustaining flight possible).

    Args:
        glider: GliderModel to evaluate
        solar_efficiency: Solar cell efficiency (0-1)
        power_consumption: Average power consumption in Watts
        solar_irradiance: Solar irradiance in W/m²

    Returns:
        Net power balance in Watts (positive = energy surplus)
    """
    solar_area = glider.wing.solar_panel_area()
    packing_factor = 0.85  # Area actually covered by cells

    power_generated = solar_area * packing_factor * solar_irradiance * solar_efficiency
    power_balance = power_generated - power_consumption

    return power_balance


def weighted_objective(
    glider: GliderModel,
    w_glide_ratio: float = 1.0,
    w_solar_area: float = 0.5,
    w_sink_rate: float = 0.3,
) -> float:
    """
    Compute weighted sum of normalized objectives.

    For single-objective optimization approaches.

    Args:
        glider: GliderModel to evaluate
        w_glide_ratio: Weight for glide ratio (maximize)
        w_solar_area: Weight for solar area (maximize)
        w_sink_rate: Weight for sink rate (minimize)

    Returns:
        Weighted objective value (higher is better)
    """
    # Reference values for normalization
    ref_glide_ratio = 40.0  # Target L/D
    ref_solar_area = 1.0    # m²
    ref_sink_rate = 0.5     # m/s

    # Get objectives
    result = evaluate_objectives(glider)

    # Normalize
    norm_ld = result.glide_ratio / ref_glide_ratio
    norm_solar = result.solar_area / ref_solar_area
    norm_sink = ref_sink_rate / result.min_sink_rate if result.min_sink_rate > 0 else 0

    # Weighted sum
    return (
        w_glide_ratio * norm_ld
        + w_solar_area * norm_solar
        + w_sink_rate * norm_sink
    )


def compute_pareto_dominance(
    obj1: ObjectiveResult,
    obj2: ObjectiveResult
) -> int:
    """
    Determine Pareto dominance between two solutions.

    Args:
        obj1: First objective result
        obj2: Second objective result

    Returns:
        1 if obj1 dominates obj2
        -1 if obj2 dominates obj1
        0 if neither dominates (non-dominated)
    """
    # Objectives: maximize L/D, maximize solar, minimize sink
    better1 = 0
    better2 = 0

    # Glide ratio (maximize)
    if obj1.glide_ratio > obj2.glide_ratio:
        better1 += 1
    elif obj1.glide_ratio < obj2.glide_ratio:
        better2 += 1

    # Solar area (maximize)
    if obj1.solar_area > obj2.solar_area:
        better1 += 1
    elif obj1.solar_area < obj2.solar_area:
        better2 += 1

    # Sink rate (minimize)
    if obj1.min_sink_rate < obj2.min_sink_rate:
        better1 += 1
    elif obj1.min_sink_rate > obj2.min_sink_rate:
        better2 += 1

    if better1 > 0 and better2 == 0:
        return 1  # obj1 dominates
    elif better2 > 0 and better1 == 0:
        return -1  # obj2 dominates
    else:
        return 0  # Non-dominated
