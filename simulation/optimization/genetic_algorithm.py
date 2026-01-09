"""
Multi-Objective Genetic Algorithm for Wing Optimization.

Implements NSGA-II style optimization to find Pareto-optimal
wing designs that maximize:
- Glide ratio (L/D)
- Solar panel area

Subject to constraints on:
- Stability margin
- Wing loading
- Structural feasibility
- Total weight
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Callable
import random
from concurrent.futures import ProcessPoolExecutor
import sys

from ..aerodynamics.wing_geometry import WingGeometry
from ..aerodynamics.glider_model import GliderModel, MassProperties
from .objectives import evaluate_objectives, ObjectiveResult
from .constraints import evaluate_constraints, ConstraintResult


@dataclass
class Individual:
    """Single design in the population."""

    # Design variables
    wingspan: float
    aspect_ratio: float
    taper_ratio: float
    twist: float

    # Computed values (set after evaluation)
    objectives: Optional[ObjectiveResult] = None
    constraints: Optional[ConstraintResult] = None
    fitness: float = 0.0

    # NSGA-II attributes
    rank: int = 0
    crowding_distance: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.wingspan, self.aspect_ratio, self.taper_ratio, self.twist])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Individual":
        """Create from numpy array."""
        return cls(
            wingspan=arr[0],
            aspect_ratio=arr[1],
            taper_ratio=arr[2],
            twist=arr[3],
        )

    def create_wing(self) -> WingGeometry:
        """Create wing geometry from design variables."""
        # Calculate root chord from AR and wingspan
        wing_area = self.wingspan**2 / self.aspect_ratio
        root_chord = 2 * wing_area / (self.wingspan * (1 + self.taper_ratio))

        return WingGeometry(
            wingspan=self.wingspan,
            root_chord=root_chord,
            taper_ratio=self.taper_ratio,
            twist=self.twist,
            airfoil_name="HQW2512",
        )

    @property
    def is_feasible(self) -> bool:
        """Check if design satisfies all constraints."""
        if self.constraints is None:
            return False
        return self.constraints.is_feasible


@dataclass
class OptimizationConfig:
    """Configuration for genetic algorithm."""

    # Variable bounds [min, max]
    wingspan_bounds: Tuple[float, float] = (3.0, 5.0)
    aspect_ratio_bounds: Tuple[float, float] = (15.0, 25.0)
    taper_ratio_bounds: Tuple[float, float] = (0.3, 0.6)
    twist_bounds: Tuple[float, float] = (-5.0, 0.0)

    # Algorithm parameters
    population_size: int = 100
    generations: int = 50
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    mutation_scale: float = 0.1

    # Mass properties (fixed for optimization)
    empty_mass: float = 8.0
    payload_mass: float = 2.0
    battery_mass: float = 2.5

    # Constraint limits
    max_wing_loading: float = 100.0  # N/m²
    min_stability_margin: float = 0.05  # fraction of MAC
    max_weight: float = 20.0  # kg
    min_stall_speed: float = 8.0  # m/s


class NSGA2Optimizer:
    """
    NSGA-II Multi-Objective Optimizer.

    Non-dominated Sorting Genetic Algorithm II for finding
    Pareto-optimal wing designs.
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.population: List[Individual] = []
        self.pareto_front: List[Individual] = []
        self.history: List[Dict] = []

    def initialize_population(self) -> List[Individual]:
        """Create initial random population."""
        population = []
        for _ in range(self.config.population_size):
            ind = Individual(
                wingspan=random.uniform(*self.config.wingspan_bounds),
                aspect_ratio=random.uniform(*self.config.aspect_ratio_bounds),
                taper_ratio=random.uniform(*self.config.taper_ratio_bounds),
                twist=random.uniform(*self.config.twist_bounds),
            )
            population.append(ind)
        return population

    def evaluate_individual(self, ind: Individual) -> Individual:
        """Evaluate objectives and constraints for one individual."""
        try:
            wing = ind.create_wing()

            # Estimate solar cell mass from wing area
            solar_area = wing.solar_panel_area()
            solar_mass = solar_area * 0.8  # kg/m²

            mass = MassProperties(
                empty_mass=self.config.empty_mass,
                payload_mass=self.config.payload_mass,
                battery_mass=self.config.battery_mass,
                solar_cell_mass=solar_mass,
            )

            glider = GliderModel(wing=wing, mass=mass)

            # Evaluate objectives
            ind.objectives = evaluate_objectives(glider)

            # Evaluate constraints
            ind.constraints = evaluate_constraints(
                glider,
                max_wing_loading=self.config.max_wing_loading,
                min_stability_margin=self.config.min_stability_margin,
                max_weight=self.config.max_weight,
            )

        except Exception as e:
            # Invalid design
            ind.objectives = ObjectiveResult(
                glide_ratio=0.0,
                solar_area=0.0,
                min_sink_rate=float("inf"),
            )
            ind.constraints = ConstraintResult(
                wing_loading_violation=1000.0,
                stability_violation=1000.0,
                weight_violation=1000.0,
                is_feasible=False,
            )

        return ind

    def evaluate_population(self, population: List[Individual]) -> List[Individual]:
        """Evaluate all individuals in population."""
        return [self.evaluate_individual(ind) for ind in population]

    def dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """Check if ind1 dominates ind2 (better in all objectives)."""
        if ind1.objectives is None or ind2.objectives is None:
            return False

        # Objectives to maximize (negate for minimization)
        obj1 = [-ind1.objectives.glide_ratio, -ind1.objectives.solar_area]
        obj2 = [-ind2.objectives.glide_ratio, -ind2.objectives.solar_area]

        # Add constraint penalty
        if not ind1.is_feasible:
            pen1 = ind1.constraints.total_violation if ind1.constraints else 1000
            obj1 = [o + pen1 for o in obj1]
        if not ind2.is_feasible:
            pen2 = ind2.constraints.total_violation if ind2.constraints else 1000
            obj2 = [o + pen2 for o in obj2]

        better_in_one = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:  # Worse (minimizing)
                return False
            if o1 < o2:
                better_in_one = True

        return better_in_one

    def non_dominated_sort(self, population: List[Individual]) -> List[List[Individual]]:
        """Sort population into non-dominated fronts."""
        fronts: List[List[Individual]] = [[]]

        # For each individual, find domination relationships
        domination_count = {id(ind): 0 for ind in population}
        dominated_set = {id(ind): [] for ind in population}

        for p in population:
            for q in population:
                if p is q:
                    continue
                if self.dominates(p, q):
                    dominated_set[id(p)].append(q)
                elif self.dominates(q, p):
                    domination_count[id(p)] += 1

            if domination_count[id(p)] == 0:
                p.rank = 0
                fronts[0].append(p)

        # Build subsequent fronts
        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in dominated_set[id(p)]:
                    domination_count[id(q)] -= 1
                    if domination_count[id(q)] == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def crowding_distance(self, front: List[Individual]):
        """Assign crowding distance to individuals in a front."""
        n = len(front)
        if n == 0:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        # For each objective
        objectives = ["glide_ratio", "solar_area"]
        for obj_name in objectives:
            # Sort by this objective
            front.sort(key=lambda x: getattr(x.objectives, obj_name) if x.objectives else 0)

            # Boundary points get infinite distance
            front[0].crowding_distance = float("inf")
            front[-1].crowding_distance = float("inf")

            # Calculate range
            obj_min = getattr(front[0].objectives, obj_name) if front[0].objectives else 0
            obj_max = getattr(front[-1].objectives, obj_name) if front[-1].objectives else 0
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # Assign distances
            for i in range(1, n - 1):
                prev_obj = getattr(front[i - 1].objectives, obj_name) if front[i - 1].objectives else 0
                next_obj = getattr(front[i + 1].objectives, obj_name) if front[i + 1].objectives else 0
                front[i].crowding_distance += (next_obj - prev_obj) / obj_range

    def tournament_select(self, population: List[Individual], k: int = 2) -> Individual:
        """Binary tournament selection."""
        contestants = random.sample(population, k)

        # Compare by rank first, then crowding distance
        best = contestants[0]
        for ind in contestants[1:]:
            if ind.rank < best.rank:
                best = ind
            elif ind.rank == best.rank and ind.crowding_distance > best.crowding_distance:
                best = ind

        return best

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """SBX crossover operator."""
        if random.random() > self.config.crossover_prob:
            return parent1, parent2

        arr1 = parent1.to_array()
        arr2 = parent2.to_array()

        # SBX crossover
        eta = 20  # Distribution index
        child1_arr = np.zeros_like(arr1)
        child2_arr = np.zeros_like(arr2)

        for i in range(len(arr1)):
            if random.random() < 0.5:
                if abs(arr1[i] - arr2[i]) > 1e-10:
                    if arr1[i] < arr2[i]:
                        y1, y2 = arr1[i], arr2[i]
                    else:
                        y1, y2 = arr2[i], arr1[i]

                    beta = 1.0 + (2.0 * (y1 - 0) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1.0))

                    rand = random.random()
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1.0))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1.0))

                    child1_arr[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    child2_arr[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                else:
                    child1_arr[i] = arr1[i]
                    child2_arr[i] = arr2[i]
            else:
                child1_arr[i] = arr1[i]
                child2_arr[i] = arr2[i]

        return Individual.from_array(child1_arr), Individual.from_array(child2_arr)

    def mutate(self, ind: Individual) -> Individual:
        """Polynomial mutation operator."""
        arr = ind.to_array()
        bounds = [
            self.config.wingspan_bounds,
            self.config.aspect_ratio_bounds,
            self.config.taper_ratio_bounds,
            self.config.twist_bounds,
        ]

        eta = 20  # Distribution index

        for i in range(len(arr)):
            if random.random() < self.config.mutation_prob:
                y = arr[i]
                lb, ub = bounds[i]

                delta1 = (y - lb) / (ub - lb)
                delta2 = (ub - y) / (ub - lb)

                rand = random.random()
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** (1.0 / (eta + 1.0))

                arr[i] = y + deltaq * (ub - lb)
                arr[i] = np.clip(arr[i], lb, ub)

        return Individual.from_array(arr)

    def run(self, verbose: bool = True) -> List[Individual]:
        """Run the optimization."""
        # Initialize
        self.population = self.initialize_population()
        self.population = self.evaluate_population(self.population)

        for gen in range(self.config.generations):
            # Create offspring
            offspring = []
            while len(offspring) < self.config.population_size:
                parent1 = self.tournament_select(self.population)
                parent2 = self.tournament_select(self.population)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                offspring.extend([child1, child2])

            offspring = offspring[: self.config.population_size]
            offspring = self.evaluate_population(offspring)

            # Combine populations
            combined = self.population + offspring

            # Non-dominated sorting
            fronts = self.non_dominated_sort(combined)

            # Select next generation
            self.population = []
            for front in fronts:
                if len(self.population) + len(front) <= self.config.population_size:
                    self.crowding_distance(front)
                    self.population.extend(front)
                else:
                    # Need to select from this front
                    self.crowding_distance(front)
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    remaining = self.config.population_size - len(self.population)
                    self.population.extend(front[:remaining])
                    break

            # Update Pareto front
            self.pareto_front = fronts[0] if fronts else []

            # Log progress
            if verbose and (gen % 10 == 0 or gen == self.config.generations - 1):
                feasible = [ind for ind in self.pareto_front if ind.is_feasible]
                if feasible:
                    best_ld = max(ind.objectives.glide_ratio for ind in feasible if ind.objectives)
                    best_solar = max(ind.objectives.solar_area for ind in feasible if ind.objectives)
                else:
                    best_ld = 0
                    best_solar = 0

                print(f"Gen {gen:3d}: Pareto size={len(self.pareto_front)}, "
                      f"Feasible={len(feasible)}, Best L/D={best_ld:.1f}, "
                      f"Best Solar={best_solar:.3f}m²")

            # Record history
            self.history.append({
                "generation": gen,
                "pareto_size": len(self.pareto_front),
                "feasible_count": len([ind for ind in self.pareto_front if ind.is_feasible]),
            })

        return self.pareto_front


def run_optimization(
    config: Dict,
    population_size: int = 100,
    generations: int = 50,
    verbose: bool = True,
) -> Optional[Dict]:
    """
    Run wing optimization with given configuration.

    Args:
        config: Configuration dictionary
        population_size: GA population size
        generations: Number of generations
        verbose: Print progress

    Returns:
        Best design parameters or None if failed
    """
    opt_cfg = config.get("optimization", {})
    var_bounds = opt_cfg.get("variables", {})

    opt_config = OptimizationConfig(
        wingspan_bounds=tuple(var_bounds.get("wingspan", [3.0, 5.0])),
        aspect_ratio_bounds=tuple(var_bounds.get("aspect_ratio", [15.0, 25.0])),
        taper_ratio_bounds=tuple(var_bounds.get("taper_ratio", [0.3, 0.6])),
        twist_bounds=tuple(var_bounds.get("twist", [-5.0, 0.0])),
        population_size=population_size,
        generations=generations,
    )

    optimizer = NSGA2Optimizer(opt_config)
    pareto_front = optimizer.run(verbose=verbose)

    # Find best trade-off solution
    feasible = [ind for ind in pareto_front if ind.is_feasible]
    if not feasible:
        print("Warning: No feasible solutions found")
        return None

    # Select by highest L/D among top 25% solar area
    feasible.sort(key=lambda x: x.objectives.solar_area if x.objectives else 0, reverse=True)
    top_solar = feasible[: max(1, len(feasible) // 4)]
    best = max(top_solar, key=lambda x: x.objectives.glide_ratio if x.objectives else 0)

    return {
        "wingspan": best.wingspan,
        "aspect_ratio": best.aspect_ratio,
        "taper_ratio": best.taper_ratio,
        "twist": best.twist,
        "glide_ratio": best.objectives.glide_ratio if best.objectives else 0,
        "solar_area": best.objectives.solar_area if best.objectives else 0,
    }


def main():
    """Command-line entry point."""
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    result = run_optimization(config, verbose=True)
    if result:
        print("\n" + "=" * 60)
        print("OPTIMAL DESIGN")
        print("=" * 60)
        for k, v in result.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
