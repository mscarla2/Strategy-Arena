"""
Multi-Objective Optimization (NSGA-II) for GP Evolution

Implements Non-dominated Sorting Genetic Algorithm II (NSGA-II) for optimizing
multiple objectives simultaneously:
- Maximize returns
- Minimize drawdown
- Minimize complexity
- Maximize novelty

Instead of a single fitness value, NSGA-II maintains a Pareto front of
non-dominated solutions, allowing the user to choose trade-offs between objectives.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import random


class ObjectiveType(Enum):
    """Type of objective (minimize or maximize)."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


@dataclass
class Objective:
    """Defines an optimization objective."""
    name: str
    objective_type: ObjectiveType
    weight: float = 1.0  # For weighted sum if needed


@dataclass
class MultiObjectiveFitness:
    """Multi-objective fitness values for a strategy."""
    strategy_id: str
    objectives: Dict[str, float]  # objective_name -> value
    rank: int = 0  # Pareto rank (0 = best front)
    crowding_distance: float = 0.0  # Diversity measure
    
    def dominates(self, other: 'MultiObjectiveFitness', objectives: List[Objective]) -> bool:
        """
        Check if this solution dominates another.
        
        A solution dominates another if it's better in at least one objective
        and not worse in any objective.
        """
        better_in_any = False
        worse_in_any = False
        
        for obj in objectives:
            self_value = self.objectives.get(obj.name, 0)
            other_value = other.objectives.get(obj.name, 0)
            
            if obj.objective_type == ObjectiveType.MAXIMIZE:
                if self_value > other_value:
                    better_in_any = True
                elif self_value < other_value:
                    worse_in_any = True
            else:  # MINIMIZE
                if self_value < other_value:
                    better_in_any = True
                elif self_value > other_value:
                    worse_in_any = True
        
        return better_in_any and not worse_in_any


class NSGA2:
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II).
    
    Multi-objective optimization algorithm that maintains a Pareto front
    of non-dominated solutions.
    
    Reference: Deb et al. (2002) "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II"
    """
    
    def __init__(
        self,
        objectives: List[Objective],
        population_size: int = 100,
        tournament_size: int = 2
    ):
        """
        Args:
            objectives: List of objectives to optimize
            population_size: Size of population
            tournament_size: Tournament size for selection
        """
        self.objectives = objectives
        self.population_size = population_size
        self.tournament_size = tournament_size
    
    def fast_non_dominated_sort(
        self,
        population: List,  # List of GPStrategy objects
        fitness_values: Dict[str, MultiObjectiveFitness]
    ) -> List[List[str]]:
        """
        Sort population into Pareto fronts using fast non-dominated sorting.
        
        Args:
            population: List of strategies
            fitness_values: Dict mapping strategy_id to MultiObjectiveFitness
        
        Returns:
            List of fronts, where each front is a list of strategy IDs
        """
        # Initialize domination sets and counters
        dominated_by = {s.strategy_id: [] for s in population}  # Strategies dominated by s
        domination_count = {s.strategy_id: 0 for s in population}  # How many dominate s
        
        fronts = [[]]  # First front
        
        # Compare all pairs
        for i, strategy1 in enumerate(population):
            for strategy2 in population[i + 1:]:
                fitness1 = fitness_values[strategy1.strategy_id]
                fitness2 = fitness_values[strategy2.strategy_id]
                
                if fitness1.dominates(fitness2, self.objectives):
                    dominated_by[strategy1.strategy_id].append(strategy2.strategy_id)
                    domination_count[strategy2.strategy_id] += 1
                elif fitness2.dominates(fitness1, self.objectives):
                    dominated_by[strategy2.strategy_id].append(strategy1.strategy_id)
                    domination_count[strategy1.strategy_id] += 1
        
        # First front: strategies not dominated by anyone
        for strategy in population:
            if domination_count[strategy.strategy_id] == 0:
                fitness_values[strategy.strategy_id].rank = 0
                fronts[0].append(strategy.strategy_id)
        
        # Build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            
            for strategy_id in fronts[current_front]:
                # For each strategy this one dominates
                for dominated_id in dominated_by[strategy_id]:
                    domination_count[dominated_id] -= 1
                    
                    # If no longer dominated by anyone, add to next front
                    if domination_count[dominated_id] == 0:
                        fitness_values[dominated_id].rank = current_front + 1
                        next_front.append(dominated_id)
            
            current_front += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts
    
    def calculate_crowding_distance(
        self,
        front: List[str],  # List of strategy IDs
        fitness_values: Dict[str, MultiObjectiveFitness]
    ):
        """
        Calculate crowding distance for strategies in a front.
        
        Crowding distance measures how close a solution is to its neighbors.
        Higher distance = more isolated = more diverse.
        
        Modifies fitness_values in place.
        """
        if len(front) <= 2:
            # Boundary solutions get infinite distance
            for strategy_id in front:
                fitness_values[strategy_id].crowding_distance = float('inf')
            return
        
        # Initialize distances
        for strategy_id in front:
            fitness_values[strategy_id].crowding_distance = 0.0
        
        # For each objective
        for obj in self.objectives:
            # Sort front by this objective
            sorted_front = sorted(
                front,
                key=lambda sid: fitness_values[sid].objectives.get(obj.name, 0)
            )
            
            # Boundary solutions get infinite distance
            fitness_values[sorted_front[0]].crowding_distance = float('inf')
            fitness_values[sorted_front[-1]].crowding_distance = float('inf')
            
            # Calculate range
            obj_min = fitness_values[sorted_front[0]].objectives.get(obj.name, 0)
            obj_max = fitness_values[sorted_front[-1]].objectives.get(obj.name, 0)
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calculate crowding distance for middle solutions
            for i in range(1, len(sorted_front) - 1):
                prev_value = fitness_values[sorted_front[i - 1]].objectives.get(obj.name, 0)
                next_value = fitness_values[sorted_front[i + 1]].objectives.get(obj.name, 0)
                
                distance = (next_value - prev_value) / obj_range
                fitness_values[sorted_front[i]].crowding_distance += distance
    
    def crowded_comparison(
        self,
        fitness1: MultiObjectiveFitness,
        fitness2: MultiObjectiveFitness
    ) -> int:
        """
        Compare two solutions using crowded comparison operator.
        
        Prefers:
        1. Lower rank (better Pareto front)
        2. Higher crowding distance (more diverse)
        
        Returns:
            1 if fitness1 is better, -1 if fitness2 is better, 0 if equal
        """
        if fitness1.rank < fitness2.rank:
            return 1
        elif fitness1.rank > fitness2.rank:
            return -1
        else:
            # Same rank, compare crowding distance
            if fitness1.crowding_distance > fitness2.crowding_distance:
                return 1
            elif fitness1.crowding_distance < fitness2.crowding_distance:
                return -1
            else:
                return 0
    
    def tournament_select(
        self,
        population: List,
        fitness_values: Dict[str, MultiObjectiveFitness]
    ):
        """
        Select parent using tournament selection with crowded comparison.
        
        Returns:
            Selected strategy
        """
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        
        best = tournament[0]
        best_fitness = fitness_values[best.strategy_id]
        
        for strategy in tournament[1:]:
            strategy_fitness = fitness_values[strategy.strategy_id]
            if self.crowded_comparison(strategy_fitness, best_fitness) > 0:
                best = strategy
                best_fitness = strategy_fitness
        
        return best
    
    def get_pareto_front(
        self,
        population: List,
        fitness_values: Dict[str, MultiObjectiveFitness]
    ) -> List:
        """
        Get first Pareto front (non-dominated solutions).
        
        Returns:
            List of strategies in first front
        """
        fronts = self.fast_non_dominated_sort(population, fitness_values)
        
        if not fronts or not fronts[0]:
            return []
        
        # Return strategies in first front
        first_front_ids = fronts[0]
        return [s for s in population if s.strategy_id in first_front_ids]
    
    def select_next_generation(
        self,
        population: List,
        offspring: List,
        fitness_values: Dict[str, MultiObjectiveFitness]
    ) -> List:
        """
        Select next generation from combined population and offspring.
        
        Uses elitist selection: best solutions are preserved.
        
        Returns:
            Next generation population
        """
        # Combine population and offspring
        combined = population + offspring
        
        # Fast non-dominated sort
        fronts = self.fast_non_dominated_sort(combined, fitness_values)
        
        # Calculate crowding distance for each front
        for front in fronts:
            self.calculate_crowding_distance(front, fitness_values)
        
        # Select next generation
        next_generation = []
        front_idx = 0
        
        while len(next_generation) < self.population_size and front_idx < len(fronts):
            front = fronts[front_idx]
            
            if len(next_generation) + len(front) <= self.population_size:
                # Add entire front
                next_generation.extend([s for s in combined if s.strategy_id in front])
            else:
                # Add part of front based on crowding distance
                remaining = self.population_size - len(next_generation)
                front_strategies = [s for s in combined if s.strategy_id in front]
                
                # Sort by crowding distance (descending)
                front_strategies.sort(
                    key=lambda s: fitness_values[s.strategy_id].crowding_distance,
                    reverse=True
                )
                
                next_generation.extend(front_strategies[:remaining])
            
            front_idx += 1
        
        return next_generation[:self.population_size]
    
    def calculate_hypervolume(
        self,
        pareto_front: List,
        fitness_values: Dict[str, MultiObjectiveFitness],
        reference_point: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate hypervolume indicator for Pareto front.
        
        Hypervolume measures the volume of objective space dominated by the front.
        Higher is better.
        
        Args:
            pareto_front: List of strategies in Pareto front
            fitness_values: Dict mapping strategy_id to MultiObjectiveFitness
            reference_point: Reference point for hypervolume calculation
        
        Returns:
            Hypervolume value
        """
        if not pareto_front:
            return 0.0
        
        # Use worst values as reference point if not provided
        if reference_point is None:
            reference_point = {}
            for obj in self.objectives:
                values = [
                    fitness_values[s.strategy_id].objectives.get(obj.name, 0)
                    for s in pareto_front
                ]
                if obj.objective_type == ObjectiveType.MAXIMIZE:
                    reference_point[obj.name] = min(values) - 1.0
                else:
                    reference_point[obj.name] = max(values) + 1.0
        
        # Simplified 2D hypervolume calculation
        # For more objectives, use specialized algorithms
        if len(self.objectives) == 2:
            obj1, obj2 = self.objectives
            
            # Get points
            points = []
            for strategy in pareto_front:
                fitness = fitness_values[strategy.strategy_id]
                x = fitness.objectives.get(obj1.name, 0)
                y = fitness.objectives.get(obj2.name, 0)
                points.append((x, y))
            
            # Sort by first objective
            if obj1.objective_type == ObjectiveType.MAXIMIZE:
                points.sort(reverse=True)
            else:
                points.sort()
            
            # Calculate hypervolume
            hypervolume = 0.0
            ref_x = reference_point[obj1.name]
            ref_y = reference_point[obj2.name]
            
            for i, (x, y) in enumerate(points):
                if i == 0:
                    width = abs(x - ref_x)
                else:
                    width = abs(x - points[i - 1][0])
                
                height = abs(y - ref_y)
                hypervolume += width * height
            
            return hypervolume
        else:
            # For >2 objectives, return approximate measure
            # (Full hypervolume calculation is complex)
            return float(len(pareto_front))


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Multi-Objective Optimization (NSGA-II) - Example")
    print("=" * 60)
    
    # Define objectives
    objectives = [
        Objective("return", ObjectiveType.MAXIMIZE),
        Objective("drawdown", ObjectiveType.MINIMIZE),
        Objective("complexity", ObjectiveType.MINIMIZE),
        Objective("novelty", ObjectiveType.MAXIMIZE)
    ]
    
    print("\nObjectives:")
    for obj in objectives:
        print(f"  {obj.name}: {obj.objective_type.value}")
    
    # Create mock strategies with multi-objective fitness
    class MockStrategy:
        def __init__(self, strategy_id):
            self.strategy_id = strategy_id
            self.fitness = 0.0
    
    population = [MockStrategy(f"s{i}") for i in range(20)]
    
    # Assign random fitness values
    fitness_values = {}
    for strategy in population:
        fitness_values[strategy.strategy_id] = MultiObjectiveFitness(
            strategy_id=strategy.strategy_id,
            objectives={
                "return": random.uniform(0.1, 0.5),
                "drawdown": random.uniform(0.1, 0.4),
                "complexity": random.uniform(10, 100),
                "novelty": random.uniform(0.0, 1.0)
            }
        )
    
    # Initialize NSGA-II
    nsga2 = NSGA2(objectives=objectives, population_size=20, tournament_size=2)
    
    # Perform non-dominated sorting
    print("\nPerforming non-dominated sorting...")
    fronts = nsga2.fast_non_dominated_sort(population, fitness_values)
    
    print(f"Found {len(fronts)} Pareto fronts:")
    for i, front in enumerate(fronts):
        print(f"  Front {i}: {len(front)} strategies")
    
    # Calculate crowding distance for first front
    if fronts and fronts[0]:
        nsga2.calculate_crowding_distance(fronts[0], fitness_values)
        
        print("\nFirst Pareto Front (non-dominated solutions):")
        for strategy_id in fronts[0][:5]:  # Show first 5
            fitness = fitness_values[strategy_id]
            print(f"  {strategy_id}:")
            print(f"    Return: {fitness.objectives['return']:.3f}")
            print(f"    Drawdown: {fitness.objectives['drawdown']:.3f}")
            print(f"    Complexity: {fitness.objectives['complexity']:.1f}")
            print(f"    Novelty: {fitness.objectives['novelty']:.3f}")
            print(f"    Crowding Distance: {fitness.crowding_distance:.3f}")
    
    # Get Pareto front
    pareto_front = nsga2.get_pareto_front(population, fitness_values)
    print(f"\nPareto front size: {len(pareto_front)}")
    
    # Calculate hypervolume
    hypervolume = nsga2.calculate_hypervolume(pareto_front, fitness_values)
    print(f"Hypervolume: {hypervolume:.3f}")
    
    print("\n" + "=" * 60)
    print("NSGA-II ready for integration into GP evolution!")
