# evolution/population.py
"""
Island-based population management for GP evolution.

Contains: GPPopulation, evaluate_strategy_parallel, create_random_gp_strategy
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from evolution.nodes import Node
from evolution.tree_ops import TreeGenerator, GPOperators
from evolution.features import FeatureLibrary
from evolution.strategy import GPStrategy
from evolution.walkforward import WalkForwardEvaluator


# ═══════════════════════════════════════════════════════════════════════════════
# PARALLEL EVALUATION SUPPORT
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_strategy_parallel(args: Tuple[GPStrategy, Dict[str, Any]]) -> Tuple[str, float, List[Dict]]:
    """
    Module-level function for parallel strategy evaluation.
    
    This function is picklable and can be used with multiprocessing.Pool.
    
    Args:
        args: Tuple of (strategy, evaluator_config)
    
    Returns:
        Tuple of (strategy_id, fitness, period_results)
    """
    strategy, evaluator_config = args
    
    # Reconstruct evaluator from config
    evaluator = WalkForwardEvaluator.from_config(evaluator_config)
    
    # Evaluate strategy
    fitness_result = evaluator.evaluate_strategy(strategy)
    
    # Update strategy in-place
    strategy.period_metrics = fitness_result.period_results
    strategy.fitness = fitness_result.total
    
    return (strategy.strategy_id, fitness_result.total, fitness_result.period_results)


# ═══════════════════════════════════════════════════════════════════════════════
# POPULATION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class GPPopulation:
    """Island-based population for GP evolution."""
    
    def __init__(
        self,
        population_size: int = 100,
        n_islands: int = 4,
        migration_rate: float = 0.1,
        migration_interval: int = 5,
        max_depth: int = 6,
        tournament_size: int = 5,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.25,
        elite_count: int = 2,
        parsimony_coefficient: float = 0.002,
        enable_smc: bool = False,
        enable_sr: bool = False,
        enable_oil: bool = False,
    ):
        self.population_size = population_size
        self.n_islands = n_islands
        self.migration_rate = migration_rate
        self.migration_interval = migration_interval
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_count = elite_count
        self.parsimony_coefficient = parsimony_coefficient
        
        self.feature_lib = FeatureLibrary(
            enable_smc=enable_smc,
            enable_sr=enable_sr,
            enable_oil=enable_oil
        )
        self.generator = TreeGenerator(self.feature_lib.feature_names)
        self.operators = GPOperators(self.feature_lib.feature_names, max_depth)
        
        self.islands: List[List[GPStrategy]] = [[] for _ in range(n_islands)]
        self.generation = 0
        self.best_strategy: Optional[GPStrategy] = None
        self.best_fitness = 0.0
        
        self.seen_formulas: set = set()
    
    def initialize(self):
        """Initialize islands with diverse strategies."""
        island_size = self.population_size // self.n_islands
        
        for island_idx in range(self.n_islands):
            island = []
            
            preferred_depth = 2 + (island_idx % (self.max_depth - 1))
            
            for _ in range(island_size):
                depth = preferred_depth + random.randint(-1, 1)
                depth = max(2, min(self.max_depth, depth))
                
                strategy = self._create_random_strategy(depth)
                island.append(strategy)
            
            self.islands[island_idx] = island
    
    def _create_random_strategy(self, depth: int) -> GPStrategy:
        """Create a random strategy."""
        tree = self.generator.random_tree(max_depth=depth, method="grow")
        
        return GPStrategy(
            tree=tree,
            top_pct=random.choice([10, 15, 20, 25]),
            holding_period=random.choice([5, 10, 21, 42]),
            execution_lag=1,
            generation=0,
            origin="random"
        )
    
    def adjusted_fitness(self, strategy: GPStrategy) -> float:
        """Fitness with parsimony and novelty adjustments."""
        base = strategy.fitness
        
        complexity_penalty = self.parsimony_coefficient * strategy.complexity()
        
        formula = strategy.get_formula()
        formula_hash = hash(formula)
        
        if formula_hash not in self.seen_formulas:
            novelty_bonus = 0.02
            self.seen_formulas.add(formula_hash)
        else:
            novelty_bonus = 0
        
        return base - complexity_penalty + novelty_bonus
    
    def evolve_island(self, island_idx: int, evaluator: WalkForwardEvaluator):
        """Evolve a single island."""
        island = self.islands[island_idx]
        
        for strategy in island:
            result = evaluator.evaluate_strategy(strategy)
            strategy.fitness = result.total
        
        island.sort(key=lambda s: self.adjusted_fitness(s), reverse=True)
        
        if island[0].fitness > self.best_fitness:
            self.best_fitness = island[0].fitness
            self.best_strategy = island[0].copy()
        
        new_island = []
        
        for elite in island[:self.elite_count]:
            elite_copy = elite.copy()
            elite_copy.generation = self.generation + 1
            elite_copy.origin = "elite"
            new_island.append(elite_copy)
        
        while len(new_island) < len(island):
            if random.random() < self.crossover_prob:
                p1 = self._tournament_select(island)
                p2 = self._tournament_select(island)
                child = self.operators.crossover(p1, p2)
            else:
                parent = self._tournament_select(island)
                child = self.operators.mutate(parent)
            
            if child.tree.depth() <= self.max_depth:
                new_island.append(child)
        
        self.islands[island_idx] = new_island
    
    def _tournament_select(self, population: List[GPStrategy]) -> GPStrategy:
        """Tournament selection."""
        tournament = random.sample(
            population, 
            min(self.tournament_size, len(population))
        )
        return max(tournament, key=lambda s: self.adjusted_fitness(s))
    
    def migrate(self):
        """Migrate best individuals between islands."""
        n_migrants = max(1, int(len(self.islands[0]) * self.migration_rate))
        
        migrants = []
        for island in self.islands:
            island.sort(key=lambda s: self.adjusted_fitness(s), reverse=True)
            migrants.append(island[:n_migrants])
        
        for i, island in enumerate(self.islands):
            source_island = (i - 1) % self.n_islands
            incoming = [m.copy() for m in migrants[source_island]]
            
            island.sort(key=lambda s: self.adjusted_fitness(s), reverse=True)
            self.islands[i] = island[:-n_migrants] + incoming
    
    def evolve(self, evaluator: WalkForwardEvaluator):
        """Evolve all islands for one generation."""
        for island_idx in range(self.n_islands):
            self.evolve_island(island_idx, evaluator)
        
        if self.generation > 0 and self.generation % self.migration_interval == 0:
            self.migrate()
        
        self.generation += 1
    
    @property
    def population(self) -> List[GPStrategy]:
        """Flatten islands into single population."""
        return [s for island in self.islands for s in island]
    
    def get_statistics(self) -> Dict:
        """Population statistics."""
        all_strategies = self.population
        fitnesses = [s.fitness for s in all_strategies]
        complexities = [s.complexity() for s in all_strategies]
        
        return {
            'generation': self.generation,
            'best_fitness': max(fitnesses) if fitnesses else 0,
            'avg_fitness': np.mean(fitnesses) if fitnesses else 0,
            'min_fitness': min(fitnesses) if fitnesses else 0,
            'avg_complexity': np.mean(complexities) if complexities else 0,
            'unique_formulas': len(self.seen_formulas),
            'best_formula': self.best_strategy.get_formula() if self.best_strategy else "",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_random_gp_strategy(
    max_depth: int = 4,
    top_pct: float = 20.0,
    method: str = "grow"
) -> GPStrategy:
    """Create a random GP strategy."""
    feature_lib = FeatureLibrary()
    generator = TreeGenerator(feature_lib.feature_names)
    tree = generator.random_tree(max_depth=max_depth, method=method)
    
    return GPStrategy(
        tree=tree,
        top_pct=top_pct,
        execution_lag=1,
        generation=0,
        origin="random"
    )
