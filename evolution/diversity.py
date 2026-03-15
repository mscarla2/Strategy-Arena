"""
Diversity Preservation for GP Evolution

Implements mechanisms to maintain population diversity and prevent premature convergence:
- Fitness Sharing: Penalize similar strategies to maintain diversity
- Novelty Search: Reward strategies that behave differently
- Island Model: Multiple sub-populations with periodic migration
- Behavioral Distance: Measure similarity by behavior, not structure

These mechanisms help the GP system explore the search space more thoroughly
and discover genuinely novel strategies instead of converging to local optima.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import random
import copy


@dataclass
class BehavioralDescriptor:
    """
    Describes a strategy's behavior across test periods.
    
    Used to measure behavioral similarity between strategies.
    """
    strategy_id: str
    period_returns: List[float]
    period_sharpes: List[float]
    period_drawdowns: List[float]
    turnover: float
    win_rate: float
    avg_position_count: float
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for distance calculation."""
        features = []
        features.extend(self.period_returns)
        features.extend(self.period_sharpes)
        features.extend(self.period_drawdowns)
        features.append(self.turnover)
        features.append(self.win_rate)
        features.append(self.avg_position_count)
        return np.array(features)


class FitnessSharing:
    """
    Fitness sharing mechanism to maintain population diversity.
    
    Penalizes strategies that are behaviorally similar to others,
    encouraging the population to spread out in behavioral space.
    """
    
    def __init__(
        self,
        sigma_share: float = 0.1,
        alpha: float = 1.0,
        distance_metric: str = 'euclidean'
    ):
        """
        Args:
            sigma_share: Sharing radius (strategies within this distance share fitness)
            alpha: Sharing function exponent (higher = more aggressive sharing)
            distance_metric: 'euclidean', 'manhattan', or 'cosine'
        """
        self.sigma_share = sigma_share
        self.alpha = alpha
        self.distance_metric = distance_metric
    
    def calculate_behavioral_distance(
        self,
        desc1: BehavioralDescriptor,
        desc2: BehavioralDescriptor
    ) -> float:
        """
        Calculate behavioral distance between two strategies.
        
        Returns:
            Distance in [0, inf), where 0 = identical behavior
        """
        vec1 = desc1.to_vector()
        vec2 = desc2.to_vector()
        
        if self.distance_metric == 'euclidean':
            distance = np.linalg.norm(vec1 - vec2)
        elif self.distance_metric == 'manhattan':
            distance = np.sum(np.abs(vec1 - vec2))
        elif self.distance_metric == 'cosine':
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            if norm_product == 0:
                distance = 1.0
            else:
                cosine_sim = dot_product / norm_product
                distance = 1.0 - cosine_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        return distance
    
    def sharing_function(self, distance: float) -> float:
        """
        Calculate sharing coefficient based on distance.
        
        Returns:
            Sharing coefficient in [0, 1], where 1 = full sharing
        """
        if distance >= self.sigma_share:
            return 0.0
        else:
            return 1.0 - (distance / self.sigma_share) ** self.alpha
    
    def apply_fitness_sharing(
        self,
        strategies: List,  # List of GPStrategy objects
        descriptors: Dict[str, BehavioralDescriptor]
    ) -> Dict[str, float]:
        """
        Apply fitness sharing to population.
        
        Args:
            strategies: List of GPStrategy objects
            descriptors: Dict mapping strategy_id to BehavioralDescriptor
        
        Returns:
            Dict mapping strategy_id to shared fitness
        """
        shared_fitness = {}
        
        for strategy in strategies:
            if strategy.strategy_id not in descriptors:
                # No descriptor, use original fitness
                shared_fitness[strategy.strategy_id] = strategy.fitness
                continue
            
            # Calculate niche count (sum of sharing coefficients)
            niche_count = 0.0
            desc1 = descriptors[strategy.strategy_id]
            
            for other_strategy in strategies:
                if other_strategy.strategy_id not in descriptors:
                    continue
                
                desc2 = descriptors[other_strategy.strategy_id]
                distance = self.calculate_behavioral_distance(desc1, desc2)
                niche_count += self.sharing_function(distance)
            
            # Shared fitness = original fitness / niche count
            if niche_count > 0:
                shared_fitness[strategy.strategy_id] = strategy.fitness / niche_count
            else:
                shared_fitness[strategy.strategy_id] = strategy.fitness
        
        return shared_fitness


class NoveltySearch:
    """
    Novelty search mechanism to reward behavioral diversity.
    
    Instead of (or in addition to) optimizing fitness, reward strategies
    that behave differently from the archive of past strategies.
    """
    
    def __init__(
        self,
        k_nearest: int = 15,
        archive_size: int = 100,
        novelty_threshold: float = 0.5,
        add_probability: float = 0.1
    ):
        """
        Args:
            k_nearest: Number of nearest neighbors for novelty calculation
            archive_size: Maximum size of behavioral archive
            novelty_threshold: Minimum novelty to add to archive
            add_probability: Probability of adding novel strategy to archive
        """
        self.k_nearest = k_nearest
        self.archive_size = archive_size
        self.novelty_threshold = novelty_threshold
        self.add_probability = add_probability
        self.archive: List[BehavioralDescriptor] = []
    
    def calculate_novelty(
        self,
        descriptor: BehavioralDescriptor,
        population_descriptors: List[BehavioralDescriptor]
    ) -> float:
        """
        Calculate novelty score for a strategy.
        
        Novelty = average distance to k-nearest neighbors in archive + population
        
        Returns:
            Novelty score (higher = more novel)
        """
        # Combine archive and current population
        all_descriptors = self.archive + population_descriptors
        
        if len(all_descriptors) < self.k_nearest:
            # Not enough neighbors, return high novelty
            return 1.0
        
        # Calculate distances to all other strategies
        distances = []
        for other_desc in all_descriptors:
            if other_desc.strategy_id == descriptor.strategy_id:
                continue
            
            vec1 = descriptor.to_vector()
            vec2 = other_desc.to_vector()
            distance = np.linalg.norm(vec1 - vec2)
            distances.append(distance)
        
        # Average distance to k-nearest neighbors
        distances.sort()
        k_nearest_distances = distances[:self.k_nearest]
        novelty = np.mean(k_nearest_distances)
        
        return novelty
    
    def update_archive(
        self,
        descriptor: BehavioralDescriptor,
        novelty: float
    ):
        """
        Add strategy to behavioral archive if sufficiently novel.
        
        Args:
            descriptor: Behavioral descriptor of strategy
            novelty: Novelty score
        """
        # Add if novel enough and random chance
        if novelty > self.novelty_threshold and random.random() < self.add_probability:
            self.archive.append(descriptor)
            
            # Trim archive if too large
            if len(self.archive) > self.archive_size:
                # Remove least novel strategies
                # (This is a simplification; could use more sophisticated pruning)
                self.archive.pop(0)
    
    def get_archive_statistics(self) -> Dict:
        """Get statistics about the behavioral archive."""
        if not self.archive:
            return {
                'size': 0,
                'avg_return': 0,
                'avg_sharpe': 0,
                'diversity': 0
            }
        
        returns = [np.mean(desc.period_returns) for desc in self.archive]
        sharpes = [np.mean(desc.period_sharpes) for desc in self.archive]
        
        # Calculate diversity as average pairwise distance
        distances = []
        for i, desc1 in enumerate(self.archive):
            for desc2 in self.archive[i + 1:]:
                vec1 = desc1.to_vector()
                vec2 = desc2.to_vector()
                distances.append(np.linalg.norm(vec1 - vec2))
        
        return {
            'size': len(self.archive),
            'avg_return': np.mean(returns),
            'avg_sharpe': np.mean(sharpes),
            'diversity': np.mean(distances) if distances else 0
        }


class IslandModel:
    """
    Island model for parallel evolution with migration.
    
    Maintains multiple sub-populations (islands) that evolve independently,
    with periodic migration of best strategies between islands.
    """
    
    def __init__(
        self,
        n_islands: int = 4,
        migration_interval: int = 5,
        migration_rate: float = 0.1,
        migration_topology: str = 'ring'
    ):
        """
        Args:
            n_islands: Number of islands (sub-populations)
            migration_interval: Generations between migrations
            migration_rate: Fraction of population to migrate
            migration_topology: 'ring', 'star', or 'fully_connected'
        """
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.migration_topology = migration_topology
        self.islands: List[List] = [[] for _ in range(n_islands)]
        self.generation = 0
    
    def initialize_islands(self, population: List):
        """
        Distribute initial population across islands.
        
        Args:
            population: Initial population to distribute
        """
        # Shuffle and distribute evenly
        shuffled = population.copy()
        random.shuffle(shuffled)
        
        island_size = len(shuffled) // self.n_islands
        for i in range(self.n_islands):
            start_idx = i * island_size
            end_idx = start_idx + island_size if i < self.n_islands - 1 else len(shuffled)
            self.islands[i] = shuffled[start_idx:end_idx]
    
    def should_migrate(self) -> bool:
        """Check if it's time to migrate."""
        return self.generation > 0 and self.generation % self.migration_interval == 0
    
    def migrate(self):
        """
        Perform migration between islands.
        
        Sends best strategies from each island to neighboring islands.
        """
        if self.migration_topology == 'ring':
            self._migrate_ring()
        elif self.migration_topology == 'star':
            self._migrate_star()
        elif self.migration_topology == 'fully_connected':
            self._migrate_fully_connected()
        else:
            raise ValueError(f"Unknown topology: {self.migration_topology}")
    
    def _migrate_ring(self):
        """Ring topology: Each island sends to next island in ring."""
        migrants_per_island = max(1, int(len(self.islands[0]) * self.migration_rate))
        
        # Collect migrants from each island (best strategies)
        migrants = []
        for island in self.islands:
            sorted_island = sorted(island, key=lambda s: s.fitness, reverse=True)
            island_migrants = sorted_island[:migrants_per_island]
            migrants.append(island_migrants)
        
        # Send migrants to next island in ring
        for i in range(self.n_islands):
            next_island = (i + 1) % self.n_islands
            
            # Replace worst strategies with migrants
            self.islands[next_island].sort(key=lambda s: s.fitness)
            self.islands[next_island][:migrants_per_island] = [
                copy.deepcopy(m) for m in migrants[i]
            ]
    
    def _migrate_star(self):
        """Star topology: All islands exchange with central island (island 0)."""
        migrants_per_island = max(1, int(len(self.islands[0]) * self.migration_rate))
        
        # Collect best from all islands
        all_migrants = []
        for island in self.islands:
            sorted_island = sorted(island, key=lambda s: s.fitness, reverse=True)
            all_migrants.extend(sorted_island[:migrants_per_island])
        
        # Sort all migrants and distribute best to each island
        all_migrants.sort(key=lambda s: s.fitness, reverse=True)
        best_migrants = all_migrants[:migrants_per_island * self.n_islands]
        
        for i, island in enumerate(self.islands):
            start_idx = i * migrants_per_island
            end_idx = start_idx + migrants_per_island
            migrants_for_island = best_migrants[start_idx:end_idx]
            
            # Replace worst strategies
            island.sort(key=lambda s: s.fitness)
            island[:migrants_per_island] = [copy.deepcopy(m) for m in migrants_for_island]
    
    def _migrate_fully_connected(self):
        """Fully connected: Each island exchanges with all others."""
        migrants_per_island = max(1, int(len(self.islands[0]) * self.migration_rate))
        
        # Collect best from each island
        migrants = []
        for island in self.islands:
            sorted_island = sorted(island, key=lambda s: s.fitness, reverse=True)
            migrants.append(sorted_island[:migrants_per_island])
        
        # Each island receives migrants from all others
        for i, island in enumerate(self.islands):
            # Collect migrants from all other islands
            incoming_migrants = []
            for j, other_migrants in enumerate(migrants):
                if i != j:
                    incoming_migrants.extend(other_migrants)
            
            # Select best migrants
            incoming_migrants.sort(key=lambda s: s.fitness, reverse=True)
            selected_migrants = incoming_migrants[:migrants_per_island]
            
            # Replace worst strategies
            island.sort(key=lambda s: s.fitness)
            island[:migrants_per_island] = [copy.deepcopy(m) for m in selected_migrants]
    
    def get_all_strategies(self) -> List:
        """Get all strategies from all islands."""
        all_strategies = []
        for island in self.islands:
            all_strategies.extend(island)
        return all_strategies
    
    def get_best_strategy(self):
        """Get best strategy across all islands."""
        all_strategies = self.get_all_strategies()
        return max(all_strategies, key=lambda s: s.fitness)
    
    def get_island_statistics(self) -> List[Dict]:
        """Get statistics for each island."""
        stats = []
        for i, island in enumerate(self.islands):
            if not island:
                stats.append({
                    'island_id': i,
                    'size': 0,
                    'avg_fitness': 0,
                    'best_fitness': 0,
                    'diversity': 0
                })
                continue
            
            fitnesses = [s.fitness for s in island]
            
            # Calculate diversity (variance in fitness)
            diversity = np.std(fitnesses) if len(fitnesses) > 1 else 0
            
            stats.append({
                'island_id': i,
                'size': len(island),
                'avg_fitness': np.mean(fitnesses),
                'best_fitness': max(fitnesses),
                'diversity': diversity
            })
        
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Diversity Preservation Mechanisms - Example")
    print("=" * 60)
    
    # Create sample behavioral descriptors
    descriptors = []
    for i in range(10):
        desc = BehavioralDescriptor(
            strategy_id=f"strat_{i}",
            period_returns=[random.gauss(0.1, 0.05) for _ in range(5)],
            period_sharpes=[random.gauss(1.0, 0.3) for _ in range(5)],
            period_drawdowns=[random.gauss(0.15, 0.05) for _ in range(5)],
            turnover=random.uniform(0.3, 0.8),
            win_rate=random.uniform(0.4, 0.7),
            avg_position_count=random.uniform(3, 10)
        )
        descriptors.append(desc)
    
    # Test Fitness Sharing
    print("\n1. Fitness Sharing")
    print("-" * 60)
    fs = FitnessSharing(sigma_share=0.1, alpha=1.0)
    
    # Calculate pairwise distances
    print("Behavioral distances:")
    for i in range(3):
        for j in range(i + 1, 3):
            distance = fs.calculate_behavioral_distance(descriptors[i], descriptors[j])
            print(f"  Strategy {i} <-> Strategy {j}: {distance:.4f}")
    
    # Test Novelty Search
    print("\n2. Novelty Search")
    print("-" * 60)
    ns = NoveltySearch(k_nearest=5, archive_size=20)
    
    # Calculate novelty for each strategy
    print("Novelty scores:")
    for i, desc in enumerate(descriptors[:5]):
        novelty = ns.calculate_novelty(desc, descriptors)
        print(f"  Strategy {i}: {novelty:.4f}")
        ns.update_archive(desc, novelty)
    
    archive_stats = ns.get_archive_statistics()
    print(f"\nArchive statistics:")
    print(f"  Size: {archive_stats['size']}")
    print(f"  Diversity: {archive_stats['diversity']:.4f}")
    
    # Test Island Model
    print("\n3. Island Model")
    print("-" * 60)
    
    # Create mock strategies
    class MockStrategy:
        def __init__(self, strategy_id, fitness):
            self.strategy_id = strategy_id
            self.fitness = fitness
    
    population = [MockStrategy(f"s{i}", random.uniform(0, 1)) for i in range(20)]
    
    im = IslandModel(n_islands=4, migration_interval=5, migration_rate=0.2)
    im.initialize_islands(population)
    
    print(f"Initialized {im.n_islands} islands")
    island_stats = im.get_island_statistics()
    for stats in island_stats:
        print(f"  Island {stats['island_id']}: "
              f"size={stats['size']}, "
              f"avg_fitness={stats['avg_fitness']:.3f}, "
              f"best={stats['best_fitness']:.3f}")
    
    # Simulate migration
    im.generation = 5
    if im.should_migrate():
        print("\nPerforming migration...")
        im.migrate()
        island_stats = im.get_island_statistics()
        for stats in island_stats:
            print(f"  Island {stats['island_id']}: "
                  f"avg_fitness={stats['avg_fitness']:.3f}, "
                  f"best={stats['best_fitness']:.3f}")
    
    print("\n" + "=" * 60)
    print("Diversity mechanisms ready for integration!")
