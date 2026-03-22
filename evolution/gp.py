# evolution/gp.py — Re-export shim for backward compatibility
"""
GP Strategy Discovery - Complete Implementation

This module re-exports all classes and functions from the decomposed modules
so that existing `from evolution.gp import X` statements continue to work.

Decomposed into:
- evolution/nodes.py       — Node, FeatureNode, ConstantNode, BinaryOpNode, UnaryOpNode, ConditionalNode
- evolution/tree_ops.py    — TreeGenerator, GPOperators
- evolution/features.py    — FeatureLibrary
- evolution/gp_fitness.py  — FitnessResult, calculate_fitness, calculate_fitness_v2, etc.
- evolution/strategy.py    — GPStrategy
- evolution/walkforward.py — WalkForwardEvaluator
- evolution/population.py  — GPPopulation, evaluate_strategy_parallel, create_random_gp_strategy
"""

# Node classes
from evolution.nodes import (
    Node,
    FeatureNode,
    ConstantNode,
    BinaryOpNode,
    UnaryOpNode,
    ConditionalNode,
)

# Tree operations
from evolution.tree_ops import (
    TreeGenerator,
    GPOperators,
)

# Feature library
from evolution.features import (
    FeatureLibrary,
)

# Fitness functions
from evolution.gp_fitness import (
    FitnessResult,
    calculate_fitness,
    calculate_fitness_v2,
    _recency_weighted_mean,
    _get_drawdown_thresholds,
)

# Strategy
from evolution.strategy import (
    GPStrategy,
)

# Walk-forward evaluator
from evolution.walkforward import (
    WalkForwardEvaluator,
)

# Population management and convenience functions
from evolution.population import (
    GPPopulation,
    evaluate_strategy_parallel,
    create_random_gp_strategy,
)

__all__ = [
    # Nodes
    'Node', 'FeatureNode', 'ConstantNode', 'BinaryOpNode', 'UnaryOpNode', 'ConditionalNode',
    # Tree ops
    'TreeGenerator', 'GPOperators',
    # Features
    'FeatureLibrary',
    # Fitness
    'FitnessResult', 'calculate_fitness', 'calculate_fitness_v2',
    '_recency_weighted_mean', '_get_drawdown_thresholds',
    # Strategy
    'GPStrategy',
    # Walk-forward
    'WalkForwardEvaluator',
    # Population
    'GPPopulation', 'evaluate_strategy_parallel', 'create_random_gp_strategy',
]
