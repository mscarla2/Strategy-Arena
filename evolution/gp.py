# evolution/gp.py
"""
GP Strategy Discovery - Complete Implementation

Includes:
1. Expanded feature library (~90 features)
2. Expression tree nodes and operators
3. Tree generation and mutation
4. Walk-forward evaluation
5. Island-based population management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import uuid
import random
import copy
import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache


# ═══════════════════════════════════════════════════════════════════════════════
# EXPRESSION TREE NODES
# ═══════════════════════════════════════════════════════════════════════════════

class Node(ABC):
    """Base class for expression tree nodes."""
    
    @abstractmethod
    def evaluate(self, features: Dict[str, pd.Series]) -> pd.Series:
        """Evaluate this node given feature values."""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Human-readable representation."""
        pass
    
    @abstractmethod
    def copy(self) -> 'Node':
        """Deep copy of this node."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Number of nodes in this subtree."""
        pass
    
    @abstractmethod
    def depth(self) -> int:
        """Depth of this subtree."""
        pass
    
    @abstractmethod
    def get_children(self) -> List['Node']:
        """Get child nodes."""
        pass
    
    @abstractmethod
    def set_children(self, children: List['Node']):
        """Set child nodes."""
        pass


class FeatureNode(Node):
    """Leaf node that returns a feature value."""
    
    def __init__(self, feature_name: str):
        self.feature_name = feature_name
    
    def evaluate(self, features: Dict[str, pd.Series]) -> pd.Series:
        if self.feature_name in features:
            return features[self.feature_name]
        # Return zeros if feature not found
        first_feature = next(iter(features.values()))
        return pd.Series(0.0, index=first_feature.index)
    
    def to_string(self) -> str:
        return self.feature_name
    
    def copy(self) -> 'FeatureNode':
        return FeatureNode(self.feature_name)
    
    def size(self) -> int:
        return 1
    
    def depth(self) -> int:
        return 1
    
    def get_children(self) -> List[Node]:
        return []
    
    def set_children(self, children: List[Node]):
        pass  # No children


class ConstantNode(Node):
    """Leaf node that returns a constant value."""
    
    def __init__(self, value: float):
        self.value = value
    
    def evaluate(self, features: Dict[str, pd.Series]) -> pd.Series:
        first_feature = next(iter(features.values()))
        return pd.Series(self.value, index=first_feature.index)
    
    def to_string(self) -> str:
        return f"{self.value:.3f}"
    
    def copy(self) -> 'ConstantNode':
        return ConstantNode(self.value)
    
    def size(self) -> int:
        return 1
    
    def depth(self) -> int:
        return 1
    
    def get_children(self) -> List[Node]:
        return []
    
    def set_children(self, children: List[Node]):
        pass


class BinaryOpNode(Node):
    """Node that applies a binary operation to two children."""
    
    OPERATIONS = {
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'div': lambda a, b: a / b.replace(0, np.nan),
        'max': lambda a, b: np.maximum(a, b),
        'min': lambda a, b: np.minimum(a, b),
        'avg': lambda a, b: (a + b) / 2,
    }
    
    def __init__(self, op: str, left: Node, right: Node):
        self.op = op
        self.left = left
        self.right = right
    
    def evaluate(self, features: Dict[str, pd.Series]) -> pd.Series:
        left_val = self.left.evaluate(features)
        right_val = self.right.evaluate(features)
        
        try:
            result = self.OPERATIONS[self.op](left_val, right_val)
            return result.fillna(0).replace([np.inf, -np.inf], 0)
        except Exception:
            return left_val.fillna(0)
    
    def to_string(self) -> str:
        return f"({self.left.to_string()} {self.op} {self.right.to_string()})"
    
    def copy(self) -> 'BinaryOpNode':
        return BinaryOpNode(self.op, self.left.copy(), self.right.copy())
    
    def size(self) -> int:
        return 1 + self.left.size() + self.right.size()
    
    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())
    
    def get_children(self) -> List[Node]:
        return [self.left, self.right]
    
    def set_children(self, children: List[Node]):
        if len(children) >= 2:
            self.left = children[0]
            self.right = children[1]


class UnaryOpNode(Node):
    """Node that applies a unary operation to one child."""
    
    OPERATIONS = {
        'neg': lambda x: -x,
        'abs': lambda x: x.abs(),
        'sign': lambda x: np.sign(x),
        'sqrt': lambda x: np.sqrt(x.abs()),
        'square': lambda x: x ** 2,
        'inv': lambda x: 1 / x.replace(0, np.nan),
        'log': lambda x: np.log(x.abs() + 1e-10),
        'sigmoid': lambda x: 1 / (1 + np.exp(-x.clip(-10, 10))),
        'tanh': lambda x: np.tanh(x),
        'rank': lambda x: x.rank(pct=True),
        'zscore': lambda x: ((x - x.mean()) / x.std()).fillna(0) if not isinstance(x.std(), pd.Series) and x.std() > 0 else x * 0,
    }
    
    def __init__(self, op: str, child: Node):
        self.op = op
        self.child = child
    
    def evaluate(self, features: Dict[str, pd.Series]) -> pd.Series:
        child_val = self.child.evaluate(features)
        
        try:
            result = self.OPERATIONS[self.op](child_val)
            return result.fillna(0).replace([np.inf, -np.inf], 0)
        except Exception:
            return child_val.fillna(0)
    
    def to_string(self) -> str:
        return f"{self.op}({self.child.to_string()})"
    
    def copy(self) -> 'UnaryOpNode':
        return UnaryOpNode(self.op, self.child.copy())
    
    def size(self) -> int:
        return 1 + self.child.size()
    
    def depth(self) -> int:
        return 1 + self.child.depth()
    
    def get_children(self) -> List[Node]:
        return [self.child]
    
    def set_children(self, children: List[Node]):
        if len(children) >= 1:
            self.child = children[0]


class ConditionalNode(Node):
    """If-then-else node."""
    
    def __init__(self, condition: Node, if_true: Node, if_false: Node):
        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false
    
    def evaluate(self, features: Dict[str, pd.Series]) -> pd.Series:
        cond = self.condition.evaluate(features)
        true_val = self.if_true.evaluate(features)
        false_val = self.if_false.evaluate(features)
        
        return pd.Series(
            np.where(cond > 0, true_val, false_val),
            index=cond.index
        ).fillna(0)
    
    def to_string(self) -> str:
        return f"if({self.condition.to_string()} > 0, {self.if_true.to_string()}, {self.if_false.to_string()})"
    
    def copy(self) -> 'ConditionalNode':
        return ConditionalNode(
            self.condition.copy(),
            self.if_true.copy(),
            self.if_false.copy()
        )
    
    def size(self) -> int:
        return 1 + self.condition.size() + self.if_true.size() + self.if_false.size()
    
    def depth(self) -> int:
        return 1 + max(self.condition.depth(), self.if_true.depth(), self.if_false.depth())
    
    def get_children(self) -> List[Node]:
        return [self.condition, self.if_true, self.if_false]
    
    def set_children(self, children: List[Node]):
        if len(children) >= 3:
            self.condition = children[0]
            self.if_true = children[1]
            self.if_false = children[2]


# ═══════════════════════════════════════════════════════════════════════════════
# TREE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class TreeGenerator:
    """Generates random expression trees."""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.binary_ops = list(BinaryOpNode.OPERATIONS.keys())
        self.unary_ops = list(UnaryOpNode.OPERATIONS.keys())
    
    def random_tree(self, max_depth: int = 4, method: str = "grow") -> Node:
        """
        Generate a random tree.
        
        Args:
            max_depth: Maximum tree depth
            method: "grow" (variable depth) or "full" (always max depth)
        """
        return self._random_node(max_depth, method)
    
    def _random_node(self, depth_remaining: int, method: str) -> Node:
        """Recursively generate a random node."""
        
        if depth_remaining <= 1:
            # Must be a terminal
            return self._random_terminal()
        
        if method == "grow":
            # Choose between terminal and function
            if random.random() < 0.3:
                return self._random_terminal()
        
        # Choose a function node
        node_type = random.choices(
            ['binary', 'unary', 'condition'],
            weights=[0.5, 0.35, 0.15]
        )[0]
        
        if node_type == 'binary':
            op = random.choice(self.binary_ops)
            left = self._random_node(depth_remaining - 1, method)
            right = self._random_node(depth_remaining - 1, method)
            return BinaryOpNode(op, left, right)
        
        elif node_type == 'unary':
            op = random.choice(self.unary_ops)
            child = self._random_node(depth_remaining - 1, method)
            return UnaryOpNode(op, child)
        
        else:  # condition
            condition = self._random_node(depth_remaining - 1, method)
            if_true = self._random_node(depth_remaining - 1, method)
            if_false = self._random_node(depth_remaining - 1, method)
            return ConditionalNode(condition, if_true, if_false)
    
    def _random_terminal(self) -> Node:
        """Generate a random terminal (feature or constant)."""
        if random.random() < 0.85:
            return FeatureNode(random.choice(self.feature_names))
        else:
            # Random constant
            value = random.choice([0.0, 0.5, 1.0, -1.0, 2.0, -0.5])
            return ConstantNode(value)


# ═══════════════════════════════════════════════════════════════════════════════
# GP OPERATORS (Crossover and Mutation)
# ═══════════════════════════════════════════════════════════════════════════════

class GPOperators:
    """Genetic operators for GP trees."""
    
    def __init__(self, feature_names: List[str], max_depth: int = 6):
        self.feature_names = feature_names
        self.max_depth = max_depth
        self.generator = TreeGenerator(feature_names)
    
    def crossover(self, parent1: 'GPStrategy', parent2: 'GPStrategy') -> 'GPStrategy':
        """Subtree crossover between two strategies."""
        tree1 = parent1.tree.copy()
        tree2 = parent2.tree.copy()
        
        # Select random nodes
        nodes1 = self._get_all_nodes(tree1)
        nodes2 = self._get_all_nodes(tree2)
        
        if len(nodes1) < 2 or len(nodes2) < 2:
            return parent1.copy()
        
        # Pick crossover points (avoid root)
        point1 = random.choice(nodes1[1:]) if len(nodes1) > 1 else nodes1[0]
        point2 = random.choice(nodes2[1:]) if len(nodes2) > 1 else nodes2[0]
        
        # Swap subtrees
        self._replace_node(tree1, point1, point2.copy())
        
        # Check depth constraint
        if tree1.depth() > self.max_depth:
            return parent1.copy()
        
        return GPStrategy(
            tree=tree1,
            top_pct=(parent1.top_pct + parent2.top_pct) / 2,
            holding_period=random.choice([parent1.holding_period, parent2.holding_period]),
            execution_lag=1,
            generation=max(parent1.generation, parent2.generation) + 1,
            origin="crossover"
        )
    
    def mutate(self, parent: 'GPStrategy') -> 'GPStrategy':
        """Mutate a strategy's tree."""
        tree = parent.tree.copy()
        
        mutation_type = random.choices(
            ['point', 'subtree', 'hoist', 'shrink'],
            weights=[0.4, 0.3, 0.15, 0.15]
        )[0]
        
        if mutation_type == 'point':
            tree = self._point_mutation(tree)
        elif mutation_type == 'subtree':
            tree = self._subtree_mutation(tree)
        elif mutation_type == 'hoist':
            tree = self._hoist_mutation(tree)
        else:  # shrink
            tree = self._shrink_mutation(tree)
        
        # Check depth constraint
        if tree.depth() > self.max_depth:
            return parent.copy()
        
        return GPStrategy(
            tree=tree,
            top_pct=parent.top_pct,
            holding_period=parent.holding_period,
            execution_lag=1,
            generation=parent.generation + 1,
            origin="mutation"
        )
    
    def _point_mutation(self, tree: Node) -> Node:
        """Change a single node's operation or value."""
        nodes = self._get_all_nodes(tree)
        if not nodes:
            return tree
        
        node = random.choice(nodes)
        
        if isinstance(node, FeatureNode):
            node.feature_name = random.choice(self.feature_names)
        elif isinstance(node, ConstantNode):
            node.value = node.value + random.gauss(0, 0.5)
        elif isinstance(node, BinaryOpNode):
            node.op = random.choice(list(BinaryOpNode.OPERATIONS.keys()))
        elif isinstance(node, UnaryOpNode):
            node.op = random.choice(list(UnaryOpNode.OPERATIONS.keys()))
        
        return tree
    
    def _subtree_mutation(self, tree: Node) -> Node:
        """Replace a random subtree with a new random tree."""
        nodes = self._get_all_nodes(tree)
        if len(nodes) < 2:
            return self.generator.random_tree(max_depth=3)
        
        # Pick a non-root node to replace
        target = random.choice(nodes[1:])
        new_subtree = self.generator.random_tree(max_depth=3)
        
        self._replace_node(tree, target, new_subtree)
        return tree
    
    def _hoist_mutation(self, tree: Node) -> Node:
        """Replace tree with one of its subtrees."""
        nodes = self._get_all_nodes(tree)
        non_terminals = [n for n in nodes if n.get_children()]
        
        if non_terminals:
            return random.choice(non_terminals).copy()
        return tree
    
    def _shrink_mutation(self, tree: Node) -> Node:
        """Replace a subtree with a terminal."""
        nodes = self._get_all_nodes(tree)
        non_terminals = [n for n in nodes[1:] if n.get_children()]  # Exclude root
        
        if non_terminals:
            target = random.choice(non_terminals)
            replacement = self.generator._random_terminal()
            self._replace_node(tree, target, replacement)
        
        return tree
    
    def _get_all_nodes(self, tree: Node) -> List[Node]:
        """Get all nodes in the tree."""
        nodes = [tree]
        for child in tree.get_children():
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _replace_node(self, tree: Node, target: Node, replacement: Node) -> bool:
        """Replace target node with replacement in tree."""
        for i, child in enumerate(tree.get_children()):
            if child is target:
                children = tree.get_children()
                children[i] = replacement
                tree.set_children(children)
                return True
            if self._replace_node(child, target, replacement):
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# EXPANDED FEATURE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureLibrary:
    """
    Comprehensive feature library with ~90 orthogonalized features.
    """
    
    def __init__(self, enable_smc=False, enable_sr=False, enable_oil=False):
        self.enable_smc = enable_smc
        self.enable_sr = enable_sr
        self.enable_oil = enable_oil
        
        self.feature_specs = self._build_feature_specs()
        self.feature_names = list(self.feature_specs.keys())
        self.max_lookback = 504  # 2 years
        self.feature_categories = self._build_category_map()
        
        # Features to exclude from rank transform (already ranked or naturally bounded 0-1)
        self._skip_rank_transform = {
            'relative_strength_21d', 'relative_strength_63d', 'relative_strength_126d',
            'relative_volatility_21d', 'relative_volatility_63d',
            'range_position_21d', 'range_position_63d', 'range_position_252d',
            'high_proximity_63d', 'high_proximity_252d',
            'recovery_rate_63d', 'recovery_rate_126d',
            'mom_consistency_21d', 'mom_consistency_63d',
            'hurst_proxy_63d', 'hurst_proxy_126d',
        }
        
        # Initialize advanced feature calculators if enabled
        self.smc_features = None
        self.sr_features = None
        self.oil_features = None
        
        if self.enable_smc:
            from evolution.smart_money_features import SmartMoneyFeatures
            self.smc_features = SmartMoneyFeatures()
        
        if self.enable_sr:
            from evolution.support_resistance_features import SupportResistanceFeatures
            self.sr_features = SupportResistanceFeatures()
        
        if self.enable_oil:
            from evolution.oil_specific_features import OilSpecificFeatures
            self.oil_features = OilSpecificFeatures()
        
        # Feature cache for performance optimization
        self._feature_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _build_feature_specs(self) -> Dict:
        """Build all feature specifications."""
        specs = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # MOMENTUM FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['mom_5d'] = ('momentum', 5)
        specs['mom_10d'] = ('momentum', 10)
        specs['mom_21d'] = ('momentum', 21)
        specs['mom_63d'] = ('momentum', 63)
        specs['mom_126d'] = ('momentum', 126)
        specs['mom_252d'] = ('momentum', 252)
        
        specs['mom_12_1'] = ('momentum_skip', 252, 21)
        specs['mom_6_1'] = ('momentum_skip', 126, 21)
        specs['mom_3_1'] = ('momentum_skip', 63, 21)
        
        specs['mom_accel_short'] = ('momentum_accel', 5, 21)
        specs['mom_accel_medium'] = ('momentum_accel', 21, 63)
        specs['mom_accel_long'] = ('momentum_accel', 63, 252)
        
        specs['mom_consistency_21d'] = ('momentum_consistency', 21, 5)
        specs['mom_consistency_63d'] = ('momentum_consistency', 63, 21)
        
        specs['mom_smoothness_21d'] = ('momentum_smoothness', 21)
        specs['mom_smoothness_63d'] = ('momentum_smoothness', 63)
        
        # ═══════════════════════════════════════════════════════════════════
        # MEAN REVERSION FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['zscore_5d'] = ('zscore', 5)
        specs['zscore_10d'] = ('zscore', 10)
        specs['zscore_21d'] = ('zscore', 21)
        specs['zscore_63d'] = ('zscore', 63)
        
        specs['dist_ma_10'] = ('distance_from_ma', 10)
        specs['dist_ma_21'] = ('distance_from_ma', 21)
        specs['dist_ma_50'] = ('distance_from_ma', 50)
        specs['dist_ma_200'] = ('distance_from_ma', 200)
        
        specs['reversion_speed_10d'] = ('reversion_speed', 10)
        specs['reversion_speed_21d'] = ('reversion_speed', 21)
        specs['reversion_speed_63d'] = ('reversion_speed', 63)
        
        specs['rsi_14'] = ('rsi', 14)
        specs['rsi_28'] = ('rsi', 28)
        
        # ═══════════════════════════════════════════════════════════════════
        # VOLATILITY FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['vol_5d'] = ('volatility', 5)
        specs['vol_21d'] = ('volatility', 21)
        specs['vol_63d'] = ('volatility', 63)
        specs['vol_252d'] = ('volatility', 252)
        
        specs['vol_ratio_5_21'] = ('vol_ratio', 5, 21)
        specs['vol_ratio_21_63'] = ('vol_ratio', 21, 63)
        specs['vol_ratio_63_252'] = ('vol_ratio', 63, 252)
        
        specs['vol_of_vol_21d'] = ('vol_of_vol', 21)
        specs['vol_of_vol_63d'] = ('vol_of_vol', 63)
        
        specs['vol_trend_21d'] = ('vol_trend', 21, 63)
        
        # ═══════════════════════════════════════════════════════════════════
        # DRAWDOWN AND RISK FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['drawdown_21d'] = ('drawdown', 21)
        specs['drawdown_63d'] = ('drawdown', 63)
        specs['drawdown_126d'] = ('drawdown', 126)
        specs['drawdown_252d'] = ('drawdown', 252)
        
        specs['drawdown_duration_63d'] = ('drawdown_duration', 63)
        specs['drawdown_duration_252d'] = ('drawdown_duration', 252)
        
        specs['recovery_rate_63d'] = ('recovery_rate', 63)
        specs['recovery_rate_126d'] = ('recovery_rate', 126)
        
        specs['max_dd_63d'] = ('max_drawdown', 63)
        specs['max_dd_126d'] = ('max_drawdown', 126)
        
        specs['ulcer_index_63d'] = ('ulcer_index', 63)
        
        # ═══════════════════════════════════════════════════════════════════
        # HIGHER MOMENTS
        # ═══════════════════════════════════════════════════════════════════
        
        specs['skew_21d'] = ('skewness', 21)
        specs['skew_63d'] = ('skewness', 63)
        specs['skew_126d'] = ('skewness', 126)
        
        specs['kurt_21d'] = ('kurtosis', 21)
        specs['kurt_63d'] = ('kurtosis', 63)
        specs['kurt_126d'] = ('kurtosis', 126)
        
        specs['downside_dev_21d'] = ('downside_deviation', 21)
        specs['downside_dev_63d'] = ('downside_deviation', 63)
        
        specs['up_down_ratio_63d'] = ('up_down_capture', 63)
        
        specs['left_tail_21d'] = ('left_tail', 21, 0.05)
        specs['left_tail_63d'] = ('left_tail', 63, 0.05)
        
        # ═══════════════════════════════════════════════════════════════════
        # TREND QUALITY FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['trend_r2_21d'] = ('trend_r2', 21)
        specs['trend_r2_63d'] = ('trend_r2', 63)
        specs['trend_r2_126d'] = ('trend_r2', 126)
        
        specs['trend_slope_21d'] = ('trend_slope', 21)
        specs['trend_slope_63d'] = ('trend_slope', 63)
        
        specs['trend_deviation_21d'] = ('trend_deviation', 21)
        specs['trend_deviation_63d'] = ('trend_deviation', 63)
        
        specs['hurst_proxy_63d'] = ('hurst_proxy', 63)
        specs['hurst_proxy_126d'] = ('hurst_proxy', 126)
        
        # ═══════════════════════════════════════════════════════════════════
        # PRICE LEVEL FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['range_position_21d'] = ('range_position', 21)
        specs['range_position_63d'] = ('range_position', 63)
        specs['range_position_252d'] = ('range_position', 252)
        
        specs['high_proximity_63d'] = ('high_proximity', 63)
        specs['high_proximity_252d'] = ('high_proximity', 252)
        
        specs['breakout_21d'] = ('breakout_strength', 21)
        specs['breakout_63d'] = ('breakout_strength', 63)
        
        specs['support_distance_21d'] = ('support_distance', 21)
        specs['resistance_distance_21d'] = ('resistance_distance', 21)
        
        # ═══════════════════════════════════════════════════════════════════
        # CROSS-SECTIONAL FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['rel_strength_21d'] = ('relative_strength', 21)
        specs['rel_strength_63d'] = ('relative_strength', 63)
        specs['rel_strength_126d'] = ('relative_strength', 126)
        
        specs['rel_vol_21d'] = ('relative_volatility', 21)
        specs['rel_vol_63d'] = ('relative_volatility', 63)
        
        specs['excess_mom_21d'] = ('excess_momentum', 21)
        specs['excess_mom_63d'] = ('excess_momentum', 63)
        
        specs['rel_value_21d'] = ('relative_value', 21)
        specs['rel_value_63d'] = ('relative_value', 63)
        
        # ═══════════════════════════════════════════════════════════════════
        # RISK-ADJUSTED FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['sharpe_21d'] = ('sharpe_ratio', 21)
        specs['sharpe_63d'] = ('sharpe_ratio', 63)
        specs['sharpe_126d'] = ('sharpe_ratio', 126)
        
        specs['sortino_21d'] = ('sortino_ratio', 21)
        specs['sortino_63d'] = ('sortino_ratio', 63)
        
        specs['calmar_63d'] = ('calmar_ratio', 63)
        specs['calmar_126d'] = ('calmar_ratio', 126)
        
        specs['info_ratio_63d'] = ('information_ratio', 63)
        specs['info_ratio_126d'] = ('information_ratio', 126)
        
        # ═══════════════════════════════════════════════════════════════════
        # VOLUME/LIQUIDITY FEATURES (optional)
        # ═══════════════════════════════════════════════════════════════════
        
        specs['volume_trend_21d'] = ('volume_trend', 21)
        specs['volume_ratio_5_21'] = ('volume_ratio', 5, 21)
        specs['price_volume_corr_21d'] = ('price_volume_corr', 21)
        specs['volume_volatility_21d'] = ('volume_volatility', 21)
        
        # ═══════════════════════════════════════════════════════════════════
        # PATTERN FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        specs['gap_reversal_10d'] = ('gap_reversal', 10)
        specs['streak_strength'] = ('streak_strength', 21)
        specs['return_autocorr_1d'] = ('return_autocorr', 21, 1)
        specs['return_autocorr_5d'] = ('return_autocorr', 21, 5)
        specs['abs_return_21d'] = ('abs_return_momentum', 21)
        specs['abs_return_63d'] = ('abs_return_momentum', 63)
        
        # ═══════════════════════════════════════════════════════════════════
        # SMART MONEY CONCEPTS (SMC) - Optional
        # ═══════════════════════════════════════════════════════════════════
        if self.enable_smc:
            specs['smc_order_block_bull'] = ('smc', 'order_block_bull')
            specs['smc_order_block_bear'] = ('smc', 'order_block_bear')
            specs['smc_fvg_bull'] = ('smc', 'fvg_bull')
            specs['smc_fvg_bear'] = ('smc', 'fvg_bear')
            specs['smc_liquidity_sweep'] = ('smc', 'liquidity_sweep')
            specs['smc_break_of_structure'] = ('smc', 'break_of_structure')
        
        # ═══════════════════════════════════════════════════════════════════
        # SUPPORT/RESISTANCE - Optional
        # ═══════════════════════════════════════════════════════════════════
        if self.enable_sr:
            specs['sr_poc_distance'] = ('sr', 'poc_distance')
            specs['sr_value_area_position'] = ('sr', 'value_area_position')
            specs['sr_pivot_traditional'] = ('sr', 'pivot_traditional')
            specs['sr_pivot_fibonacci'] = ('sr', 'pivot_fibonacci')
            specs['sr_pivot_camarilla'] = ('sr', 'pivot_camarilla')
            specs['sr_bb_position'] = ('sr', 'bb_position')
            specs['sr_keltner_position'] = ('sr', 'keltner_position')
            specs['sr_historical_level'] = ('sr', 'historical_level')
        
        # ═══════════════════════════════════════════════════════════════════
        # OIL-SPECIFIC FEATURES - Optional
        # ═══════════════════════════════════════════════════════════════════
        if self.enable_oil:
            specs['oil_wti_correlation'] = ('oil', 'wti_correlation')
            specs['oil_brent_correlation'] = ('oil', 'brent_correlation')
            specs['oil_wti_beta'] = ('oil', 'wti_beta')
            specs['oil_inventory_zscore'] = ('oil', 'inventory_zscore')
            specs['oil_inventory_change'] = ('oil', 'inventory_change')
            specs['oil_crack_spread_321'] = ('oil', 'crack_spread_321')
            specs['oil_crack_spread_532'] = ('oil', 'crack_spread_532')
            specs['oil_seasonal_driving'] = ('oil', 'seasonal_driving')
            specs['oil_seasonal_heating'] = ('oil', 'seasonal_heating')
            specs['oil_wti_brent_spread'] = ('oil', 'wti_brent_spread')
        
        return specs
    
    def _build_category_map(self) -> Dict[str, List[str]]:
        """Map features to categories for analysis."""
        categories = {
            'momentum': [],
            'mean_reversion': [],
            'volatility': [],
            'drawdown': [],
            'higher_moments': [],
            'trend': [],
            'price_level': [],
            'cross_sectional': [],
            'risk_adjusted': [],
            'volume': [],
            'pattern': [],
        }
        
        for name in self.feature_names:
            if name.startswith('mom_'):
                categories['momentum'].append(name)
            elif name.startswith(('zscore', 'dist_ma', 'reversion', 'rsi')):
                categories['mean_reversion'].append(name)
            elif name.startswith('vol_'):
                categories['volatility'].append(name)
            elif name.startswith(('drawdown', 'recovery', 'max_dd', 'ulcer')):
                categories['drawdown'].append(name)
            elif name.startswith(('skew', 'kurt', 'downside', 'up_down', 'left_tail')):
                categories['higher_moments'].append(name)
            elif name.startswith(('trend', 'hurst')):
                categories['trend'].append(name)
            elif name.startswith(('range', 'high_prox', 'breakout', 'support', 'resistance')):
                categories['price_level'].append(name)
            elif name.startswith(('rel_', 'excess')):
                categories['cross_sectional'].append(name)
            elif name.startswith(('sharpe', 'sortino', 'calmar', 'info_ratio')):
                categories['risk_adjusted'].append(name)
            elif name.startswith(('volume', 'price_volume')):
                categories['volume'].append(name)
            else:
                categories['pattern'].append(name)
        
        return categories
    
    def _make_cache_key(self, prices: pd.DataFrame, lag: int, rank_transform: bool) -> str:
        """Create a hashable cache key from prices DataFrame."""
        # Use last date, shape, and column hash as key
        last_date = str(prices.index[-1]) if len(prices) > 0 else "empty"
        shape_str = f"{len(prices)}x{len(prices.columns)}"
        cols_hash = hashlib.md5(','.join(sorted(prices.columns)).encode()).hexdigest()[:8]
        return f"{last_date}_{shape_str}_{cols_hash}_lag{lag}_rank{rank_transform}"
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate': hit_rate,
            'cache_size': len(self._feature_cache)
        }
    
    def clear_cache(self):
        """Clear feature cache and reset statistics."""
        self._feature_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def compute_all(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame = None,
        lag: int = 1,
        rank_transform: bool = True,
        use_cache: bool = True,
    ) -> Dict[str, pd.Series]:
        """
        Compute all features with optional caching.
        
        Args:
            prices: Price DataFrame
            volume: Volume DataFrame (optional)
            lag: Lag for feature computation
            rank_transform: Apply rank transformation
            use_cache: Enable feature caching (default: True)
        
        Returns:
            Dict mapping feature names to Series
        """
        # Check cache first
        if use_cache:
            cache_key = self._make_cache_key(prices, lag, rank_transform)
            if cache_key in self._feature_cache:
                self._cache_hits += 1
                return self._feature_cache[cache_key].copy()
            self._cache_misses += 1
        
        # Compute features
        if lag > 0 and len(prices) > lag:
            lagged_prices = prices.iloc[:-lag]
            lagged_volume = volume.iloc[:-lag] if volume is not None else None
        else:
            lagged_prices = prices
            lagged_volume = volume
        
        features = {}
        
        for name, spec in self.feature_specs.items():
            try:
                features[name] = self._compute_feature(lagged_prices, lagged_volume, spec)
            except Exception:
                features[name] = pd.Series(0.0, index=prices.columns)
        
        # Apply rank transform
        if rank_transform:
            features = self._rank_transform(features)
        
        # Cache the result
        if use_cache:
            # Limit cache size to prevent memory issues (keep last 1000 entries)
            if len(self._feature_cache) >= 1000:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self._feature_cache))
                del self._feature_cache[oldest_key]
            self._feature_cache[cache_key] = features.copy()
        
        return features
    
    def _rank_transform(self, features: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Transform features to cross-sectional percentile ranks."""
        ranked = {}
        
        for name, series in features.items():
            if name in self._skip_rank_transform:
                ranked[name] = series
                continue
            
            # Guard: if a feature accidentally returned a DataFrame, squeeze to Series
            if isinstance(series, pd.DataFrame):
                if series.shape[1] == 1:
                    series = series.iloc[:, 0]
                else:
                    series = series.iloc[:, 0]  # Take first column as fallback
            if series.dtype == object:
                print(f"BAD FEATURE: {name} has object dtype — first value type: {type(series.iloc[0])}")
                try:
                    series = series.astype(float)
                except (ValueError, TypeError):
                    ranked[name] = pd.Series(0.0, index=series.index)
                    continue
            if series.empty or bool(series.isna().all()):
                ranked[name] = series
                continue
            
            ranked[name] = series.rank(pct=True, method='average', na_option='keep')
        
        return ranked
    
    def _compute_feature(
        self, 
        prices: pd.DataFrame, 
        volume: Optional[pd.DataFrame],
        spec: tuple
    ) -> pd.Series:
        """Compute a single feature."""
        
        feature_type = spec[0]
        tickers = prices.columns
        
        # ═══════════════════════════════════════════════════════════════════
        # MOMENTUM
        # ═══════════════════════════════════════════════════════════════════
        
        if feature_type == 'momentum':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            return (prices.iloc[-1] / prices.iloc[-period - 1]) - 1
        
        elif feature_type == 'momentum_skip':
            lookback, skip = spec[1], spec[2]
            if len(prices) < lookback + skip + 1:
                return pd.Series(0.0, index=tickers)
            start_idx = -(lookback + skip + 1)
            end_idx = -(skip + 1) if skip > 0 else -1
            return (prices.iloc[end_idx] / prices.iloc[start_idx]) - 1
        
        elif feature_type == 'momentum_accel':
            short, long = spec[1], spec[2]
            if len(prices) < long + 1:
                return pd.Series(0.0, index=tickers)
            mom_short = (prices.iloc[-1] / prices.iloc[-short - 1]) - 1
            mom_long = (prices.iloc[-1] / prices.iloc[-long - 1]) - 1
            return mom_short - (mom_long * short / long)
        
        elif feature_type == 'momentum_consistency':
            period, sub_period = spec[1], spec[2]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            
            returns = prices.pct_change().iloc[-period:]
            n_sub = period // sub_period
            
            def calc_consistency(col):
                positive_subs = 0
                for i in range(n_sub):
                    start = i * sub_period
                    end = (i + 1) * sub_period
                    sub_return = col.iloc[start:end].sum()
                    if sub_return > 0:
                        positive_subs += 1
                return positive_subs / max(n_sub, 1)
            
            return returns.apply(calc_consistency)
        
        elif feature_type == 'momentum_smoothness':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            
            returns = prices.pct_change().iloc[-period:]
            total_return = (prices.iloc[-1] / prices.iloc[-period]) - 1
            abs_sum = returns.abs().sum()
            smoothness = total_return / abs_sum.replace(0, np.nan)
            return smoothness.fillna(0).clip(-1, 1)
        
        # ═══════════════════════════════════════════════════════════════════
        # MEAN REVERSION
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'zscore':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            mean = recent.mean()
            std = recent.std()
            current = prices.iloc[-1]
            z = (current - mean) / std.replace(0, np.nan)
            return z.fillna(0).clip(-4, 4)
        
        elif feature_type == 'distance_from_ma':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            ma = prices.iloc[-period:].mean()
            current = prices.iloc[-1]
            return (current / ma) - 1
        
        elif feature_type == 'reversion_speed':
            period = spec[1]
            if len(prices) < period * 2:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            
            def calc_autocorr(x):
                if len(x.dropna()) < 5:
                    return 0
                ac = x.autocorr(lag=1)
                return -ac if not np.isnan(ac) else 0
            
            return returns.apply(calc_autocorr)
        
        elif feature_type == 'rsi':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(50.0, index=tickers)
            
            returns = prices.pct_change().iloc[-period:]
            gains = returns.clip(lower=0).mean()
            losses = (-returns.clip(upper=0)).mean()
            
            rs = gains / losses.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)
        
        # ═══════════════════════════════════════════════════════════════════
        # VOLATILITY
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'volatility':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            return returns.std() * np.sqrt(252)
        
        elif feature_type == 'vol_ratio':
            short, long = spec[1], spec[2]
            if len(prices) < long:
                return pd.Series(1.0, index=tickers)
            returns = prices.pct_change()
            vol_short = returns.iloc[-short:].std()
            vol_long = returns.iloc[-long:].std()
            return (vol_short / vol_long.replace(0, np.nan)).fillna(1.0)
        
        elif feature_type == 'vol_of_vol':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change()
            rolling_vol = returns.rolling(5).std()
            return rolling_vol.iloc[-period:].std()
        
        elif feature_type == 'vol_trend':
            short, long = spec[1], spec[2]
            if len(prices) < long:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change()
            vol_now = returns.iloc[-short:].std()
            vol_before = returns.iloc[-long:-short].std()
            return (vol_now / vol_before.replace(0, np.nan) - 1).fillna(0)
        
        # ═══════════════════════════════════════════════════════════════════
        # DRAWDOWN AND RISK
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'drawdown':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            peak = recent.max()
            current = prices.iloc[-1]
            return (current - peak) / peak
        
        elif feature_type == 'drawdown_duration':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            at_high = (recent == running_max)
            
            def days_since_high(col):
                at_high_col = at_high[col.name]
                if at_high_col.iloc[-1]:
                    return 0
                try:
                    last_high_idx = at_high_col[::-1].idxmax()
                    return len(at_high_col) - at_high_col.index.get_loc(last_high_idx) - 1
                except:
                    return period
            
            return recent.apply(days_since_high) / period
        
        elif feature_type == 'recovery_rate':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            drawdown = (recent - running_max) / running_max
            near_high = (drawdown > -0.05).mean()
            return near_high
        
        elif feature_type == 'max_drawdown':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            drawdown = (recent - running_max) / running_max
            return drawdown.min()
        
        elif feature_type == 'ulcer_index':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            drawdown_pct = (recent - running_max) / running_max * 100
            ulcer = np.sqrt((drawdown_pct ** 2).mean())
            return -ulcer / 100
        
        # ═══════════════════════════════════════════════════════════════════
        # HIGHER MOMENTS
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'skewness':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            return returns.skew().clip(-3, 3)
        
        elif feature_type == 'kurtosis':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            return returns.kurtosis().clip(-5, 10)
        
        elif feature_type == 'downside_deviation':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            negative_returns = returns.clip(upper=0)
            return negative_returns.std() * np.sqrt(252)
        
        elif feature_type == 'up_down_capture':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(1.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            market_returns = returns.mean(axis=1)
            
            up_days = market_returns > 0
            down_days = market_returns < 0
            
            if up_days.sum() > 0 and down_days.sum() > 0:
                up_capture = returns.loc[up_days].mean() / market_returns.loc[up_days].mean()
                down_capture = returns.loc[down_days].mean() / market_returns.loc[down_days].mean()
                return (up_capture - down_capture).fillna(0)
            return pd.Series(0.0, index=tickers)
        
        elif feature_type == 'left_tail':
            period, quantile = spec[1], spec[2]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            es = returns.apply(lambda x: x.quantile(quantile))
            return -es
        
        # ═══════════════════════════════════════════════════════════════════
        # TREND QUALITY
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'trend_r2':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            
            def calc_r2(series):
                y = series.values
                x = np.arange(len(y))
                if len(y) < 5:
                    return 0
                corr = np.corrcoef(x, y)[0, 1]
                return corr ** 2 if not np.isnan(corr) else 0
            
            return prices.iloc[-period:].apply(calc_r2)
        
        elif feature_type == 'trend_slope':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            
            def calc_slope(series):
                y = series.values
                x = np.arange(len(y))
                if len(y) < 5:
                    return 0
                slope = np.polyfit(x, y, 1)[0]
                return slope / series.mean() * period
            
            return prices.iloc[-period:].apply(calc_slope)
        
        elif feature_type == 'trend_deviation':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            
            def calc_deviation(series):
                y = series.values
                x = np.arange(len(y))
                if len(y) < 5:
                    return 0
                slope, intercept = np.polyfit(x, y, 1)
                trend_line = slope * x + intercept
                deviation = (y[-1] - trend_line[-1]) / series.mean()
                return deviation
            
            return prices.iloc[-period:].apply(calc_deviation)
        
        elif feature_type == 'hurst_proxy':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            
            returns = prices.pct_change().iloc[-period:]
            
            def calc_hurst_proxy(col):
                if len(col.dropna()) < 20:
                    return 0.5
                r = col.cumsum()
                s = col.expanding().std()
                rs = (r.max() - r.min()) / s.iloc[-1] if s.iloc[-1] > 0 else 1
                h = np.log(rs) / np.log(len(col)) if rs > 0 else 0.5
                return min(max(h, 0), 1)
            
            return returns.apply(calc_hurst_proxy)
        
        # ═══════════════════════════════════════════════════════════════════
        # PRICE LEVEL
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'range_position':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            recent = prices.iloc[-period:]
            high = recent.max()
            low = recent.min()
            current = prices.iloc[-1]
            range_size = high - low
            position = (current - low) / range_size.replace(0, np.nan)
            return position.fillna(0.5)
        
        elif feature_type == 'high_proximity':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            high = recent.max()
            current = prices.iloc[-1]
            return current / high
        
        elif feature_type == 'breakout_strength':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            historical = prices.iloc[-(period + 1):-1]
            high = historical.max()
            current = prices.iloc[-1]
            breakout = (current / high) - 1
            return breakout.clip(0, 0.5)
        
        elif feature_type == 'support_distance':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            low = recent.min()
            current = prices.iloc[-1]
            return (current / low) - 1
        
        elif feature_type == 'resistance_distance':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            recent = prices.iloc[-period:]
            high = recent.max()
            current = prices.iloc[-1]
            return (high / current) - 1
        
        # ═══════════════════════════════════════════════════════════════════
        # CROSS-SECTIONAL
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'relative_strength':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.5, index=tickers)
            returns = (prices.iloc[-1] / prices.iloc[-period - 1]) - 1
            return returns.rank(pct=True)
        
        elif feature_type == 'relative_volatility':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.5, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            vol = returns.std()
            return vol.rank(pct=True)
        
        elif feature_type == 'excess_momentum':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            returns = (prices.iloc[-1] / prices.iloc[-period - 1]) - 1
            mean_return = returns.mean()
            return returns - mean_return
        
        elif feature_type == 'relative_value':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            ma = prices.iloc[-period:].mean()
            discount = prices.iloc[-1] / ma - 1
            return -(discount - discount.mean())
        
        # ═══════════════════════════════════════════════════════════════════
        # RISK-ADJUSTED
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'sharpe_ratio':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            mean_ret = returns.mean() * 252
            vol = returns.std() * np.sqrt(252)
            sharpe = mean_ret / vol.replace(0, np.nan)
            return sharpe.fillna(0).clip(-3, 3)
        
        elif feature_type == 'sortino_ratio':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            returns = prices.pct_change().iloc[-period:]
            mean_ret = returns.mean() * 252
            downside = returns.clip(upper=0).std() * np.sqrt(252)
            sortino = mean_ret / downside.replace(0, np.nan)
            return sortino.fillna(0).clip(-5, 5)
        
        elif feature_type == 'calmar_ratio':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            
            total_return = (prices.iloc[-1] / prices.iloc[-period]) - 1
            ann_return = (1 + total_return) ** (252 / period) - 1
            
            recent = prices.iloc[-period:]
            running_max = recent.cummax()
            max_dd = ((recent - running_max) / running_max).min().abs()
            
            calmar = ann_return / max_dd.replace(0, np.nan)
            return calmar.fillna(0).clip(-5, 5)
        
        elif feature_type == 'information_ratio':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            
            returns = prices.pct_change().iloc[-period:]
            market_returns = returns.mean(axis=1)
            
            excess_returns = returns.sub(market_returns, axis=0)
            mean_excess = excess_returns.mean()
            tracking_error = excess_returns.std()
            
            ir = mean_excess / tracking_error.replace(0, np.nan) * np.sqrt(252)
            return ir.fillna(0).clip(-3, 3)
        
        # ═══════════════════════════════════════════════════════════════════
        # VOLUME (optional)
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'volume_trend':
            if volume is None:
                return pd.Series(0.0, index=tickers)
            period = spec[1]
            if len(volume) < period:
                return pd.Series(0.0, index=tickers)
            
            recent = volume.iloc[-period:]
            first_half = recent.iloc[:period // 2].mean()
            second_half = recent.iloc[period // 2:].mean()
            trend = (second_half / first_half.replace(0, np.nan)) - 1
            return trend.fillna(0)
        
        elif feature_type == 'volume_ratio':
            if volume is None:
                return pd.Series(1.0, index=tickers)
            short, long = spec[1], spec[2]
            if len(volume) < long:
                return pd.Series(1.0, index=tickers)
            
            vol_short = volume.iloc[-short:].mean()
            vol_long = volume.iloc[-long:].mean()
            return (vol_short / vol_long.replace(0, np.nan)).fillna(1.0)
        
        elif feature_type == 'price_volume_corr':
            if volume is None:
                return pd.Series(0.0, index=tickers)
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            
            price_ret = prices.pct_change().iloc[-period:]
            vol_ret = volume.pct_change().iloc[-period:]
            
            def calc_corr(ticker):
                try:
                    return price_ret[ticker].corr(vol_ret[ticker])
                except:
                    return 0
            
            return pd.Series({t: calc_corr(t) for t in tickers})
        
        elif feature_type == 'volume_volatility':
            if volume is None:
                return pd.Series(0.0, index=tickers)
            period = spec[1]
            if len(volume) < period:
                return pd.Series(0.0, index=tickers)
            
            vol_pct = volume.pct_change().iloc[-period:]
            return vol_pct.std()
        
        # ═══════════════════════════════════════════════════════════════════
        # PATTERN
        # ═══════════════════════════════════════════════════════════════════
        
        elif feature_type == 'gap_reversal':
            period = spec[1]
            if len(prices) < period + 1:
                return pd.Series(0.0, index=tickers)
            
            returns = prices.pct_change().iloc[-period:]
            large_moves = returns.abs() > returns.std() * 2
            
            def gap_reversal_score(col):
                big_days = large_moves[col.name]
                if big_days.sum() == 0:
                    return 0
                
                reversals = 0
                total = 0
                
                for i, is_big in enumerate(big_days):
                    if is_big and i < len(col) - 1:
                        if np.sign(col.iloc[i]) != np.sign(col.iloc[i + 1]):
                            reversals += 1
                        total += 1
                
                return reversals / max(total, 1)
            
            return returns.apply(gap_reversal_score)
        
        elif feature_type == 'streak_strength':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            
            returns = prices.pct_change().iloc[-period:]
            
            def calc_streak(col):
                sign = np.sign(col.iloc[-1])
                streak = 1
                for i in range(len(col) - 2, -1, -1):
                    if np.sign(col.iloc[i]) == sign:
                        streak += 1
                    else:
                        break
                return sign * streak / period
            
            return returns.apply(calc_streak)
        
        elif feature_type == 'return_autocorr':
            period, lag = spec[1], spec[2]
            if len(prices) < period + lag:
                return pd.Series(0.0, index=tickers)
            
            returns = prices.pct_change().iloc[-period:]
            
            def calc_autocorr(col):
                try:
                    ac = col.autocorr(lag=lag)
                    return ac if not np.isnan(ac) else 0
                except:
                    return 0
            
            return returns.apply(calc_autocorr)
        
        elif feature_type == 'abs_return_momentum':
            period = spec[1]
            if len(prices) < period:
                return pd.Series(0.0, index=tickers)
            
            returns = prices.pct_change().iloc[-period:]
            return returns.abs().mean() * np.sqrt(252)
        
        # ═══════════════════════════════════════════════════════════════════
        # SMART MONEY CONCEPTS (SMC)
        # ═══════════════════════════════════════════════════════════════════
        elif feature_type == 'smc':
            if self.smc_features is None:
                return pd.Series(0.0, index=tickers)
            
            feature_name = spec[1]
            
            # SMC features need high/low/close/volume for each ticker
            # Calculate per ticker and aggregate
            results = {}
            for ticker in tickers:
                try:
                    ticker_prices = prices[ticker]
                    ticker_volume = volume[ticker] if volume is not None else pd.Series(0, index=ticker_prices.index)
                    
                    # For SMC, we need OHLC but we only have close prices
                    # Use close as proxy for high/low (conservative estimate)
                    smc_result = self.smc_features.calculate_all_features(
                        high=ticker_prices,
                        low=ticker_prices,
                        close=ticker_prices,
                        volume=ticker_volume
                    )
                    
                    if feature_name in smc_result:
                        feat = smc_result[feature_name]
                        results[ticker] = float(feat.iloc[-1]) if hasattr(feat, 'iloc') and len(feat) > 0 else 0.0
                    else:
                        results[ticker] = 0.0
                except:
                    results[ticker] = 0.0
            
            return pd.Series(results)
        
        # ═══════════════════════════════════════════════════════════════════
        # SUPPORT/RESISTANCE
        # ═══════════════════════════════════════════════════════════════════
        elif feature_type == 'sr':
            if self.sr_features is None:
                return pd.Series(0.0, index=tickers)
            
            feature_name = spec[1]
            
            # S/R features need high/low/close/volume for each ticker
            results = {}
            for ticker in tickers:
                try:
                    ticker_prices = prices[ticker]
                    ticker_volume = volume[ticker] if volume is not None else pd.Series(0, index=ticker_prices.index)
                    
                    # Use close as proxy for high/low
                    sr_result = self.sr_features.calculate_all_features(
                        high=ticker_prices,
                        low=ticker_prices,
                        close=ticker_prices,
                        volume=ticker_volume
                    )
                    
                    if feature_name in sr_result:
                        feat = sr_result[feature_name]
                        results[ticker] = float(feat.iloc[-1]) if hasattr(feat, 'iloc') and len(feat) > 0 else 0.0
                    else:
                        results[ticker] = 0.0
                except:
                    results[ticker] = 0.0
            
            return pd.Series(results)
        
        # ═══════════════════════════════════════════════════════════════════
        # OIL-SPECIFIC FEATURES
        # ═══════════════════════════════════════════════════════════════════
        elif feature_type == 'oil':
            if self.oil_features is None:
                return pd.Series(0.0, index=tickers)
            
            feature_name = spec[1]
            
            if len(prices) == 0:
                return pd.Series(0.0, index=tickers)
            
            # RC-4 FIX: The oil feature integration was broken — calculate_all_features()
            # expects (stock_prices: pd.Series, oil_market_data: OilMarketData) but was
            # being called with (prices, volume, start_date, end_date).
            #
            # Lazy-load oil market data (cached per date range)
            if not hasattr(self, '_oil_market_data') or self._oil_market_data is None:
                try:
                    from evolution.oil_specific_features import OilMarketData
                    # Use USO as WTI proxy, BNO as Brent proxy if available in prices
                    wti_proxy = prices['USO'] if 'USO' in prices.columns else prices.iloc[:, 0]
                    brent_proxy = prices['BNO'] if 'BNO' in prices.columns else wti_proxy
                    
                    self._oil_market_data = OilMarketData(
                        wti_price=wti_proxy,
                        brent_price=brent_proxy,
                        inventory=None,
                        refinery_utilization=None,
                        gasoline_price=None,
                        diesel_price=None,
                    )
                except Exception:
                    self._oil_market_data = None
            
            if self._oil_market_data is None:
                return pd.Series(0.0, index=tickers)
            
            # Calculate per-ticker oil features
            results = {}
            for ticker in tickers:
                try:
                    if ticker in prices.columns:
                        ticker_features = self.oil_features.calculate_all_features(
                            stock_prices=prices[ticker],
                            oil_market_data=self._oil_market_data
                        )
                        # Get the last value of the feature series
                        if feature_name in ticker_features:
                            feat_val = ticker_features[feature_name]
                            results[ticker] = feat_val.iloc[-1] if hasattr(feat_val, 'iloc') and len(feat_val) > 0 else 0.0
                        else:
                            results[ticker] = 0.0
                    else:
                        results[ticker] = 0.0
                except Exception:
                    results[ticker] = 0.0
            
            return pd.Series(results)
        
        # Default
        return pd.Series(0.0, index=tickers)


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FitnessResult:
    """Comprehensive fitness evaluation."""
    total: float
    sharpe_component: float
    return_component: float
    stability_component: float
    cost_penalty: float
    
    avg_sharpe: float
    sharpe_std: float
    avg_return: float
    return_std: float
    avg_turnover: float
    win_rate: float
    worst_period_return: float
    information_ratio: float
    
    n_periods: int
    period_results: List[Dict] = field(default_factory=list)


def calculate_fitness(
    period_results: List[Dict],
    benchmark_results: List[Dict],
    transaction_cost: float = 0.002,
) -> FitnessResult:
    """Calculate fitness centered on benchmark-relative performance."""
    
    if not period_results or not benchmark_results:
        return FitnessResult(
            total=0, sharpe_component=0, return_component=0,
            stability_component=0, cost_penalty=0, avg_sharpe=0,
            sharpe_std=0, avg_return=0, return_std=0, avg_turnover=0,
            win_rate=0, worst_period_return=-1, information_ratio=0,
            n_periods=0, period_results=[]
        )
    
    n = len(period_results)
    
    # Extract metrics
    sharpes = [r['sharpe_ratio'] for r in period_results]
    returns = [r['total_return'] for r in period_results]
    turnovers = [r.get('turnover', 0) for r in period_results]
    
    bench_returns = [b.get('total_return', 0) for b in benchmark_results[:n]]
    bench_sharpes = [b.get('sharpe_ratio', 0) for b in benchmark_results[:n]]
    
    # EXCESS metrics (strategy minus benchmark)
    excess_returns = [r - b for r, b in zip(returns, bench_returns)]
    excess_sharpes = [s - b for s, b in zip(sharpes, bench_sharpes)]
    
    avg_sharpe = np.mean(sharpes)
    sharpe_std = np.std(sharpes) if n > 1 else 0
    avg_return = np.mean(returns)
    return_std = np.std(returns) if n > 1 else 0
    avg_turnover = np.mean(turnovers)
    
    avg_excess_return = np.mean(excess_returns)
    avg_excess_sharpe = np.mean(excess_sharpes)
    
    # Win rate = fraction of periods beating benchmark
    win_rate = sum(1 for e in excess_returns if e > 0) / n
    worst_return = min(returns)
    worst_dd = max(r.get('max_drawdown', 0) for r in period_results)
    
    # Information ratio
    if n > 1:
        excess_std = np.std(excess_returns)
        ir = avg_excess_return / excess_std if excess_std > 0 else 0
    else:
        ir = 0
    
    # ═══════════════════════════════════════════════════════════════
    # SHARPE COMPONENT
    # Excess Sharpe component: 0.5 excess Sharpe = full score of 1.0
    # ═══════════════════════════════════════════════════════════════
    
    sharpe_component = np.clip(avg_excess_sharpe / 0.5, -1, 1)
    
    # Penalize variance in Sharpe
    sharpe_variance_penalty = min(sharpe_std / 1.0, 0.3)
    sharpe_component -= sharpe_variance_penalty
    
    # ═══════════════════════════════════════════════════════════════
    # RETURN COMPONENT
    # Excess return component: 10% annual excess = full score
    # ═══════════════════════════════════════════════════════════════
    
    return_component = np.clip(avg_excess_return / 0.10, -1, 1)
    
    # Information ratio bonus: IR of 0.5 = full bonus
    ir_bonus = np.clip(ir / 0.5, -0.5, 0.5)
    return_component = 0.7 * return_component + 0.3 * ir_bonus
    
    # ═══════════════════════════════════════════════════════════════
    # STABILITY COMPONENT
    # Penalties are explicitly weighted so their sum is bounded to 1.0,
    # preserving gradient signal across the full [-1, 1] range.
    # ═══════════════════════════════════════════════════════════════
    
    # Win rate baseline: 50% = 0, 100% = 1, 0% = -1
    stability_component = (win_rate - 0.5) * 2
    
    # Drawdown penalty (weight: 0.50)
    if worst_dd > 0.35:
        dd_penalty = 1.0   # near-disqualifying
    elif worst_dd > 0.25:
        dd_penalty = 0.6
    elif worst_dd > 0.15:
        dd_penalty = 0.2
    else:
        dd_penalty = 0.0
    
    # Worst period return penalty (weight: 0.25)
    return_penalty = 0.3 if worst_return < -0.20 else 0.0
    
    # Consistency penalty: high variance in excess returns (weight: 0.25)
    excess_return_std = np.std(excess_returns)
    consistency_penalty = np.clip(excess_return_std / 0.20, 0, 0.4)
    
    # Combine with explicit weights — max total penalty = 1.0
    stability_component -= (
        0.50 * dd_penalty +
        0.25 * return_penalty +
        0.25 * consistency_penalty
    )
    stability_component = np.clip(stability_component, -1, 1)
    
    # ═══════════════════════════════════════════════════════════════
    # COST PENALTY
    # ═══════════════════════════════════════════════════════════════
    
    annual_cost_drag = avg_turnover * 12 * transaction_cost * 2
    cost_penalty = np.clip(annual_cost_drag / 0.03, 0, 1)  # 3% drag = full penalty
    
    # ═══════════════════════════════════════════════════════════════
    # TOTAL - zero means "matches benchmark"
    # Positive = beats benchmark, Negative = loses to benchmark
    # ═══════════════════════════════════════════════════════════════
    
    total = (
        0.30 * sharpe_component +
        0.25 * return_component +
        0.30 * stability_component -
        0.15 * cost_penalty
    )
    
    # Period count adjustment (need enough data)
    if n < 4:
        total *= n / 4
    
    total = np.clip(total, -1, 1)
    
    return FitnessResult(
        total=total,
        sharpe_component=sharpe_component,
        return_component=return_component,
        stability_component=stability_component,
        cost_penalty=cost_penalty,
        avg_sharpe=avg_sharpe,
        sharpe_std=sharpe_std,
        avg_return=avg_return,
        return_std=return_std,
        avg_turnover=avg_turnover,
        win_rate=win_rate,
        worst_period_return=worst_return,
        information_ratio=ir,
        n_periods=n,
        period_results=period_results
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FITNESS FUNCTION v2 — RC-3: Recency Weighting + Universe-Adaptive Penalties
# ═══════════════════════════════════════════════════════════════════════════════

def _recency_weighted_mean(values: List[float], half_life_periods: int = 4) -> float:
    """
    Exponentially-weighted mean giving more weight to recent periods.
    
    Args:
        values: List of per-period values (oldest first, most recent last)
        half_life_periods: Number of periods for weight to decay by half
        
    Returns:
        Weighted mean
    """
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return values[0]
    
    decay = np.log(2) / max(half_life_periods, 1)
    weights = np.array([np.exp(-decay * (n - 1 - i)) for i in range(n)])
    weights /= weights.sum()
    
    return float(np.dot(values, weights))


def _get_drawdown_thresholds(universe_type: str) -> Dict[str, float]:
    """
    Get drawdown penalty thresholds based on universe characteristics.
    
    Oil microcaps routinely hit 40%+ drawdowns, so the thresholds must be
    calibrated differently than for large-cap S&P 500 stocks.
    
    Args:
        universe_type: One of 'general', 'oil_microcap', 'oil_largecap'
        
    Returns:
        Dict with 'severe', 'moderate', 'mild' thresholds
    """
    if universe_type == 'oil_microcap':
        return {
            'severe': 0.50,    # Was 0.35 — oil microcaps routinely hit 40%+
            'moderate': 0.35,  # Was 0.25
            'mild': 0.20,      # Was 0.15
        }
    elif universe_type == 'oil_largecap':
        return {
            'severe': 0.40,
            'moderate': 0.30,
            'mild': 0.20,
        }
    else:  # general / S&P 500
        return {
            'severe': 0.35,
            'moderate': 0.25,
            'mild': 0.15,
        }


def calculate_fitness_v2(
    period_results: List[Dict],
    benchmark_results: List[Dict],
    transaction_cost: float = 0.002,
    recency_half_life: int = 4,
    universe_type: str = 'general',
) -> FitnessResult:
    """
    Enhanced fitness with recency weighting and universe-adaptive penalties.
    
    Improvements over calculate_fitness():
    - Recency-weighted means instead of simple means (recent periods matter more)
    - Universe-adaptive drawdown thresholds (oil microcaps get wider thresholds)
    - Blended Sharpe + Calmar scoring for multi-objective optimization
    
    Args:
        period_results: List of per-period performance dicts
        benchmark_results: List of benchmark performance dicts
        transaction_cost: Per-trade transaction cost
        recency_half_life: Half-life for recency weighting (in periods)
        universe_type: 'general', 'oil_microcap', or 'oil_largecap'
        
    Returns:
        FitnessResult with fitness score and components
    """
    if not period_results or not benchmark_results:
        return FitnessResult(
            total=0, sharpe_component=0, return_component=0,
            stability_component=0, cost_penalty=0, avg_sharpe=0,
            sharpe_std=0, avg_return=0, return_std=0, avg_turnover=0,
            win_rate=0, worst_period_return=-1, information_ratio=0,
            n_periods=0, period_results=[]
        )
    
    n = len(period_results)
    
    # Extract metrics
    sharpes = [r['sharpe_ratio'] for r in period_results]
    returns = [r['total_return'] for r in period_results]
    turnovers = [r.get('turnover', 0) for r in period_results]
    
    bench_returns = [b.get('total_return', 0) for b in benchmark_results[:n]]
    bench_sharpes = [b.get('sharpe_ratio', 0) for b in benchmark_results[:n]]
    
    # EXCESS metrics (strategy minus benchmark)
    excess_returns = [r - b for r, b in zip(returns, bench_returns)]
    excess_sharpes = [s - b for s, b in zip(sharpes, bench_sharpes)]
    
    avg_sharpe = np.mean(sharpes)
    sharpe_std = np.std(sharpes) if n > 1 else 0
    avg_return = np.mean(returns)
    return_std = np.std(returns) if n > 1 else 0
    avg_turnover = np.mean(turnovers)
    
    # ═══════════════════════════════════════════════════════════════
    # RECENCY-WEIGHTED MEANS (RC-3 enhancement)
    # ═══════════════════════════════════════════════════════════════
    avg_excess_return = _recency_weighted_mean(excess_returns, recency_half_life)
    avg_excess_sharpe = _recency_weighted_mean(excess_sharpes, recency_half_life)
    
    # Win rate = fraction of periods beating benchmark
    win_rate = sum(1 for e in excess_returns if e > 0) / n
    worst_return = min(returns)
    worst_dd = max(r.get('max_drawdown', 0) for r in period_results)
    
    # Information ratio (recency-weighted)
    if n > 1:
        excess_std = np.std(excess_returns)
        ir = avg_excess_return / excess_std if excess_std > 0 else 0
    else:
        ir = 0
    
    # ═══════════════════════════════════════════════════════════════
    # SHARPE COMPONENT (same structure, recency-weighted input)
    # ═══════════════════════════════════════════════════════════════
    
    sharpe_component = np.clip(avg_excess_sharpe / 0.5, -1, 1)
    
    # Penalize variance in Sharpe
    sharpe_variance_penalty = min(sharpe_std / 1.0, 0.3)
    sharpe_component -= sharpe_variance_penalty
    
    # ═══════════════════════════════════════════════════════════════
    # CALMAR COMPONENT (NEW — blended with Sharpe for multi-objective)
    # ═══════════════════════════════════════════════════════════════
    
    # Calculate per-period Calmar ratios
    calmars = []
    bench_calmars = []
    for i in range(n):
        ret = returns[i]
        dd = max(period_results[i].get('max_drawdown', 0.01), 0.01)
        ann_ret = (1 + ret) ** 4 - 1  # Annualize from quarterly
        calmars.append(ann_ret / dd)
        
        b_ret = bench_returns[i]
        b_dd = max(benchmark_results[i].get('max_drawdown', 0.01), 0.01)
        b_ann_ret = (1 + b_ret) ** 4 - 1
        bench_calmars.append(b_ann_ret / b_dd)
    
    excess_calmars = [c - b for c, b in zip(calmars, bench_calmars)]
    avg_excess_calmar = _recency_weighted_mean(excess_calmars, recency_half_life)
    calmar_component = np.clip(avg_excess_calmar / 1.0, -1, 1)
    
    # ═══════════════════════════════════════════════════════════════
    # RETURN COMPONENT (recency-weighted)
    # ═══════════════════════════════════════════════════════════════
    
    return_component = np.clip(avg_excess_return / 0.10, -1, 1)
    
    # Information ratio bonus
    ir_bonus = np.clip(ir / 0.5, -0.5, 0.5)
    return_component = 0.7 * return_component + 0.3 * ir_bonus
    
    # ═══════════════════════════════════════════════════════════════
    # STABILITY COMPONENT — UNIVERSE-ADAPTIVE PENALTIES (RC-3)
    # ═══════════════════════════════════════════════════════════════
    
    stability_component = (win_rate - 0.5) * 2
    
    # Universe-adaptive drawdown thresholds
    dd_thresholds = _get_drawdown_thresholds(universe_type)
    
    if worst_dd > dd_thresholds['severe']:
        dd_penalty = 1.0
    elif worst_dd > dd_thresholds['moderate']:
        dd_penalty = 0.6
    elif worst_dd > dd_thresholds['mild']:
        dd_penalty = 0.2
    else:
        dd_penalty = 0.0
    
    # Worst period return penalty (also universe-adaptive)
    worst_return_threshold = -0.30 if universe_type.startswith('oil') else -0.20
    return_penalty = 0.3 if worst_return < worst_return_threshold else 0.0
    
    # Consistency penalty
    excess_return_std = np.std(excess_returns)
    consistency_threshold = 0.30 if universe_type.startswith('oil') else 0.20
    consistency_penalty = np.clip(excess_return_std / consistency_threshold, 0, 0.4)
    
    stability_component -= (
        0.50 * dd_penalty +
        0.25 * return_penalty +
        0.25 * consistency_penalty
    )
    stability_component = np.clip(stability_component, -1, 1)
    
    # ═══════════════════════════════════════════════════════════════
    # COST PENALTY
    # ═══════════════════════════════════════════════════════════════
    
    annual_cost_drag = avg_turnover * 12 * transaction_cost * 2
    cost_penalty = np.clip(annual_cost_drag / 0.03, 0, 1)
    
    # ═══════════════════════════════════════════════════════════════
    # TOTAL — Blended Sharpe + Calmar (multi-objective)
    # ═══════════════════════════════════════════════════════════════
    
    total = (
        0.25 * sharpe_component +      # Risk-adjusted returns (Sharpe)
        0.20 * calmar_component +       # Drawdown-adjusted returns (Calmar)
        0.20 * return_component +       # Absolute excess returns
        0.20 * stability_component -    # Consistency & win rate
        0.15 * cost_penalty             # Transaction costs
    )
    
    # Period count adjustment
    if n < 4:
        total *= n / 4
    
    total = np.clip(total, -1, 1)
    
    return FitnessResult(
        total=total,
        sharpe_component=sharpe_component,
        return_component=return_component,
        stability_component=stability_component,
        cost_penalty=cost_penalty,
        avg_sharpe=avg_sharpe,
        sharpe_std=sharpe_std,
        avg_return=avg_return,
        return_std=return_std,
        avg_turnover=avg_turnover,
        win_rate=win_rate,
        worst_period_return=worst_return,
        information_ratio=ir,
        n_periods=n,
        period_results=period_results
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GP STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass 
class GPStrategy:
    """GP Strategy with expression tree."""
    
    tree: Node
    top_pct: float = 20.0
    holding_period: int = 21
    execution_lag: int = 1
    
    generation: int = 0
    origin: str = "random"
    strategy_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    fitness: float = 0.0
    period_metrics: Optional[List[Dict]] = field(default=None, repr=False)
    
    def __post_init__(self):
        self.feature_lib = FeatureLibrary()
        self._last_positions: Optional[List[str]] = None
        self._position_ages: Dict[str, int] = {}

    def copy(self) -> 'GPStrategy':
        """Deep copy."""
        new_strategy = GPStrategy(
            tree=self.tree.copy(),
            top_pct=self.top_pct,
            holding_period=self.holding_period,
            execution_lag=self.execution_lag,
            generation=self.generation,
            origin=self.origin,
        )
        new_strategy.fitness = self.fitness
        if self.period_metrics:
            new_strategy.period_metrics = list(self.period_metrics)
        return new_strategy

    def score_stocks(
        self, 
        prices: pd.DataFrame,
        current_date_idx: int = -1
    ) -> pd.Series:
        """Score stocks with proper execution lag."""
        if prices.empty:
            return pd.Series(dtype=float)
        
        if self.execution_lag > 0:
            if current_date_idx == -1:
                effective_idx = len(prices) - 1 - self.execution_lag
            else:
                effective_idx = current_date_idx - self.execution_lag
            
            if effective_idx < self.feature_lib.max_lookback:
                return pd.Series(0.5, index=prices.columns)
            
            lagged_prices = prices.iloc[:effective_idx + 1]
        else:
            lagged_prices = prices
        
        features = self.feature_lib.compute_all(lagged_prices, lag=0)
        
        try:
            scores = self.tree.evaluate(features)
            
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                scores = (scores - min_s) / (max_s - min_s)
            else:
                scores = pd.Series(0.5, index=scores.index)
            
            return scores.fillna(0.5)
        except Exception:
            return pd.Series(0.5, index=prices.columns)
    
    def select_stocks(
        self, 
        prices: pd.DataFrame,
        current_positions: List[str] = None
    ) -> Tuple[List[str], float]:
        """Select stocks with turnover tracking."""
        scores = self.score_stocks(prices)
        if scores.empty:
            return [], 0.0
        
        n_select = max(1, int(len(scores) * self.top_pct / 100))
        top_stocks = scores.nlargest(n_select).index.tolist()
        
        if current_positions:
            current_set = set(current_positions)
            new_set = set(top_stocks)
            
            exited = len(current_set - new_set)
            entered = len(new_set - current_set)
            
            turnover = (exited + entered) / (2 * max(len(current_set), len(new_set), 1))
        else:
            turnover = 1.0
        
        return top_stocks, turnover
    
    def get_formula(self) -> str:
        """Get human-readable formula."""
        return self.tree.to_string()
    
    def complexity(self) -> int:
        """Tree complexity."""
        return self.tree.size()


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class WalkForwardEvaluator:
    """Walk-forward backtesting evaluator."""
    
    def __init__(
        self,
        prices: pd.DataFrame,
        periods: List[Tuple[str, str, str, str]],
        benchmark_results: List[Dict] = None,
        transaction_cost: float = 0.002,
        rebalance_frequency: int = 21,
        # Priority 1 & 2 components
        rebalancer = None,
        stop_manager = None,
        position_sizer = None,
        use_calmar_fitness: bool = False,
        # RC-3: Fitness v2 parameters
        universe_type: str = 'general',
        recency_half_life: int = 4,
        use_fitness_v2: bool = False,
        # RC-4: Oil reference panel — tickers to exclude from portfolio selection
        tradeable_tickers: Optional[List[str]] = None,
    ):
        self.prices = prices
        self.periods = periods
        self.benchmark_results = benchmark_results or []
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency
        
        # Priority 1 & 2 components
        self.rebalancer = rebalancer
        self.stop_manager = stop_manager
        self.position_sizer = position_sizer
        self.use_calmar_fitness = use_calmar_fitness
        
        # RC-3: Fitness v2 parameters
        self.universe_type = universe_type
        self.recency_half_life = recency_half_life
        self.use_fitness_v2 = use_fitness_v2
        
        # RC-4: If tradeable_tickers is set, only these tickers can be held
        # (reference panel tickers are used for features but not traded)
        self.tradeable_tickers = tradeable_tickers
    
    def evaluate_strategy(self, strategy: GPStrategy) -> FitnessResult:
        """Evaluate strategy across all periods."""
        period_results = []
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(self.periods):
            result = self._evaluate_single_period(strategy, test_start, test_end)
            
            if result is not None:
                period_results.append(result)
        
        # Use pre-computed benchmarks, sliced to match period count
        benchmarks = self.benchmark_results[:len(period_results)]
        
        # RC-3: Use fitness v2 with recency weighting + universe-adaptive penalties
        if self.use_fitness_v2:
            return calculate_fitness_v2(
                period_results,
                benchmarks,
                transaction_cost=self.transaction_cost,
                recency_half_life=self.recency_half_life,
                universe_type=self.universe_type,
            )
        # Use Calmar fitness if enabled (Priority 2.4)
        elif self.use_calmar_fitness:
            from evolution.fitness_calmar import calculate_calmar_fitness
            return calculate_calmar_fitness(
                period_results,
                benchmarks,
                target_turnover=0.5
            )
        else:
            return calculate_fitness(
                period_results,
                benchmarks,
                self.transaction_cost
            )
    
    def _evaluate_single_period(
        self,
        strategy: GPStrategy,
        test_start: str,
        test_end: str,
    ) -> Optional[Dict]:  # Remove benchmark_type param, return single Dict
        """Evaluate on a single test period."""

        test_start_dt = pd.Timestamp(test_start)
        test_end_dt = pd.Timestamp(test_end)
        
        mask = self.prices.index <= test_end_dt
        available_prices = self.prices.loc[mask]
        
        test_mask = (available_prices.index >= test_start_dt)
        test_indices = available_prices.index[test_mask]
        
        if len(test_indices) < 5:
            return None
        
        portfolio_returns = []
        turnovers = []
        current_positions = []
        
        for i, date in enumerate(test_indices):
            rebalance_today = (i == 0 or i % self.rebalance_frequency == 0)
            
            if rebalance_today:
                date_idx = available_prices.index.get_loc(date)
                prices_to_date = available_prices.iloc[:date_idx + 1]
                
                # RC-4: Score ALL tickers (including reference panel for cross-sectional context)
                # but only select from tradeable tickers for the portfolio
                if self.tradeable_tickers:
                    # Score using full universe (reference panel provides context)
                    scores = strategy.score_stocks(prices_to_date)
                    # Filter to only tradeable tickers
                    tradeable_in_prices = [t for t in self.tradeable_tickers if t in scores.index]
                    if tradeable_in_prices:
                        tradeable_scores = scores[tradeable_in_prices]
                        n_select = max(1, int(len(tradeable_scores) * strategy.top_pct / 100))
                        new_positions = tradeable_scores.nlargest(n_select).index.tolist()
                        
                        if current_positions:
                            current_set = set(current_positions)
                            new_set = set(new_positions)
                            exited = len(current_set - new_set)
                            entered = len(new_set - current_set)
                            turnover = (exited + entered) / (2 * max(len(current_set), len(new_set), 1))
                        else:
                            turnover = 1.0
                    else:
                        new_positions = []
                        turnover = 0.0
                else:
                    new_positions, turnover = strategy.select_stocks(
                        prices_to_date,
                        current_positions
                    )
                
                turnovers.append(turnover)
                current_positions = new_positions
            
            # Calculate return for this day
            if current_positions and i > 0:
                prev_date = test_indices[i-1]
                day_return = (
                    available_prices.loc[date, current_positions] / 
                    available_prices.loc[prev_date, current_positions] - 1
                ).mean()
                
                # Deduct transaction costs on rebalance
                if rebalance_today and turnovers:
                    cost = turnovers[-1] * self.transaction_cost * 2
                    day_return -= cost
                
                portfolio_returns.append(day_return)
        
        if not portfolio_returns:
            return None
        
        returns_series = pd.Series(portfolio_returns)
        cumulative = (1 + returns_series).cumprod()
        
        total_return = cumulative.iloc[-1] - 1
        
        if returns_series.std() > 0:
            sharpe = (returns_series.mean() / returns_series.std()) * np.sqrt(252)
        else:
            sharpe = 0
        
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = abs(drawdown.min())
        
        avg_turnover = np.mean(turnovers) if turnovers else 0
        
        result = {
            'period_start': test_start,   # ADD
            'period_end': test_end,       # ADD
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'turnover': avg_turnover,
            'n_days': len(portfolio_returns),
        }
        
        return result
    
    def to_config(self) -> Dict[str, Any]:
        """Serialize evaluator to picklable config dict."""
        return {
            'prices': self.prices,
            'periods': self.periods,
            'benchmark_results': self.benchmark_results,
            'transaction_cost': self.transaction_cost,
            'rebalance_frequency': self.rebalance_frequency,
            'rebalancer': self.rebalancer,
            'stop_manager': self.stop_manager,
            'position_sizer': self.position_sizer,
            'use_calmar_fitness': self.use_calmar_fitness,
            # RC-3
            'universe_type': self.universe_type,
            'recency_half_life': self.recency_half_life,
            'use_fitness_v2': self.use_fitness_v2,
            # RC-4
            'tradeable_tickers': self.tradeable_tickers,
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'WalkForwardEvaluator':
        """Deserialize evaluator from config dict."""
        return cls(**config)


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