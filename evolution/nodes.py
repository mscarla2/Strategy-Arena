# evolution/nodes.py
"""
Expression tree nodes for GP strategies.

Contains: Node (ABC), FeatureNode, ConstantNode, BinaryOpNode, UnaryOpNode, ConditionalNode
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from abc import ABC, abstractmethod


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
