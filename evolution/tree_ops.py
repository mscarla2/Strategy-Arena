# evolution/tree_ops.py
"""
Tree generation and genetic operators for GP expression trees.

Contains: TreeGenerator, GPOperators
"""

import random
from typing import List

from evolution.nodes import (
    Node, FeatureNode, ConstantNode, BinaryOpNode, UnaryOpNode, ConditionalNode
)


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
    
    def crossover(self, parent1, parent2):
        """Subtree crossover between two strategies."""
        # Import here to avoid circular imports
        from evolution.strategy import GPStrategy
        
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
    
    def mutate(self, parent):
        """Mutate a strategy's tree."""
        # Import here to avoid circular imports
        from evolution.strategy import GPStrategy
        
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
