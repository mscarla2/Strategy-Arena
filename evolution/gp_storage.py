# evolution/gp_storage.py
"""
Database storage for GP strategies and their results.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

from .gp import GPStrategy, Node, FeatureNode, ConstantNode, BinaryOpNode, UnaryOpNode, ConditionalNode


# ═══════════════════════════════════════════════════════════════════════════════
# TREE SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def tree_to_dict(node: Node) -> Dict:
    """Serialize expression tree to dictionary."""
    if isinstance(node, FeatureNode):
            return {'type': 'feature', 'name': node.feature_name}
    elif isinstance(node, ConstantNode):
        return {'type': 'constant', 'value': node.value}
    elif isinstance(node, BinaryOpNode):
        return {
            'type': 'binary',
            'op': node.op,
            'left': tree_to_dict(node.left),
            'right': tree_to_dict(node.right)
        }
    elif isinstance(node, UnaryOpNode):
        return {
            'type': 'unary',
            'op': node.op,
            'child': tree_to_dict(node.child)
        }
    elif isinstance(node, ConditionalNode):
        return {
            'type': 'conditional',
            'condition': tree_to_dict(node.condition),
            'if_true': tree_to_dict(node.if_true),    # was: then_branch
            'if_false': tree_to_dict(node.if_false)   # was: else_branch
        }
    else:
        raise ValueError(f"Unknown node type: {type(node)}")


def dict_to_tree(d: Dict) -> Node:
    """Deserialize dictionary to expression tree."""
    node_type = d['type']
    
    if node_type == 'feature':
        return FeatureNode(d['name'])
    elif node_type == 'constant':
        return ConstantNode(d['value'])
    elif node_type == 'binary':
        return BinaryOpNode(
            d['op'],
            dict_to_tree(d['left']),
            dict_to_tree(d['right'])
        )
    elif node_type == 'unary':
        return UnaryOpNode(d['op'], dict_to_tree(d['child']))
    elif node_type == 'conditional':
        return ConditionalNode(
            dict_to_tree(d['condition']),
            dict_to_tree(d['if_true']),    # was: then_branch
            dict_to_tree(d['if_false'])    # was: else_branch
        )
    else:
        raise ValueError(f"Unknown node type: {node_type}")


# ═══════════════════════════════════════════════════════════════════════════════
# GP DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

class GPDatabase:
    """SQLite database for storing GP evolution results."""
    
    def __init__(self, db_path: str = "data/gp_strategies.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                -- Evolution runs
                CREATE TABLE IF NOT EXISTS evolution_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    config TEXT,  -- JSON
                    generations_completed INTEGER,
                    best_fitness REAL,
                    best_strategy_id TEXT,
                    notes TEXT
                );
                
                -- GP Strategies
                CREATE TABLE IF NOT EXISTS gp_strategies (
                    strategy_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    formula TEXT,
                    tree_json TEXT,  -- Serialized tree
                    top_pct REAL,
                    complexity INTEGER,
                    generation INTEGER,
                    origin TEXT,
                    fitness REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id)
                );
                
                -- Period-by-period results
                CREATE TABLE IF NOT EXISTS gp_period_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT,
                    run_id TEXT,
                    period_start TEXT,
                    period_end TEXT,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    num_trades INTEGER,
                    win_rate REAL,
                    FOREIGN KEY (strategy_id) REFERENCES gp_strategies(strategy_id)
                );
                
                -- Benchmark results for comparison
                CREATE TABLE IF NOT EXISTS benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    benchmark_name TEXT,  -- 'buy_hold', 'spy', 'equal_weight'
                    period_start TEXT,
                    period_end TEXT,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    UNIQUE(run_id, benchmark_name, period_start, period_end)
                );
                
                -- Generation statistics
                CREATE TABLE IF NOT EXISTS generation_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    generation INTEGER,
                    avg_fitness REAL,
                    max_fitness REAL,
                    min_fitness REAL,
                    avg_sharpe REAL,
                    avg_return REAL,
                    avg_complexity REAL,
                    diversity_score REAL,
                    FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id)
                );
                
                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_strategies_fitness ON gp_strategies(fitness DESC);
                CREATE INDEX IF NOT EXISTS idx_strategies_run ON gp_strategies(run_id);
                CREATE INDEX IF NOT EXISTS idx_period_results_strategy ON gp_period_results(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_benchmarks_run ON benchmarks(run_id);
            """)
    
    # ───────────────────────────────────────────────────────────────────────────
    # Evolution Run Management
    # ───────────────────────────────────────────────────────────────────────────
    
    def create_run(self, run_id: str, config: Dict) -> str:
        """Create a new evolution run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO evolution_runs (run_id, started_at, config)
                VALUES (?, ?, ?)
            """, (run_id, datetime.now(), json.dumps(config)))
        return run_id
    
    def get_benchmarks_for_strategy(self, strategy_id: str) -> Dict[str, List[Dict]]:
        """Get benchmark results for a strategy's run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # First get the run_id for this strategy
            row = conn.execute("""
                SELECT run_id FROM gp_strategies WHERE strategy_id = ?
            """, (strategy_id,)).fetchone()
            
            if not row:
                return {}
            
            run_id = row['run_id']
        
        return self.get_benchmarks_for_run(run_id)
    def complete_run(self, run_id: str, generations: int, best_fitness: float, best_strategy_id: str):
        """Mark run as complete."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE evolution_runs
                SET completed_at = ?, generations_completed = ?, 
                    best_fitness = ?, best_strategy_id = ?
                WHERE run_id = ?
            """, (datetime.now(), generations, best_fitness, best_strategy_id, run_id))
    
    def get_runs(self, limit: int = 20) -> List[Dict]:
        """Get recent evolution runs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM evolution_runs
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [dict(row) for row in rows]
    
    # ───────────────────────────────────────────────────────────────────────────
    # Strategy Storage
    # ───────────────────────────────────────────────────────────────────────────
    
    def save_strategy(self, strategy: GPStrategy, run_id: str):
        """Save a GP strategy."""
        tree_json = json.dumps(tree_to_dict(strategy.tree))
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO gp_strategies
                (strategy_id, run_id, formula, tree_json, top_pct, complexity,
                 generation, origin, fitness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy.strategy_id,
                run_id,
                strategy.get_formula(),
                tree_json,
                strategy.top_pct,
                strategy.complexity(),
                strategy.generation,
                strategy.origin,
                strategy.fitness
            ))
    
    def save_period_results(self, strategy_id: str, run_id: str, metrics: List[Dict]):
        """Save period-by-period results for a strategy."""
        with sqlite3.connect(self.db_path) as conn:
            for m in metrics:
                conn.execute("""
                    INSERT INTO gp_period_results
                    (strategy_id, run_id, period_start, period_end, total_return,
                     sharpe_ratio, max_drawdown, num_trades, win_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    strategy_id,
                    run_id,
                    m.get('test_start', m.get('period_start', '')),
                    m.get('test_end', m.get('period_end', '')),
                    m.get('total_return', 0),
                    m.get('sharpe_ratio', 0),
                    m.get('max_drawdown', 0),
                    m.get('num_trades', 0),
                    m.get('win_rate', 0)
                ))
    
    def load_strategy(self, strategy_id: str) -> Optional[GPStrategy]:
        """Load a strategy by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("""
                SELECT * FROM gp_strategies WHERE strategy_id = ?
            """, (strategy_id,)).fetchone()
        
        if not row:
            return None
        
        tree = dict_to_tree(json.loads(row['tree_json']))
        strategy = GPStrategy(
            tree=tree,
            top_pct=row['top_pct'],
            generation=row['generation'],
            origin=row['origin']
        )
        strategy.strategy_id = row['strategy_id']
        strategy.fitness = row['fitness']
        
        return strategy
    
    def get_top_strategies(self, limit: int = 20, min_fitness: float = 0) -> List[Dict]:
        """Get top strategies by fitness."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT s.*, r.started_at as run_date
                FROM gp_strategies s
                LEFT JOIN evolution_runs r ON s.run_id = r.run_id
                WHERE s.fitness >= ?
                ORDER BY s.fitness DESC
                LIMIT ?
            """, (min_fitness, limit)).fetchall()
        return [dict(row) for row in rows]
    
    def get_strategy_results(self, strategy_id: str) -> List[Dict]:
        """Get period results for a strategy."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM gp_period_results
                WHERE strategy_id = ?
                ORDER BY period_start
            """, (strategy_id,)).fetchall()
        return [dict(row) for row in rows]
    
    # ───────────────────────────────────────────────────────────────────────────
    # Benchmarks
    # ───────────────────────────────────────────────────────────────────────────
    
    def save_benchmark(self, run_id: str, name: str, period_start: str, 
                       period_end: str, metrics: Dict):
        """Save benchmark results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO benchmarks
                (run_id, benchmark_name, period_start, period_end,
                 total_return, sharpe_ratio, max_drawdown)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, name, period_start, period_end,
                metrics.get('total_return', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('max_drawdown', 0)
            ))
    
    def get_benchmarks(self, run_id: str) -> List[Dict]:
        """Get benchmark results for a run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM benchmarks WHERE run_id = ?
                ORDER BY benchmark_name, period_start
            """, (run_id,)).fetchall()
        return [dict(row) for row in rows]
    
    def get_benchmarks_for_run(self, run_id: str) -> Dict[str, List[Dict]]:
        """Get benchmark results grouped by benchmark name."""
        benchmarks = self.get_benchmarks(run_id)
        
        result = {}
        for b in benchmarks:
            name = b['benchmark_name']
            if name not in result:
                result[name] = []
            result[name].append({
                'period_start': b['period_start'],
                'period_end': b['period_end'],
                'total_return': b['total_return'],
                'sharpe_ratio': b['sharpe_ratio'],
                'max_drawdown': b['max_drawdown'],
            })
        
        return result
    
    # ───────────────────────────────────────────────────────────────────────────
    # Generation Stats
    # ───────────────────────────────────────────────────────────────────────────
    
    def save_generation_stats(self, run_id: str, stats: Dict):
        """Save generation statistics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO generation_stats
                (run_id, generation, avg_fitness, max_fitness, min_fitness,
                 avg_sharpe, avg_return, avg_complexity, diversity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                stats.get('generation', 0),
                stats.get('avg_fitness', 0),
                stats.get('max_fitness', 0),
                stats.get('min_fitness', 0),
                stats.get('avg_sharpe', 0),
                stats.get('avg_return', 0),
                stats.get('avg_complexity', 0),
                stats.get('diversity_score', 0)
            ))
    
    def get_generation_stats(self, run_id: str) -> List[Dict]:
        """Get all generation stats for a run."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM generation_stats
                WHERE run_id = ?
                ORDER BY generation
            """, (run_id,)).fetchall()
        return [dict(row) for row in rows]
    
    # ───────────────────────────────────────────────────────────────────────────
    # Analytics
    # ───────────────────────────────────────────────────────────────────────────
    
    def get_strategy_vs_benchmarks(self, strategy_id: str) -> Dict:
        """Compare strategy performance against benchmarks."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get strategy results
            strategy_results = conn.execute("""
                SELECT * FROM gp_period_results WHERE strategy_id = ?
            """, (strategy_id,)).fetchall()
            
            if not strategy_results:
                return {}
            
            # Get run_id
            run_id = strategy_results[0]['run_id']
            
            # Get benchmarks
            benchmarks = conn.execute("""
                SELECT * FROM benchmarks WHERE run_id = ?
            """, (run_id,)).fetchall()
        
        # Aggregate
        strategy_returns = [r['total_return'] for r in strategy_results]
        strategy_sharpes = [r['sharpe_ratio'] for r in strategy_results]
        
        result = {
            'strategy': {
                'avg_return': sum(strategy_returns) / len(strategy_returns) if strategy_returns else 0,
                'avg_sharpe': sum(strategy_sharpes) / len(strategy_sharpes) if strategy_sharpes else 0,
                'periods': len(strategy_results),
                'positive_periods': sum(1 for r in strategy_returns if r > 0),
            },
            'benchmarks': {}
        }
        
        # Group benchmark results
        for b in benchmarks:
            name = b['benchmark_name']
            if name not in result['benchmarks']:
                result['benchmarks'][name] = {'returns': [], 'sharpes': []}
            result['benchmarks'][name]['returns'].append(b['total_return'])
            result['benchmarks'][name]['sharpes'].append(b['sharpe_ratio'])
        
        # Compute averages
        for name, data in result['benchmarks'].items():
            result['benchmarks'][name] = {
                'avg_return': sum(data['returns']) / len(data['returns']) if data['returns'] else 0,
                'avg_sharpe': sum(data['sharpes']) / len(data['sharpes']) if data['sharpes'] else 0,
            }
        
        return result