#!/usr/bin/env python3
"""
ALPHAGENE Strategy Picker - Interactive Terminal UI
Browse, inspect, compare, and visualize evolved strategies.
Supports both GA and GP strategies with benchmark comparisons.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import DATA_DIR

# Try to import both storage systems
try:
    from evolution.storage import StrategyStorage
    HAS_GA_STORAGE = True
except ImportError:
    HAS_GA_STORAGE = False

try:
    from evolution.gp_storage import GPDatabase
    HAS_GP_STORAGE = True
except ImportError:
    HAS_GP_STORAGE = False

# Lazy import for charts (optional dependency)
def get_charts():
    try:
        from ui.charts import AlphaGeneCharts
        return AlphaGeneCharts()
    except ImportError:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# THEME & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

class Theme:
    """Terminal color theme."""
    PRIMARY = "\033[38;5;75m"
    SECONDARY = "\033[38;5;245m"
    SUCCESS = "\033[38;5;78m"
    WARNING = "\033[38;5;220m"
    ERROR = "\033[38;5;203m"
    ACCENT = "\033[38;5;183m"
    MUTED = "\033[38;5;240m"
    WHITE = "\033[38;5;255m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"
    
    GENE_COLORS = {
        "momentum": "\033[38;5;75m",
        "value": "\033[38;5;78m",
        "quality": "\033[38;5;183m",
        "volatility": "\033[38;5;220m",
        "mean_reversion": "\033[38;5;203m",
        "size": "\033[38;5;117m",
        "growth": "\033[38;5;114m",
        "dividend": "\033[38;5;218m",
    }
    
    @classmethod
    def get_gene_color(cls, gene_name: str) -> str:
        gene_lower = gene_name.lower()
        for key, color in cls.GENE_COLORS.items():
            if key in gene_lower:
                return color
        return cls.ACCENT


def style(text: str, *styles) -> str:
    if not sys.stdout.isatty():
        return text
    style_str = "".join(styles)
    return f"{style_str}{text}{Theme.RESET}"


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def print_header():
    header = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║     █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗  ██████╗ ███████╗  ║
    ║    ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗██╔════╝ ██╔════╝  ║
    ║    ███████║██║     ██████╔╝███████║███████║██║  ███╗█████╗    ║
    ║    ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║██║   ██║██╔══╝    ║
    ║    ██║  ██║███████╗██║     ██║  ██║██║  ██║╚██████╔╝███████╗  ║
    ║    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝  ║
    ║                                                               ║
    ║              🧬  Strategy Picker  v2.1  🧬                    ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(style(header, Theme.PRIMARY))


def print_section(title: str, icon: str = "═"):
    width = 60
    print()
    print(f"  {style(icon, Theme.ACCENT)}  {style(title, Theme.PRIMARY, Theme.BOLD)}")
    print(f"  {style('─' * width, Theme.MUTED)}")


def format_number(value: float, precision: int = 2, as_percent: bool = False) -> str:
    if pd.isna(value) or value is None:
        return style("N/A", Theme.MUTED)
    
    if as_percent:
        formatted = f"{value * 100:.{precision}f}%"
    else:
        formatted = f"{value:.{precision}f}"
    
    if value > 0:
        return style(formatted, Theme.SUCCESS)
    elif value < 0:
        return style(formatted, Theme.ERROR)
    else:
        return style(formatted, Theme.MUTED)


def format_metric(label: str, value: float, precision: int = 2, 
                  as_percent: bool = False, width: int = 20) -> str:
    label_str = style(f"{label}:", Theme.SECONDARY)
    value_str = format_number(value, precision, as_percent)
    return f"  {label_str:<{width+10}} {value_str}"


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY PICKER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class StrategyPicker:
    """Interactive strategy browser and analyzer."""
    
    def __init__(self, db_path: str = None, gp_db_path: str = None):
        """Initialize the strategy picker."""
        # GA storage
        if HAS_GA_STORAGE:
            self.ga_storage = StrategyStorage(db_path)
        else:
            self.ga_storage = None
        
        # GP storage
        if HAS_GP_STORAGE:
            gp_path = gp_db_path or "data/gp_strategies.db"
            self.gp_db = GPDatabase(gp_path)
        else:
            self.gp_db = None
        
        self.current_strategy: Optional[Dict] = None
        self.current_strategy_type: str = "gp"  # 'ga' or 'gp'
        self.comparison_list: List[Tuple[str, str]] = []  # (strategy_id, type)
        self.charts = get_charts()
        
        self._strategy_cache: Dict[str, Dict] = {}
    
    def get_gp_strategy_by_id(self, strategy_id: str) -> Optional[Dict]:
        """Fetch GP strategy details by ID."""
        if not self.gp_db:
            return None
        
        cache_key = f"gp_{strategy_id}"
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]
        
        try:
            strategies = self.gp_db.get_top_strategies(limit=1000)
            match = [s for s in strategies if s['strategy_id'] == strategy_id]
            
            if not match:
                return None
            
            row = match[0]
            
            # Get period results
            results = self.gp_db.get_strategy_results(strategy_id)
            if results:
                row['period_results'] = results
                
                # Calculate yearly returns
                yearly = {}
                for r in results:
                    period = r.get('period_start', '')
                    if period:
                        year = period[:4]
                        if year not in yearly:
                            yearly[year] = []
                        yearly[year].append(r.get('total_return', 0))
                
                row['yearly_returns'] = {y: np.mean(r) for y, r in yearly.items()}
            
            # Get benchmarks
            benchmarks = self.gp_db.get_benchmarks_for_strategy(strategy_id)
            row['benchmarks'] = benchmarks
            
            row['type'] = 'gp'
            self._strategy_cache[cache_key] = row
            return row
            
        except Exception as e:
            print(f"\n  {style(f'Error fetching GP strategy: {e}', Theme.ERROR)}")
            return None
    
    def get_ga_strategy_by_id(self, strategy_id: str) -> Optional[Dict]:
        """Fetch GA strategy details by ID."""
        if not self.ga_storage:
            return None
        
        cache_key = f"ga_{strategy_id}"
        if cache_key in self._strategy_cache:
            return self._strategy_cache[cache_key]
        
        try:
            top = self.ga_storage.get_top_strategies(n=1000, min_tests=0)
            match = top[top['strategy_id'] == strategy_id]
            
            if match.empty:
                return None
            
            row = match.iloc[0].to_dict()
            
            history = self.ga_storage.get_strategy_history(strategy_id)
            if not history.empty:
                row['history'] = history.to_dict('records')
                
                yearly = {}
                for _, h in history.iterrows():
                    period = h.get('test_period', '')
                    if period:
                        year = period[:4]
                        if year not in yearly:
                            yearly[year] = []
                        yearly[year].append(h.get('total_return', 0))
                
                row['yearly_returns'] = {y: np.mean(r) for y, r in yearly.items()}
            
            try:
                strategy_obj = self.ga_storage.reconstruct_strategy(strategy_id)
                if strategy_obj:
                    row['genes'] = {g.name: g.get_params() for g in strategy_obj.genes}
            except:
                row['genes'] = {}
            
            row['type'] = 'ga'
            self._strategy_cache[cache_key] = row
            return row
            
        except Exception as e:
            print(f"\n  {style(f'Error fetching GA strategy: {e}', Theme.ERROR)}")
            return None
    
    def view_top_gp_strategies(self, n: int = 10) -> List[Dict]:
        """Display top GP strategies."""
        print_section("TOP GP STRATEGIES (Expression Trees)", "🧬")
        
        if not self.gp_db:
            print(f"\n  {style('GP database not available.', Theme.WARNING)}")
            return []
        
        try:
            strategies = self.gp_db.get_top_strategies(limit=n)
        except Exception as e:
            print(f"\n  {style(f'Error loading strategies: {e}', Theme.ERROR)}")
            return []
        
        if not strategies:
            print(f"\n  {style('No GP strategies found.', Theme.WARNING)}")
            print(f"  {style('Run arena_runner_v3.py first to evolve strategies.', Theme.MUTED)}")
            return []
        
        header = (
            f"  {style('#', Theme.MUTED):>4}  "
            f"{style('ID', Theme.SECONDARY):<10}  "
            f"{style('Fitness', Theme.SECONDARY):>8}  "
            f"{style('Complexity', Theme.SECONDARY):>10}  "
            f"{style('Formula', Theme.SECONDARY):<40}"
        )
        print(f"\n{header}")
        print(f"  {style('─' * 80, Theme.MUTED)}")
        
        for i, row in enumerate(strategies, 1):
            sid = row.get('strategy_id', '')[:8]
            fitness = row.get('fitness', 0)
            complexity = row.get('complexity', 0)
            formula = str(row.get('formula', ''))[:38]
            
            if i == 1:
                sid = style(sid, Theme.SUCCESS, Theme.BOLD)
            
            line = (
                f"  {style(str(i), Theme.PRIMARY):>4}  "
                f"{style(row.get('strategy_id', '')[:8], Theme.MUTED):<10}  "
                f"{format_number(fitness):>8}  "
                f"{style(str(complexity), Theme.SECONDARY):>10}  "
                f"{style(formula, Theme.MUTED):<40}"
            )
            print(line)
        
        print(f"\n  {style(f'Showing top {len(strategies)} GP strategies', Theme.MUTED)}")
        return strategies
    
    def view_top_strategies(self, n: int = 10, min_tests: int = 1) -> List[Dict]:
        """Display top GA strategies in a table."""
        print_section("TOP GA STRATEGIES (Gene-based)", "🏆")
        
        if not self.ga_storage:
            print(f"\n  {style('GA database not available.', Theme.WARNING)}")
            return []
        
        try:
            df = self.ga_storage.get_top_strategies(n=n, min_tests=min_tests)
        except Exception as e:
            print(f"\n  {style(f'Error loading strategies: {e}', Theme.ERROR)}")
            return []
        
        if df.empty:
            print(f"\n  {style('No strategies found.', Theme.WARNING)}")
            print(f"  {style('Run the arena first to evolve strategies.', Theme.MUTED)}")
            return []
        
        header = (
            f"  {style('#', Theme.MUTED):>4}  "
            f"{style('ID', Theme.SECONDARY):<12}  "
            f"{style('Name', Theme.SECONDARY):<25}  "
            f"{style('Sharpe', Theme.SECONDARY):>8}  "
            f"{style('Return', Theme.SECONDARY):>10}  "
            f"{style('MaxDD', Theme.SECONDARY):>8}  "
            f"{style('Tests', Theme.SECONDARY):>6}"
        )
        print(f"\n{header}")
        print(f"  {style('─' * 85, Theme.MUTED)}")
        
        strategies = []
        for i, row in df.iterrows():
            idx = len(strategies) + 1
            sid = row.get('strategy_id', '')[:10]
            name = str(row.get('name', 'Unknown'))[:24]
            sharpe = row.get('avg_sharpe', 0)
            ret = row.get('avg_return', 0)
            maxdd = row.get('avg_max_drawdown', 0)
            tests = int(row.get('test_count', 0))
            
            if idx == 1:
                name = style(name, Theme.SUCCESS, Theme.BOLD)
            
            line = (
                f"  {style(str(idx), Theme.PRIMARY):>4}  "
                f"{style(sid, Theme.MUTED):<12}  "
                f"{name:<25}  "
                f"{format_number(sharpe):>8}  "
                f"{format_number(ret, as_percent=True):>10}  "
                f"{format_number(maxdd, as_percent=True):>8}  "
                f"{style(str(tests), Theme.SECONDARY):>6}"
            )
            print(line)
            strategies.append(row.to_dict())
        
        print(f"\n  {style(f'Showing top {len(strategies)} of {len(df)} strategies', Theme.MUTED)}")
        return strategies
    
    def inspect_gp_strategy(self, strategy_id: str = None):
        """Show detailed view of a GP strategy with benchmark comparison."""
        if strategy_id:
            strategy = self.get_gp_strategy_by_id(strategy_id)
        elif self.current_strategy and self.current_strategy_type == 'gp':
            strategy = self.current_strategy
        else:
            print(f"\n  {style('No GP strategy selected.', Theme.WARNING)}")
            return
        
        if not strategy:
            print(f"\n  {style('Strategy not found.', Theme.ERROR)}")
            return
        
        self.current_strategy = strategy
        self.current_strategy_type = 'gp'
        
        print_section(f"GP STRATEGY: {strategy.get('strategy_id', 'Unknown')}", "🔬")
        
        # Basic info
        print(f"\n  {style('IDENTIFICATION', Theme.ACCENT, Theme.BOLD)}")
        print(f"  {style('ID:', Theme.SECONDARY)} {style(strategy.get('strategy_id', 'N/A'), Theme.PRIMARY)}")
        print(f"  {style('Fitness:', Theme.SECONDARY)} {format_number(strategy.get('fitness', 0))}")
        print(f"  {style('Complexity:', Theme.SECONDARY)} {strategy.get('complexity', 0)} nodes")
        print(f"  {style('Top %:', Theme.SECONDARY)} {strategy.get('top_pct', 0)}%")
        print(f"  {style('Generation:', Theme.SECONDARY)} {strategy.get('generation', 'N/A')}")
        print(f"  {style('Origin:', Theme.SECONDARY)} {strategy.get('origin', 'N/A')}")
        
        # Formula
        print(f"\n  {style('FORMULA', Theme.ACCENT, Theme.BOLD)}")
        formula = strategy.get('formula', 'N/A')
        # Word wrap long formulas
        if len(formula) > 65:
            words = formula.replace('(', '( ').replace(')', ' )').split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line) + len(word) > 60:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line += " " + word if current_line else word
            if current_line:
                lines.append(current_line)
            for line in lines:
                print(f"    {style(line.strip(), Theme.PRIMARY)}")
        else:
            print(f"    {style(formula, Theme.PRIMARY)}")
        
        # Period results
        results = strategy.get('period_results', [])
        if results:
            print(f"\n  {style('PERIOD PERFORMANCE', Theme.ACCENT, Theme.BOLD)}")
            
            avg_ret = np.mean([r['total_return'] for r in results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
            avg_dd = np.mean([r['max_drawdown'] for r in results])
            win_periods = sum(1 for r in results if r['total_return'] > 0) / len(results)
            
            print(format_metric("Avg Return", avg_ret, as_percent=True))
            print(format_metric("Avg Sharpe", avg_sharpe))
            print(format_metric("Avg Max DD", avg_dd, as_percent=True))
            print(format_metric("Win Rate (periods)", win_periods, as_percent=True))
            print(format_metric("Periods Tested", len(results), precision=0))
        
        # Benchmark comparison
        benchmarks = strategy.get('benchmarks', {})
        if benchmarks and results:
            print(f"\n  {style('VS BENCHMARKS', Theme.ACCENT, Theme.BOLD)}")
            
            strat_avg_ret = np.mean([r['total_return'] for r in results])
            strat_avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
            
            print(f"\n  {'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'Alpha':>10}")
            print(f"  {style('-' * 45, Theme.MUTED)}")
            print(f"  {style('This Strategy', Theme.SUCCESS):<15} "
                  f"{format_number(strat_avg_ret, as_percent=True):>10} "
                  f"{format_number(strat_avg_sharpe):>8} "
                  f"{style('—', Theme.MUTED):>10}")
            
            for bench_name, bench_results in benchmarks.items():
                bench_avg_ret = np.mean([b['total_return'] for b in bench_results])
                bench_avg_sharpe = np.mean([b['sharpe_ratio'] for b in bench_results])
                alpha = strat_avg_ret - bench_avg_ret
                
                alpha_str = f"{alpha*100:+.1f}%"
                alpha_color = Theme.SUCCESS if alpha > 0 else Theme.ERROR
                
                # Calculate outperformance rate
                outperform = 0
                for i, r in enumerate(results):
                    if i < len(bench_results):
                        if r['total_return'] > bench_results[i]['total_return']:
                            outperform += 1
                outperform_rate = outperform / min(len(results), len(bench_results)) if results else 0
                
                print(f"  {bench_name:<15} "
                      f"{format_number(bench_avg_ret, as_percent=True):>10} "
                      f"{format_number(bench_avg_sharpe):>8} "
                      f"{style(alpha_str, alpha_color):>10}")
            
            # Summary
            print(f"\n  {style('Outperformance Summary:', Theme.SECONDARY)}")
            for bench_name, bench_results in benchmarks.items():
                outperform = 0
                for i, r in enumerate(results):
                    if i < len(bench_results):
                        if r['total_return'] > bench_results[i]['total_return']:
                            outperform += 1
                outperform_rate = outperform / min(len(results), len(bench_results)) if results else 0
                print(f"    vs {bench_name}: Beat in {outperform_rate:.0%} of periods")
        
        # Yearly returns
        yearly = strategy.get('yearly_returns', {})
        if yearly:
            print(f"\n  {style('YEARLY RETURNS', Theme.ACCENT, Theme.BOLD)}")
            for year in sorted(yearly.keys()):
                ret = yearly[year]
                print(f"    {style(year, Theme.SECONDARY)}: {format_number(ret, as_percent=True)}")
        
        print(f"\n  {style('─' * 50, Theme.MUTED)}")
        print(f"  {style('Press Enter to continue...', Theme.MUTED)}", end='')
        input()
    
    def inspect_strategy(self, strategy_id: str = None):
        """Show detailed view of a GA strategy."""
        if strategy_id:
            strategy = self.get_ga_strategy_by_id(strategy_id)
        elif self.current_strategy and self.current_strategy_type == 'ga':
            strategy = self.current_strategy
        else:
            print(f"\n  {style('No GA strategy selected.', Theme.WARNING)}")
            return
        
        if not strategy:
            print(f"\n  {style('Strategy not found.', Theme.ERROR)}")
            return
        
        self.current_strategy = strategy
        self.current_strategy_type = 'ga'
        
        print_section(f"STRATEGY: {strategy.get('name', 'Unknown')[:40]}", "🔬")
        
        print(f"\n  {style('IDENTIFICATION', Theme.ACCENT, Theme.BOLD)}")
        print(f"  {style('ID:', Theme.SECONDARY)} {style(strategy.get('strategy_id', 'N/A'), Theme.PRIMARY)}")
        print(f"  {style('Name:', Theme.SECONDARY)} {strategy.get('name', 'Unknown')}")
        print(f"  {style('Generation:', Theme.SECONDARY)} {strategy.get('generation', 'N/A')}")
        
        print(f"\n  {style('PERFORMANCE METRICS', Theme.ACCENT, Theme.BOLD)}")
        print(format_metric("Avg Sharpe Ratio", strategy.get('avg_sharpe', 0)))
        print(format_metric("Avg Total Return", strategy.get('avg_return', 0), as_percent=True))
        print(format_metric("Avg Max Drawdown", strategy.get('avg_max_drawdown', 0), as_percent=True))
        print(format_metric("Avg Win Rate", strategy.get('avg_win_rate', 0), as_percent=True))
        print(format_metric("Test Count", strategy.get('test_count', 0), precision=0))
        
        yearly = strategy.get('yearly_returns', {})
        if yearly:
            print(f"\n  {style('YEARLY RETURNS', Theme.ACCENT, Theme.BOLD)}")
            for year in sorted(yearly.keys()):
                ret = yearly[year]
                print(f"    {style(year, Theme.SECONDARY)}: {format_number(ret, as_percent=True)}")
        
        genes = strategy.get('genes', {})
        if genes:
            print(f"\n  {style('GENE COMPOSITION', Theme.ACCENT, Theme.BOLD)}")
            for gene_name, params in genes.items():
                color = Theme.get_gene_color(gene_name)
                weight = params.get('weight', 1.0) if isinstance(params, dict) else 1.0
                print(f"    {style('●', color)} {style(gene_name, color, Theme.BOLD)} (weight: {weight:.2f})")
        
        print(f"\n  {style('─' * 50, Theme.MUTED)}")
        print(f"  {style('Press Enter to continue...', Theme.MUTED)}", end='')
        input()
    
    def export_strategy(self, strategy_id: str = None, strategy_type: str = None):
        """Export strategy to JSON file."""
        if strategy_id:
            if strategy_type == 'gp':
                strategy = self.get_gp_strategy_by_id(strategy_id)
            else:
                strategy = self.get_ga_strategy_by_id(strategy_id)
        elif self.current_strategy:
            strategy = self.current_strategy
            strategy_type = self.current_strategy_type
        else:
            print(f"\n  {style('No strategy selected.', Theme.WARNING)}")
            return
        
        if not strategy:
            print(f"\n  {style('Strategy not found.', Theme.ERROR)}")
            return
        
        sid = strategy.get('strategy_id', 'unknown')[:8]
        filepath = DATA_DIR / "exports" / f"strategy_{strategy_type}_{sid}.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        export_data = {
            'strategy_id': strategy.get('strategy_id'),
            'type': strategy_type or strategy.get('type', 'unknown'),
            'exported_at': datetime.now().isoformat(),
        }
        
        if strategy_type == 'gp':
            export_data.update({
                'formula': strategy.get('formula'),
                'fitness': strategy.get('fitness'),
                'complexity': strategy.get('complexity'),
                'top_pct': strategy.get('top_pct'),
                'period_results': strategy.get('period_results', []),
                'benchmarks': strategy.get('benchmarks', {}),
            })
        else:
            export_data.update({
                'name': strategy.get('name'),
                'generation': strategy.get('generation'),
                'performance': {
                    'avg_sharpe': strategy.get('avg_sharpe'),
                    'avg_return': strategy.get('avg_return'),
                    'avg_max_drawdown': strategy.get('avg_max_drawdown'),
                },
                'genes': strategy.get('genes', {}),
            })
        
        export_data['yearly_returns'] = strategy.get('yearly_returns', {})
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n  {style('✓', Theme.SUCCESS)} Exported to: {style(str(filepath), Theme.PRIMARY)}")
    
    def show_help(self):
        """Show help information."""
        print_section("HELP", "❓")
        print(f"""
  {style('STRATEGY TYPES', Theme.ACCENT, Theme.BOLD)}
  
  • {style('GA (Gene-based)', Theme.PRIMARY)}: Traditional genetic algorithm strategies
    with discrete genes (momentum, value, quality, etc.)
  
  • {style('GP (Expression Trees)', Theme.SUCCESS)}: Genetic programming strategies
    with evolved mathematical formulas
  
  {style('BENCHMARKS', Theme.ACCENT, Theme.BOLD)}
  
  • {style('buy_hold', Theme.SECONDARY)}: Equal-weight buy & hold of universe
  • {style('momentum_12m', Theme.SECONDARY)}: Top 20% by 12-month momentum
  
  {style('METRICS', Theme.ACCENT, Theme.BOLD)}
  
  • {style('Fitness', Theme.PRIMARY)}: Overall strategy quality (0-1)
  • {style('Sharpe', Theme.PRIMARY)}: Risk-adjusted return (higher = better)
  • {style('Alpha', Theme.SUCCESS)}: Excess return vs benchmark
  • {style('MaxDD', Theme.ERROR)}: Maximum drawdown (lower = better)
  
  {style('Press Enter to continue...', Theme.MUTED)}""", end='')
        input()
    
    def run(self):
        """Main interactive loop."""
        clear_screen()
        print_header()
        
        while True:
            if self.current_strategy:
                name = self.current_strategy.get('name', 
                       self.current_strategy.get('strategy_id', 'Unknown'))[:25]
                stype = self.current_strategy_type.upper()
                print(f"\n  {style('Current:', Theme.MUTED)} {style(f'[{stype}]', Theme.ACCENT)} {style(name, Theme.PRIMARY)}")
            
            print(f"""
  {style('╭─────────────────────────────────────────╮', Theme.SECONDARY)}
  {style('│', Theme.SECONDARY)}  {style('GP STRATEGIES', Theme.SUCCESS)}                        {style('│', Theme.SECONDARY)}
  {style('│', Theme.SECONDARY)}  {style('1', Theme.PRIMARY)}  View Top GP Strategies           {style('│', Theme.SECONDARY)}
  {style('│', Theme.SECONDARY)}  {style('2', Theme.PRIMARY)}  Inspect GP Strategy              {style('│', Theme.SECONDARY)}
  {style('├─────────────────────────────────────────┤', Theme.SECONDARY)}
  {style('│', Theme.SECONDARY)}  {style('GA STRATEGIES', Theme.WARNING)}                        {style('│', Theme.SECONDARY)}
  {style('│', Theme.SECONDARY)}  {style('3', Theme.PRIMARY)}  View Top GA Strategies           {style('│', Theme.SECONDARY)}
  {style('│', Theme.SECONDARY)}  {style('4', Theme.PRIMARY)}  Inspect GA Strategy              {style('│', Theme.SECONDARY)}
  {style('├─────────────────────────────────────────┤', Theme.SECONDARY)}
  {style('│', Theme.SECONDARY)}  {style('5', Theme.PRIMARY)}  Export Current Strategy          {style('│', Theme.SECONDARY)}
  {style('│', Theme.SECONDARY)}  {style('h', Theme.MUTED)}  Help                             {style('│', Theme.SECONDARY)}
  {style('│', Theme.SECONDARY)}  {style('q', Theme.MUTED)}  Quit                             {style('│', Theme.SECONDARY)}
  {style('╰─────────────────────────────────────────╯', Theme.SECONDARY)}
""")
            
            choice = input(f"  {style('►', Theme.PRIMARY)} ").strip().lower()
            
            if choice == '1':
                strategies = self.view_top_gp_strategies()
                if strategies:
                    sub = input(f"\n  {style('Enter # to inspect (or Enter to skip):', Theme.MUTED)} ").strip()
                    if sub.isdigit() and 1 <= int(sub) <= len(strategies):
                        sid = strategies[int(sub) - 1]['strategy_id']
                        self.inspect_gp_strategy(sid)
                    elif sub:
                        self.inspect_gp_strategy(sub)
            
            elif choice == '2':
                sid = input(f"\n  {style('Enter GP strategy ID:', Theme.PRIMARY)} ").strip()
                if sid:
                    self.inspect_gp_strategy(sid)
            
            elif choice == '3':
                strategies = self.view_top_strategies()
                if strategies:
                    sub = input(f"\n  {style('Enter # to inspect (or Enter to skip):', Theme.MUTED)} ").strip()
                    if sub.isdigit() and 1 <= int(sub) <= len(strategies):
                        sid = strategies[int(sub) - 1]['strategy_id']
                        self.inspect_strategy(sid)
                    elif sub:
                        self.inspect_strategy(sub)
            
            elif choice == '4':
                sid = input(f"\n  {style('Enter GA strategy ID:', Theme.PRIMARY)} ").strip()
                if sid:
                    self.inspect_strategy(sid)
            
            elif choice == '5':
                if self.current_strategy:
                    self.export_strategy()
                else:
                    print(f"\n  {style('No strategy selected. View and inspect a strategy first.', Theme.WARNING)}")
            
            elif choice == 'h':
                self.show_help()
            
            elif choice in ('q', 'quit', 'exit', '0'):
                print(f"\n  {style('Goodbye! 🧬', Theme.PRIMARY)}\n")
                break
            
            elif choice == '':
                continue
            
            else:
                print(f"\n  {style('Unknown option. Press h for help.', Theme.WARNING)}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AlphaGene Strategy Picker - Browse and analyze evolved strategies"
    )
    parser.add_argument(
        '--db', 
        type=str, 
        default='data/cache/strategies.db',
        help='Path to GA strategy database'
    )
    parser.add_argument(
        '--gp-db',
        type=str,
        default='data/gp_strategies.db',
        help='Path to GP strategy database'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=None,
        help='Just show top N strategies and exit'
    )
    parser.add_argument(
        '--gp',
        action='store_true',
        help='Show GP strategies (default is GA)'
    )
    
    args = parser.parse_args()
    
    picker = StrategyPicker(db_path=args.db, gp_db_path=args.gp_db)
    
    if args.top:
        if args.gp:
            picker.view_top_gp_strategies(n=args.top)
        else:
            picker.view_top_strategies(n=args.top)
    else:
        picker.run()


if __name__ == "__main__":
    main()