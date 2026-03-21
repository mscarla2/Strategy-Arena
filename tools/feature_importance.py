#!/usr/bin/env python3
"""
Feature Importance Analysis Tool
Analyzes which features contribute most to alpha generation in evolved strategies.
"""

import sys
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR


class FeatureImportanceAnalyzer:
    """Analyze feature importance from evolved strategies."""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or (DATA_DIR.parent / "gp_strategies.db")
        
    def extract_features_from_formula(self, formula: str) -> List[str]:
        """Extract feature names from a strategy formula."""
        # Pattern to match feature names (alphanumeric + underscores)
        # Exclude operators and numbers
        pattern = r'\b([a-zA-Z][a-zA-Z0-9_]*)\b'
        
        # Operators to exclude
        operators = {
            'add', 'sub', 'mul', 'div', 'max', 'min', 'avg',
            'neg', 'abs', 'sign', 'sqrt', 'square', 'inv', 'log',
            'sigmoid', 'tanh', 'rank', 'zscore', 'and', 'or', 'not', 'if'
        }
        
        matches = re.findall(pattern, formula)
        features = [m for m in matches if m not in operators]
        
        return features
    
    def analyze_top_strategies(self, top_n: int = 50, min_fitness: float = 0.0) -> pd.DataFrame:
        """Analyze feature usage in top performing strategies."""
        
        if not self.db_path.exists():
            print(f"Database not found: {self.db_path}")
            return pd.DataFrame()
        
        conn = sqlite3.connect(self.db_path)
        
        # Get top strategies
        query = """
        SELECT 
            s.strategy_id,
            s.formula,
            s.fitness,
            s.generation,
            s.complexity,
            r.run_id,
            r.config
        FROM gp_strategies s
        JOIN evolution_runs r ON s.run_id = r.run_id
        WHERE s.fitness >= ?
        ORDER BY s.fitness DESC
        LIMIT ?
        """
        
        strategies = pd.read_sql_query(query, conn, params=(min_fitness, top_n))
        conn.close()
        
        if strategies.empty:
            print("No strategies found matching criteria")
            return pd.DataFrame()
        
        # Extract features from each strategy
        feature_usage = Counter()
        feature_fitness = {}  # Track fitness for each feature
        
        for _, row in strategies.iterrows():
            formula = row['formula']
            fitness = row['fitness']
            
            features = self.extract_features_from_formula(formula)
            
            for feature in features:
                feature_usage[feature] += 1
                
                if feature not in feature_fitness:
                    feature_fitness[feature] = []
                feature_fitness[feature].append(fitness)
        
        # Create analysis dataframe
        analysis = []
        for feature, count in feature_usage.items():
            fitnesses = feature_fitness[feature]
            analysis.append({
                'feature': feature,
                'usage_count': count,
                'usage_pct': count / len(strategies) * 100,
                'avg_fitness': np.mean(fitnesses),
                'max_fitness': np.max(fitnesses),
                'min_fitness': np.min(fitnesses),
                'weighted_score': count * np.mean(fitnesses),  # add this
            })
        
        df = pd.DataFrame(analysis)
        df = df.sort_values('weighted_score', ascending=False)
        
        return df
    
    def analyze_by_universe(self, universe_keyword: str = None) -> Dict[str, pd.DataFrame]:
        """Analyze feature importance by universe (e.g., 'oil')."""
        
        if not self.db_path.exists():
            print(f"Database not found: {self.db_path}")
            return {}
        
        conn = sqlite3.connect(self.db_path)
        
        # Get strategies grouped by universe
        query = """
        SELECT 
            s.strategy_id,
            s.formula,
            s.fitness,
            r.config
        FROM gp_strategies s
        JOIN evolution_runs r ON s.run_id = r.run_id
        WHERE s.fitness > 0
        """
        
        strategies = pd.read_sql_query(query, conn)
        conn.close()
        
        if strategies.empty:
            return {}
        
        # Group by universe (extract from config)
        results = {}
        
        for _, row in strategies.iterrows():
            # Simple universe detection from config
            config = row['config']
            if 'oil' in config.lower():
                universe = 'oil'
            else:
                universe = 'general'
            
            if universe not in results:
                results[universe] = []
            
            results[universe].append(row)
        
        # Analyze each universe
        analysis = {}
        for universe, strats in results.items():
            feature_usage = Counter()
            
            for strat in strats:
                features = self.extract_features_from_formula(strat['formula'])
                feature_usage.update(features)
            
            df = pd.DataFrame([
                {'feature': f, 'count': c, 'pct': c/len(strats)*100}
                for f, c in feature_usage.most_common(30)
            ])
            
            analysis[universe] = df
        
        return analysis
    
    def print_report(self, top_n: int = 50, min_fitness: float = 0.0):
        """Print feature importance report."""
        
        print("=" * 80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        print(f"\nAnalyzing top {top_n} strategies (min fitness: {min_fitness:.2f})")
        print()
        
        # Overall analysis
        df = self.analyze_top_strategies(top_n, min_fitness)
        
        if df.empty:
            print("No data available for analysis")
            return
        
        print(f"Top 20 Most Used Features:")
        print("-" * 80)
        print(f"{'Feature':<30} {'Usage':<10} {'Avg Fitness':<15} {'Max Fitness':<15} {'Weighted Score':<15}")
        print("-" * 95)

        for _, row in df.head(20).iterrows():
            print(f"{row['feature']:<30} {row['usage_count']:<4} ({row['usage_pct']:>5.1f}%)  "
                f"{row['avg_fitness']:>6.3f}         {row['max_fitness']:>6.3f}         {row['weighted_score']:>6.3f}")
        
        print()
        print(f"\nTotal unique features found: {len(df)}")
        print(f"Features used in >50% of strategies: {len(df[df['usage_pct'] > 50])}")
        print(f"Features used in >25% of strategies: {len(df[df['usage_pct'] > 25])}")
        
        # Universe-specific analysis
        print("\n" + "=" * 80)
        print("UNIVERSE-SPECIFIC ANALYSIS")
        print("=" * 80)
        
        by_universe = self.analyze_by_universe()
        
        for universe, universe_df in by_universe.items():
            print(f"\n{universe.upper()} Universe - Top 10 Features:")
            print("-" * 60)
            for _, row in universe_df.head(10).iterrows():
                print(f"  {row['feature']:<30} {row['count']:<4} ({row['pct']:>5.1f}%)")
        
        print("\n" + "=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze feature importance in evolved strategies")
    parser.add_argument('--top-n', type=int, default=50, help='Number of top strategies to analyze')
    parser.add_argument('--min-fitness', type=float, default=0.0, help='Minimum fitness threshold')
    parser.add_argument('--db-path', type=str, default=None, help='Path to database file')
    parser.add_argument('--export-csv', type=str, default=None, help='Export results to CSV')
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path) if args.db_path else None
    analyzer = FeatureImportanceAnalyzer(db_path)
    
    # Print report
    analyzer.print_report(args.top_n, args.min_fitness)
    
    # Export if requested
    if args.export_csv:
        df = analyzer.analyze_top_strategies(args.top_n, args.min_fitness)
        df.to_csv(args.export_csv, index=False)
        print(f"\nResults exported to: {args.export_csv}")


if __name__ == "__main__":
    main()
