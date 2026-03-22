#!/usr/bin/env python3
"""
Database Cleanup Utility
Removes strategies without period results (old duplicates from intermediate generations).
Keeps only strategies with complete performance data.
"""

import sqlite3
from pathlib import Path

def cleanup_database(db_path: str = "data/gp_strategies.db", dry_run: bool = True):
    """
    Remove strategies that don't have period results.
    
    Args:
        db_path: Path to the database
        dry_run: If True, only show what would be deleted without actually deleting
    """
    
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # Find strategies without period results
        cursor = conn.execute("""
            SELECT s.strategy_id, s.formula, s.fitness, s.run_id
            FROM gp_strategies s
            LEFT JOIN gp_period_results p ON s.strategy_id = p.strategy_id
            WHERE p.strategy_id IS NULL
            ORDER BY s.fitness DESC
        """)
        
        strategies_to_delete = cursor.fetchall()
        
        if not strategies_to_delete:
            print("✅ No strategies to clean up. Database is already clean!")
            return
        
        print(f"\n{'='*80}")
        print(f"Found {len(strategies_to_delete)} strategies WITHOUT period results:")
        print(f"{'='*80}\n")
        
        for i, s in enumerate(strategies_to_delete[:10], 1):
            formula = s['formula'][:60] + "..." if len(s['formula']) > 60 else s['formula']
            print(f"{i}. ID: {s['strategy_id'][:16]}... | Fitness: {s['fitness']:+.3f}")
            print(f"   Formula: {formula}")
            print(f"   Run: {s['run_id']}")
            print()
        
        if len(strategies_to_delete) > 10:
            print(f"... and {len(strategies_to_delete) - 10} more\n")
        
        # Find strategies WITH period results (these will be kept)
        cursor = conn.execute("""
            SELECT COUNT(DISTINCT s.strategy_id) as count
            FROM gp_strategies s
            INNER JOIN gp_period_results p ON s.strategy_id = p.strategy_id
        """)
        
        keep_count = cursor.fetchone()['count']
        
        print(f"{'='*80}")
        print(f"Summary:")
        print(f"  Strategies to DELETE: {len(strategies_to_delete)} (no period results)")
        print(f"  Strategies to KEEP:   {keep_count} (have period results)")
        print(f"{'='*80}\n")
        
        if dry_run:
            print("🔍 DRY RUN MODE - No changes made")
            print("   Run with --execute to actually delete these strategies")
        else:
            print("⚠️  EXECUTING DELETION...")
            
            # Delete strategies without period results
            conn.execute("""
                DELETE FROM gp_strategies
                WHERE strategy_id IN (
                    SELECT s.strategy_id
                    FROM gp_strategies s
                    LEFT JOIN gp_period_results p ON s.strategy_id = p.strategy_id
                    WHERE p.strategy_id IS NULL
                )
            """)
            
            conn.commit()
            print(f"✅ Deleted {len(strategies_to_delete)} strategies")
            print(f"✅ Kept {keep_count} strategies with complete data")


def show_database_stats(db_path: str = "data/gp_strategies.db"):
    """Show current database statistics."""
    
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        
        # Total strategies
        total = conn.execute("SELECT COUNT(*) as count FROM gp_strategies").fetchone()['count']
        
        # Strategies with period results
        with_results = conn.execute("""
            SELECT COUNT(DISTINCT s.strategy_id) as count
            FROM gp_strategies s
            INNER JOIN gp_period_results p ON s.strategy_id = p.strategy_id
        """).fetchone()['count']
        
        # Strategies without period results
        without_results = total - with_results
        
        # Total runs
        runs = conn.execute("SELECT COUNT(*) as count FROM evolution_runs").fetchone()['count']
        
        # Top 5 strategies
        top_strategies = conn.execute("""
            SELECT s.strategy_id, s.formula, s.fitness, 
                   COUNT(p.id) as period_count
            FROM gp_strategies s
            LEFT JOIN gp_period_results p ON s.strategy_id = p.strategy_id
            GROUP BY s.strategy_id
            ORDER BY s.fitness DESC
            LIMIT 5
        """).fetchall()
        
        print(f"\n{'='*80}")
        print(f"DATABASE STATISTICS")
        print(f"{'='*80}\n")
        print(f"Total Evolution Runs:     {runs}")
        print(f"Total Strategies:         {total}")
        print(f"  ✅ With period results:  {with_results}")
        print(f"  ❌ Without results:      {without_results}")
        print(f"\n{'='*80}")
        print(f"TOP 5 STRATEGIES:")
        print(f"{'='*80}\n")
        
        for i, s in enumerate(top_strategies, 1):
            formula = s['formula'][:50] + "..." if len(s['formula']) > 50 else s['formula']
            status = "✅" if s['period_count'] > 0 else "❌"
            print(f"{i}. {status} Fitness: {s['fitness']:+.3f} | Periods: {s['period_count']}")
            print(f"   {formula}")
            print()

def wipe_database(db_path: str = "data/gp_strategies.db", dry_run: bool = True):
    """Nuclear option — wipe all strategies and runs for a clean start."""
    
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    with sqlite3.connect(db_path) as conn:
        counts = {
            'strategies': conn.execute("SELECT COUNT(*) FROM gp_strategies").fetchone()[0],
            'runs': conn.execute("SELECT COUNT(*) FROM evolution_runs").fetchone()[0],
            'period_results': conn.execute("SELECT COUNT(*) FROM gp_period_results").fetchone()[0],
            'generation_stats': conn.execute("SELECT COUNT(*) FROM generation_stats").fetchone()[0],
            'benchmarks': conn.execute("SELECT COUNT(*) FROM benchmarks").fetchone()[0],
        }
        
        print(f"\n{'='*80}")
        print(f"FULL WIPE — Records to delete:")
        for table, count in counts.items():
            print(f"  {table}: {count}")
        print(f"{'='*80}\n")
        
        if dry_run:
            print("🔍 DRY RUN — Run with --wipe --execute to actually delete")
        else:
            for table in ['gp_period_results', 'generation_stats', 'benchmarks', 
                         'gp_strategies', 'evolution_runs']:
                conn.execute(f"DELETE FROM {table}")
            conn.execute("DELETE FROM sqlite_sequence")
            conn.commit()
            print("✅ Database wiped clean")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up Strategy Arena database")
    parser.add_argument('--execute', action='store_true',
                       help='Actually delete strategies (default is dry-run)')
    parser.add_argument('--stats', action='store_true',
                       help='Show database statistics')
    parser.add_argument('--db', type=str, default='data/gp_strategies.db',
                       help='Path to database (default: data/gp_strategies.db)')
    parser.add_argument('--wipe', action='store_true',
                   help='Wipe entire database (use with --execute)')
    args = parser.parse_args()
    if args.wipe:
        wipe_database(args.db, dry_run=not args.execute)
    else:
        cleanup_database(args.db, dry_run=not args.execute)
    
    if args.stats:
        show_database_stats(args.db)
    else:
        cleanup_database(args.db, dry_run=not args.execute)
        print()
        show_database_stats(args.db)
