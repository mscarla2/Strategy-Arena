"""
Evolution Progress Reporter — file-based IPC for real-time dashboard monitoring.

Writes generation stats to a JSON file that the Streamlit dashboard polls.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


class EvolutionProgressReporter:
    """Reports evolution progress to dashboard via file-based IPC."""

    def __init__(self, progress_file: str = "data/cache/evolution_progress.json"):
        self.progress_file = Path(progress_file)
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

    def report_generation(self, stats: Dict) -> None:
        """Write generation stats to progress file.

        Args:
            stats: Dictionary with keys like generation, avg_fitness,
                   max_fitness, min_fitness, stagnant, etc.
        """
        progress = {
            "timestamp": datetime.now().isoformat(),
            "generation": stats.get("generation", 0),
            "total_generations": stats.get("total_generations", 30),
            "best_fitness": stats.get("max_fitness", 0),
            "avg_fitness": stats.get("avg_fitness", 0),
            "min_fitness": stats.get("min_fitness", 0),
            "best_ever": stats.get("best_ever", 0),
            "avg_sharpe": stats.get("avg_sharpe", 0),
            "avg_return": stats.get("avg_return", 0),
            "avg_complexity": stats.get("avg_complexity", 0),
            "stagnant": stats.get("stagnant", 0),
            "status": "running",
        }

        # Append to history list so the dashboard can chart all generations
        history = self._read_history()
        history.append(progress)

        payload = {
            "current": progress,
            "history": history,
        }

        try:
            self.progress_file.write_text(json.dumps(payload, indent=2))
        except OSError:
            pass  # Non-critical — dashboard just won't update this tick

    def report_complete(self, final_stats: Dict) -> None:
        """Mark evolution as complete in the progress file."""
        progress = {
            "timestamp": datetime.now().isoformat(),
            "generation": final_stats.get("generation", 0),
            "total_generations": final_stats.get("total_generations", 30),
            "best_fitness": final_stats.get("best_fitness", 0),
            "avg_fitness": final_stats.get("avg_fitness", 0),
            "status": "complete",
        }

        history = self._read_history()
        payload = {
            "current": progress,
            "history": history,
        }

        try:
            self.progress_file.write_text(json.dumps(payload, indent=2))
        except OSError:
            pass

    def read_progress(self) -> Optional[Dict]:
        """Read current progress (for dashboard polling).

        Returns:
            Dict with 'current' and 'history' keys, or None if no file.
        """
        if not self.progress_file.exists():
            return None
        try:
            data = json.loads(self.progress_file.read_text())
            return data
        except (json.JSONDecodeError, OSError):
            return None

    def clear(self) -> None:
        """Remove the progress file (e.g. before a new run)."""
        try:
            if self.progress_file.exists():
                self.progress_file.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_history(self):
        """Read existing history list from the progress file."""
        if not self.progress_file.exists():
            return []
        try:
            data = json.loads(self.progress_file.read_text())
            return data.get("history", [])
        except (json.JSONDecodeError, OSError):
            return []
