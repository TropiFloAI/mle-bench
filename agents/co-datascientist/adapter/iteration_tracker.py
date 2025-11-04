"""
Iteration Tracker for MLE-Bench Co-DataScientist Agent

Tracks validation and test KPIs at each iteration during the optimization loop.
Creates live-updating plots showing KPI progression vs medal thresholds.
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np

# Note: Grading happens on HOST after container finishes, not inside container
# This tracker only records validation KPIs and manages the plot
# Test KPIs will be added by host-side grading script

logger = logging.getLogger(__name__)


class IterationTracker:
    """Tracks and visualizes KPI progression during optimization iterations."""
    
    def __init__(self, output_dir: str, competition_id: str):
        """
        Args:
            output_dir: Directory to save plots and tracking data (usually submission dir)
            competition_id: The competition being run
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.competition_id = competition_id
        
        # Tracking file for persistence
        self.tracking_file = self.output_dir / "iteration_tracking.json"
        self.plot_file = self.output_dir / "kpi_progression.png"
        
        # Initialize or load tracking data
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "competition_id": competition_id,
                "started_at": datetime.now().isoformat(),
                "iterations": [],
                "medal_thresholds": None,
                "is_lower_better": None
            }
        
        # Get medal thresholds (do this once)
        if self.data["medal_thresholds"] is None:
            self._initialize_medal_thresholds()
    
    def _initialize_medal_thresholds(self):
        """Medal thresholds will be loaded by host-side grading script."""
        # Just set placeholders - host will update these
        try:
            self.data["medal_thresholds"] = {
                "gold": None,
                "silver": None,
                "bronze": None,
                "median": None
            }
            self.data["is_lower_better"] = False
            return
            
            # Old code kept for reference but not used:
            # from mlebench.data import get_leaderboard
            # leaderboard = get_leaderboard(self.competition)
            # rank_info = self.competition.grader.rank_score(None, leaderboard)
            # is_lower_better = self.competition.grader.is_lower_better(leaderboard)
            
            self.data["medal_thresholds"] = {
                "gold": rank_info["gold_threshold"],
                "silver": rank_info["silver_threshold"],
                "bronze": rank_info["bronze_threshold"],
                "median": rank_info["median_threshold"]
            }
            self.data["is_lower_better"] = is_lower_better
            
            logger.info(f"📊 Medal thresholds loaded for {self.competition_id}:")
            logger.info(f"   Gold:   {rank_info['gold_threshold']:.6f}")
            logger.info(f"   Silver: {rank_info['silver_threshold']:.6f}")
            logger.info(f"   Bronze: {rank_info['bronze_threshold']:.6f}")
            logger.info(f"   Median: {rank_info['median_threshold']:.6f}")
            logger.info(f"   {'Lower' if is_lower_better else 'Higher'} is better")
            
            self._save()
        except Exception as e:
            logger.warning(f"Could not load medal thresholds: {e}")
    
    def record_iteration(self, 
                        iteration: int,
                        val_kpi: Optional[float],
                        submission_path: Optional[Path] = None,
                        code_version_id: Optional[str] = None) -> Optional[float]:
        """
        Record an iteration and optionally grade the submission to get test KPI.
        
        Args:
            iteration: Iteration number (starting from 1)
            val_kpi: Validation KPI for this iteration
            submission_path: Path to submission.csv (if available, will grade it)
            code_version_id: Identifier for this code version
            
        Returns:
            test_kpi: Test KPI if grading was successful, None otherwise
        """
        # Reload data from disk to get any updates from host grading script
        self._reload_from_disk()
        
        test_kpi = None
        test_score_details = None
        
        # Grading happens on HOST side after container finishes
        # Just log that submission was saved
        if submission_path and submission_path.exists():
            logger.info(f"   💾 Submission saved for iteration {iteration} - will be graded on host")
        else:
            logger.warning(f"   ⚠️  No submission found for iteration {iteration}")
        
        # Check if this iteration already exists (may have been graded by host)
        existing_idx = next((i for i, it in enumerate(self.data["iterations"]) 
                           if it["iteration"] == iteration), None)
        
        if existing_idx is not None:
            # Preserve existing test_kpi and test_score_details if they exist
            existing_data = self.data["iterations"][existing_idx]
            test_kpi = existing_data.get("test_kpi")
            test_score_details = existing_data.get("test_score_details")
        
        # Record iteration data (preserving test data if it exists)
        iteration_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "val_kpi": val_kpi,
            "test_kpi": test_kpi,  # Preserved from existing data
            "test_score_details": test_score_details,  # Preserved from existing data
            "code_version_id": code_version_id
        }
        
        # Update or append
        if existing_idx is not None:
            self.data["iterations"][existing_idx] = iteration_data
        else:
            self.data["iterations"].append(iteration_data)
        
        self._save()
        # Plot generation removed - grading script will generate plot with complete data
        
        return test_kpi
    
    def _reload_from_disk(self):
        """Reload tracking data from disk to get any updates from host grading."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    disk_data = json.load(f)
                # Preserve medal thresholds and test KPIs from disk
                self.data["medal_thresholds"] = disk_data.get("medal_thresholds", self.data["medal_thresholds"])
                self.data["is_lower_better"] = disk_data.get("is_lower_better", self.data["is_lower_better"])
                # Update iterations with any graded data from disk
                for disk_iter in disk_data.get("iterations", []):
                    existing_idx = next((i for i, it in enumerate(self.data["iterations"]) 
                                       if it["iteration"] == disk_iter["iteration"]), None)
                    if existing_idx is not None:
                        # Preserve test data from disk if it exists
                        if disk_iter.get("test_kpi") is not None:
                            self.data["iterations"][existing_idx]["test_kpi"] = disk_iter["test_kpi"]
                            self.data["iterations"][existing_idx]["test_score_details"] = disk_iter.get("test_score_details")
            except Exception as e:
                logger.warning(f"Could not reload data from disk: {e}")
    
    def _save(self):
        """Save tracking data to JSON file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def _update_plot(self):
        """Create or update the KPI progression plot."""
        if not self.data["iterations"]:
            return
        
        try:
            # Extract data
            iterations = [it["iteration"] for it in self.data["iterations"]]
            val_kpis = [it["val_kpi"] for it in self.data["iterations"]]
            test_kpis = [it["test_kpi"] for it in self.data["iterations"]]
            timestamps = [it["timestamp"] for it in self.data["iterations"]]
            
            # Filter out None values for plotting
            valid_val = [(i, v) for i, v in zip(iterations, val_kpis) if v is not None]
            valid_test = [(i, v) for i, v in zip(iterations, test_kpis) if v is not None]
            
            if not valid_val and not valid_test:
                return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot validation KPIs
            if valid_val:
                val_iters, val_vals = zip(*valid_val)
                ax.plot(val_iters, val_vals, 'o-', label='Validation KPI', 
                       color='#2E86AB', linewidth=2, markersize=8, alpha=0.8)
            
            # Plot test KPIs
            if valid_test:
                test_iters, test_vals = zip(*valid_test)
                ax.plot(test_iters, test_vals, 's-', label='Test KPI', 
                       color='#A23B72', linewidth=2, markersize=8, alpha=0.8)
            
            # Add medal threshold lines
            if self.data["medal_thresholds"]:
                thresholds = self.data["medal_thresholds"]
                colors = {
                    "gold": "#FFD700",
                    "silver": "#C0C0C0", 
                    "bronze": "#CD7F32",
                    "median": "#808080"
                }
                labels = {
                    "gold": "Gold Medal",
                    "silver": "Silver Medal",
                    "bronze": "Bronze Medal",
                    "median": "Median"
                }
                
                for level, threshold in thresholds.items():
                    if threshold is not None:
                        ax.axhline(y=threshold, color=colors[level], 
                                  linestyle='--', linewidth=2, alpha=0.7,
                                  label=f'{labels[level]}: {threshold:.4f}')
            
            # Styling
            ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
            ax.set_ylabel('KPI Value', fontsize=12, fontweight='bold')
            ax.set_title(f'KPI Progression: {self.competition_id}', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Add timestamp labels to x-axis
            if iterations and timestamps:
                # Format timestamps for display (show time only)
                from datetime import datetime as dt
                time_labels = []
                for ts in timestamps:
                    try:
                        parsed_ts = dt.fromisoformat(ts)
                        time_labels.append(parsed_ts.strftime('%H:%M:%S'))
                    except:
                        time_labels.append('')
                
                # Set custom x-tick labels showing iteration and time
                ax.set_xticks(iterations)
                tick_labels = [f"#{iter}\n{time}" for iter, time in zip(iterations, time_labels)]
                ax.set_xticklabels(tick_labels, fontsize=9)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
            # Legend
            ax.legend(loc='best', framealpha=0.9, fontsize=10)
            
            # Add annotations for best values
            if valid_val:
                is_lower_better = self.data.get("is_lower_better", False)
                best_val_idx = min(range(len(val_vals)), key=lambda i: val_vals[i] if is_lower_better else -val_vals[i])
                best_val = val_vals[best_val_idx]
                best_iter = val_iters[best_val_idx]
                ax.annotate(f'Best Val: {best_val:.4f}', 
                           xy=(best_iter, best_val),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Tight layout and save
            plt.tight_layout()
            plt.savefig(self.plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"   📈 Plot updated: {self.plot_file}")
            
        except Exception as e:
            logger.error(f"Error updating plot: {e}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if not self.data["iterations"]:
            return {}
        
        val_kpis = [it["val_kpi"] for it in self.data["iterations"] if it["val_kpi"] is not None]
        test_kpis = [it["test_kpi"] for it in self.data["iterations"] if it["test_kpi"] is not None]
        
        is_lower_better = self.data.get("is_lower_better", False)
        
        summary = {
            "total_iterations": len(self.data["iterations"]),
            "iterations_with_val_kpi": len(val_kpis),
            "iterations_with_test_kpi": len(test_kpis),
        }
        
        if val_kpis:
            summary["best_val_kpi"] = min(val_kpis) if is_lower_better else max(val_kpis)
            summary["worst_val_kpi"] = max(val_kpis) if is_lower_better else min(val_kpis)
            summary["mean_val_kpi"] = sum(val_kpis) / len(val_kpis)
        
        if test_kpis:
            summary["best_test_kpi"] = min(test_kpis) if is_lower_better else max(test_kpis)
            summary["worst_test_kpi"] = max(test_kpis) if is_lower_better else min(test_kpis)
            summary["mean_test_kpi"] = sum(test_kpis) / len(test_kpis)
            
            # Calculate val-test gap for reference
            if val_kpis and test_kpis:
                best_val = summary["best_val_kpi"]
                best_test = summary["best_test_kpi"]
                summary["val_test_gap"] = abs(best_val - best_test)
        
        return summary
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.get_summary()
        
        if not summary:
            print("No iterations tracked yet.")
            return
        
        print("\n" + "="*80)
        print("ITERATION TRACKING SUMMARY")
        print("="*80)
        print(f"Competition: {self.competition_id}")
        print(f"Total Iterations: {summary['total_iterations']}")
        print(f"Iterations with Val KPI: {summary['iterations_with_val_kpi']}")
        print(f"Iterations with Test KPI: {summary['iterations_with_test_kpi']}")
        
        if 'best_val_kpi' in summary:
            print(f"\n📊 Validation KPI:")
            print(f"   Best:  {summary['best_val_kpi']:.6f}")
            print(f"   Worst: {summary['worst_val_kpi']:.6f}")
            print(f"   Mean:  {summary['mean_val_kpi']:.6f}")
        
        if 'best_test_kpi' in summary:
            print(f"\n🏆 Test KPI:")
            print(f"   Best:  {summary['best_test_kpi']:.6f}")
            print(f"   Worst: {summary['worst_test_kpi']:.6f}")
            print(f"   Mean:  {summary['mean_test_kpi']:.6f}")
        
        if 'val_test_gap' in summary:
            print(f"\n📊 Val-Test Gap: {summary['val_test_gap']:.6f}")
        
        print("="*80)
        print(f"Plot saved to: {self.plot_file}")
        print(f"Data saved to: {self.tracking_file}")
        print()


def test_tracker():
    """Test the iteration tracker with dummy data."""
    import tempfile
    import shutil
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize tracker
        tracker = IterationTracker(
            output_dir=temp_dir,
            competition_id="spaceship-titanic"
        )
        
        # Simulate iterations
        for i in range(1, 11):
            val_kpi = 0.60 + (i * 0.02) + np.random.normal(0, 0.01)
            tracker.record_iteration(
                iteration=i,
                val_kpi=val_kpi,
                submission_path=None,  # No actual submissions in test
                code_version_id=f"test-{i}"
            )
        
        # Print summary
        tracker.print_summary()
        
        print(f"\nTest files created in: {temp_dir}")
        print("Check the plot and JSON file manually.")
        
    finally:
        # Cleanup (comment out if you want to inspect files)
        # shutil.rmtree(temp_dir)
        pass


if __name__ == "__main__":
    test_tracker()

