"""
Performance Tracker for MLE-Bench Runs

Tracks validation KPIs vs test KPIs to analyze overfitting and model selection quality.
Creates a summary file with all metrics for easy post-run analysis.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


class PerformanceTracker:
    """Tracks and saves performance metrics for analysis."""
    
    def __init__(self, output_dir: str):
        """
        Args:
            output_dir: Directory to save the performance summary (usually submission dir)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "performance_summary.json"
        
        self.summary = {
            "timestamp": datetime.now().isoformat(),
            "baseline": {
                "val_kpi": None,
                "description": "Initial baseline performance on validation set"
            },
            "evolution": {
                "iterations": None,
                "total_versions_generated": None,
                "best_val_kpi": None,
                "description": "Best validation KPI achieved during evolution"
            },
            "ensemble": {
                "top3_versions": [],
                "top3_val_kpis": [],
                "ensemble_strategy": "mean for numeric, mode for categorical",
                "description": "Top 3 models selected for final ensemble"
            },
            "final": {
                "test_kpi": None,
                "test_metric": None,
                "description": "Official test set performance (filled after grading)"
            },
            "analysis": {
                "val_test_gap": None,
                "overfitting_detected": None,
                "description": "Computed after test results available"
            }
        }
    
    def set_baseline(self, val_kpi: float):
        """Record baseline validation KPI."""
        self.summary["baseline"]["val_kpi"] = val_kpi
        self._save()
    
    def set_evolution_stats(self, iterations: int, total_versions: int, best_val_kpi: float):
        """Record evolution statistics."""
        self.summary["evolution"]["iterations"] = iterations
        self.summary["evolution"]["total_versions_generated"] = total_versions
        self.summary["evolution"]["best_val_kpi"] = best_val_kpi
        self._save()
    
    def set_ensemble(self, top3_ids: List[str], top3_kpis: List[float]):
        """
        Record the top 3 models selected for ensemble.
        
        Args:
            top3_ids: List of version IDs (e.g., UUIDs)
            top3_kpis: Corresponding validation KPIs
        """
        self.summary["ensemble"]["top3_versions"] = top3_ids
        self.summary["ensemble"]["top3_val_kpis"] = top3_kpis
        self._save()
    
    def set_test_results(self, test_kpi: float, metric_name: str = "roc_auc"):
        """
        Record final test results (call this after grading).
        
        Args:
            test_kpi: Test set performance
            metric_name: Name of the metric used
        """
        self.summary["final"]["test_kpi"] = test_kpi
        self.summary["final"]["test_metric"] = metric_name
        
        # Compute analysis metrics
        best_val = self.summary["evolution"]["best_val_kpi"]
        if best_val is not None:
            gap = best_val - test_kpi
            self.summary["analysis"]["val_test_gap"] = gap
            
            # Simple overfitting heuristic: val significantly better than test
            # (You can adjust this threshold based on your experience)
            self.summary["analysis"]["overfitting_detected"] = gap > 0.05
        
        self._save()
    
    def _save(self):
        """Save summary to JSON file."""
        with open(self.summary_path, 'w') as f:
            json.dump(self.summary, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Return current summary."""
        return self.summary
    
    def print_summary(self):
        """Print formatted summary to console."""
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        print(f"\n📊 BASELINE:")
        print(f"   Validation KPI: {self.summary['baseline']['val_kpi']}")
        
        print(f"\n🧬 EVOLUTION ({self.summary['evolution']['iterations']} iterations):")
        print(f"   Total versions generated: {self.summary['evolution']['total_versions_generated']}")
        print(f"   Best validation KPI: {self.summary['evolution']['best_val_kpi']}")
        
        print(f"\n🎯 ENSEMBLE (Top 3):")
        for i, (vid, kpi) in enumerate(zip(
            self.summary['ensemble']['top3_versions'],
            self.summary['ensemble']['top3_val_kpis']
        ), 1):
            print(f"   {i}. {vid[:8]}... → Val KPI: {kpi}")
        
        if self.summary['final']['test_kpi'] is not None:
            print(f"\n🏆 FINAL TEST RESULTS:")
            print(f"   Test KPI ({self.summary['final']['test_metric']}): {self.summary['final']['test_kpi']}")
            print(f"   Val-Test Gap: {self.summary['analysis']['val_test_gap']:.5f}")
            
            if self.summary['analysis']['overfitting_detected']:
                print(f"   ⚠️  Overfitting detected (val >> test)")
            else:
                print(f"   ✅ Good generalization (val ≈ test)")
        else:
            print(f"\n⏳ Test results not yet available")
        
        print("="*80)
        print(f"\nSummary saved to: {self.summary_path}")
        print()


# Example usage function that can be called from handle_results.py
def track_performance_example():
    """Example of how to use the PerformanceTracker."""
    
    # Initialize tracker
    tracker = PerformanceTracker(output_dir="/home/submission")
    
    # During baseline run
    tracker.set_baseline(val_kpi=0.68)
    
    # After evolution completes
    tracker.set_evolution_stats(
        iterations=20,
        total_versions=102,
        best_val_kpi=0.72
    )
    
    # When selecting ensemble
    tracker.set_ensemble(
        top3_ids=["uuid-1", "uuid-2", "uuid-3"],
        top3_kpis=[0.72, 0.71, 0.70]
    )
    
    # After grading (this could be called by a separate script)
    tracker.set_test_results(test_kpi=0.71487, metric_name="roc_auc")
    
    # Print summary
    tracker.print_summary()


if __name__ == "__main__":
    # Run example
    track_performance_example()

