#!/usr/bin/env python3
"""
Utility to update performance_summary.json with final test KPIs from grading report.
Usage: python update_test_scores.py <run_group_dir>
"""

import json
import sys
from pathlib import Path
from datetime import datetime


class PerformanceTracker:
    """Simple version of PerformanceTracker for updating test scores"""
    def __init__(self, data_dict):
        self.data = data_dict
    
    def set_final_test_kpi(self, test_kpi: float):
        self.data["final"]["test_kpi"] = test_kpi
        self._analyze_overfitting()
    
    def _analyze_overfitting(self):
        val_kpi = self.data["evolution"].get("best_val_kpi")
        test_kpi = self.data["final"].get("test_kpi")
        
        if val_kpi is not None and test_kpi is not None:
            val_test_gap = val_kpi - test_kpi
            self.data["analysis"]["val_test_gap"] = round(val_test_gap, 5)
            self.data["analysis"]["overfitting_detected"] = val_test_gap > 0.05
            
            # Add interpretation
            if val_test_gap > 0.1:
                self.data["analysis"]["interpretation"] = "Significant overfitting detected"
            elif val_test_gap > 0.05:
                self.data["analysis"]["interpretation"] = "Moderate overfitting detected"
            elif val_test_gap > 0:
                self.data["analysis"]["interpretation"] = "Slight overfitting"
            elif val_test_gap < -0.05:
                self.data["analysis"]["interpretation"] = "Model generalizes better than validation suggests"
            else:
                self.data["analysis"]["interpretation"] = "Good generalization"
        else:
            self.data["analysis"]["val_test_gap"] = None
            self.data["analysis"]["overfitting_detected"] = None


def main():
    if len(sys.argv) != 2:
        print("Usage: python update_test_scores.py <run_group_dir>")
        print("Example: python update_test_scores.py /path/to/runs/2025-10-31T20-53-15-GMT_run-group_co-datascientist")
        sys.exit(1)
    
    run_group_dir = Path(sys.argv[1])
    
    if not run_group_dir.exists():
        print(f"Error: Directory not found: {run_group_dir}")
        sys.exit(1)
    
    # Find the performance_summary.json (should be in submission dir of the competition run)
    summary_files = list(run_group_dir.glob("*/submission/performance_summary.json"))
    
    if not summary_files:
        print(f"Error: performance_summary.json not found in {run_group_dir}")
        print("Expected path: <run_group_dir>/<competition_run_id>/submission/performance_summary.json")
        sys.exit(1)
    
    if len(summary_files) > 1:
        print(f"Warning: Multiple performance_summary.json files found. Using the first one.")
    
    summary_path = summary_files[0]
    print(f"Found performance summary: {summary_path}")
    
    # Find the grading report
    grading_reports = list(run_group_dir.glob("*_grading_report.json"))
    
    if not grading_reports:
        print(f"Error: Grading report not found in {run_group_dir}")
        print("Make sure grading has completed.")
        sys.exit(1)
    
    grading_report_path = grading_reports[0]
    print(f"Found grading report: {grading_report_path}")
    
    # Load data
    with open(summary_path, "r") as f:
        performance_data = json.load(f)
    
    with open(grading_report_path, "r") as f:
        grading_data = json.load(f)
    
    # Extract test KPI from grading report
    # The grading report has a list of competition_reports, each with a score
    if not grading_data.get("competition_reports"):
        print("Error: No competition reports found in grading data")
        sys.exit(1)
    
    # Get the competition ID from the performance summary path
    competition_run_dir = summary_path.parent.parent.name
    competition_id = competition_run_dir.split("_")[0]  # e.g., "random-acts-of-pizza_<uuid>"
    
    # Find matching competition report
    test_kpi = None
    for report in grading_data["competition_reports"]:
        if report["competition_id"] == competition_id:
            test_kpi = report.get("score")
            break
    
    if test_kpi is None:
        print(f"Error: Could not find test score for competition '{competition_id}' in grading report")
        sys.exit(1)
    
    print(f"\n✅ Test KPI found: {test_kpi}")
    
    # Update performance data
    tracker = PerformanceTracker(performance_data)
    tracker.set_final_test_kpi(test_kpi)
    
    # Add update timestamp
    performance_data["updated_at"] = datetime.now().isoformat()
    
    # Save updated data
    with open(summary_path, "w") as f:
        json.dump(performance_data, f, indent=2)
    
    print(f"\n✅ Updated performance_summary.json with final test KPI")
    print(f"\n📊 Performance Analysis:")
    print(f"   - Baseline Val KPI:     {performance_data['baseline'].get('val_kpi', 'N/A')}")
    print(f"   - Best Val KPI:         {performance_data['evolution'].get('best_val_kpi', 'N/A')}")
    print(f"   - Final Test KPI:       {performance_data['final'].get('test_kpi', 'N/A')}")
    print(f"   - Val-Test Gap:         {performance_data['analysis'].get('val_test_gap', 'N/A')}")
    print(f"   - Interpretation:       {performance_data['analysis'].get('interpretation', 'N/A')}")
    print(f"\n📁 Updated file: {summary_path}")


if __name__ == "__main__":
    main()
