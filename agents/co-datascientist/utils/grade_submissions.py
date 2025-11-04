#!/usr/bin/env python3
"""
Host-side grading script for co-datascientist iteration tracking.

This script:
1. Reads iteration_tracking.json from a run directory
2. Finds all submission_iter*.csv files  
3. Grades each submission using mlebench (which is properly set up on host)
4. Updates iteration_tracking.json with test KPIs
5. Regenerates the plot with both validation and test KPIs

Usage:
    python grade_submissions.py <run_directory>
    
Example:
    python grade_submissions.py runs/2025-11-04T13-00-00-UTC_run-group_co-datascientist/random-acts-of-pizza_abc123/
"""

import sys
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Import mlebench for grading (works on host where data is properly prepared)
from mlebench.grade import grade_csv
from mlebench.registry import registry as DEFAULT_REGISTRY
from mlebench.data import get_leaderboard


def grade_submissions(run_dir: Path):
    """Grade all submissions in a run directory and update tracking data."""
    
    submission_dir = run_dir / "submission"
    tracking_file = submission_dir / "iteration_tracking.json"
    
    if not tracking_file.exists():
        print(f"❌ No iteration tracking file found at {tracking_file}")
        return
    
    # Load tracking data
    with open(tracking_file, 'r') as f:
        data = json.load(f)
    
    competition_id = data["competition_id"]
    print(f"📊 Grading submissions for competition: {competition_id}")
    
    # Get competition and medal thresholds
    competition = DEFAULT_REGISTRY.get_competition(competition_id)
    leaderboard = get_leaderboard(competition)
    rank_info = competition.grader.rank_score(None, leaderboard)
    is_lower_better = competition.grader.is_lower_better(leaderboard)
    
    # Update medal thresholds in data
    data["medal_thresholds"] = {
        "gold": rank_info["gold_threshold"],
        "silver": rank_info["silver_threshold"],
        "bronze": rank_info["bronze_threshold"],
        "median": leaderboard["Score"].median() if "Score" in leaderboard else None
    }
    data["is_lower_better"] = is_lower_better
    
    # Grade each iteration's submission
    graded_count = 0
    for iteration_data in data["iterations"]:
        iteration = iteration_data["iteration"]
        
        # Find submission file for this iteration
        submission_file = submission_dir / f"submission_iter{iteration}.csv"
        
        # If iteration-specific file doesn't exist, try the main submission.csv
        if not submission_file.exists() and iteration == len(data["iterations"]):
            submission_file = submission_dir / "submission.csv"
        
        if not submission_file.exists():
            print(f"   ⚠️  No submission file for iteration {iteration}")
            continue
        
        # Skip if already graded
        if iteration_data.get("test_kpi") is not None:
            print(f"   ✓ Iteration {iteration} already graded (test_kpi={iteration_data['test_kpi']:.6f})")
            continue
        
        try:
            print(f"   🔍 Grading iteration {iteration}...")
            report = grade_csv(submission_file, competition)
            test_kpi = report.score
            
            # Update iteration data
            iteration_data["test_kpi"] = test_kpi
            iteration_data["test_score_details"] = {
                "score": test_kpi,
                "gold_medal": report.gold_medal,
                "silver_medal": report.silver_medal,
                "bronze_medal": report.bronze_medal,
                "above_median": report.above_median
            }
            
            medal = ""
            if report.gold_medal:
                medal = "🥇 GOLD"
            elif report.silver_medal:
                medal = "🥈 SILVER"
            elif report.bronze_medal:
                medal = "🥉 BRONZE"
            elif report.above_median:
                medal = "📈 Above Median"
            
            print(f"       ✅ Test KPI: {test_kpi:.6f} {medal}")
            graded_count += 1
            
        except Exception as e:
            print(f"       ❌ Error grading iteration {iteration}: {e}")
            continue
    
    if graded_count == 0:
        print("❌ No new submissions were graded")
        return
    
    # Save updated tracking data
    with open(tracking_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✅ Graded {graded_count} submission(s)")
    print(f"📄 Updated: {tracking_file}")
    
    # Regenerate plot with test KPIs
    plot_file = submission_dir / "kpi_progression.png"
    generate_plot(data, plot_file)
    print(f"📈 Updated plot: {plot_file}")


def generate_plot(data, plot_file):
    """Generate KPI progression plot with both validation and test KPIs."""
    
    iterations = [item["iteration"] for item in data["iterations"]]
    val_kpis = [item["val_kpi"] for item in data["iterations"]]
    test_kpis = [item.get("test_kpi") for item in data["iterations"]]
    timestamps = [item["timestamp"] for item in data["iterations"]]
    
    plt.figure(figsize=(12, 7))
    
    # Plot validation KPIs
    plt.plot(iterations, val_kpis, 'o-', label='Validation KPI', linewidth=2, markersize=8, color='#2E86AB')
    
    # Plot test KPIs (only where available)
    test_iterations = [it for it, kpi in zip(iterations, test_kpis) if kpi is not None]
    test_kpi_values = [kpi for kpi in test_kpis if kpi is not None]
    if test_kpi_values:
        plt.plot(test_iterations, test_kpi_values, 's-', label='Test KPI', linewidth=2, markersize=8, color='#F77F00')
    
    # Add medal threshold lines
    thresholds = data.get("medal_thresholds", {})
    is_lower_better = data.get("is_lower_better", False)
    
    if thresholds.get("gold") is not None:
        plt.axhline(y=thresholds["gold"], color='gold', linestyle='--', alpha=0.7, label=f'Gold ({thresholds["gold"]:.4f})')
    if thresholds.get("silver") is not None:
        plt.axhline(y=thresholds["silver"], color='silver', linestyle='--', alpha=0.7, label=f'Silver ({thresholds["silver"]:.4f})')
    if thresholds.get("bronze") is not None:
        plt.axhline(y=thresholds["bronze"], color='#CD7F32', linestyle='--', alpha=0.7, label=f'Bronze ({thresholds["bronze"]:.4f})')
    if thresholds.get("median") is not None:
        plt.axhline(y=thresholds["median"], color='gray', linestyle=':', alpha=0.5, label=f'Median ({thresholds["median"]:.4f})')
    
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('KPI Score', fontsize=12)
    plt.title(f'KPI Progression - {data["competition_id"]}', fontsize=14, fontweight='bold')
    
    # Add timestamp labels to x-axis
    if iterations and timestamps:
        from datetime import datetime as dt
        time_labels = []
        for ts in timestamps:
            try:
                parsed_ts = dt.fromisoformat(ts)
                time_labels.append(parsed_ts.strftime('%H:%M:%S'))
            except:
                time_labels.append('')
        
        # Set custom x-tick labels showing iteration and time
        plt.xticks(iterations, [f"#{iter}\n{time}" for iter, time in zip(iterations, time_labels)], fontsize=9)
    
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Annotate best validation KPI
    if val_kpis:
        best_idx = np.argmax(val_kpis) if not is_lower_better else np.argmin(val_kpis)
        best_val = val_kpis[best_idx]
        best_iter = iterations[best_idx]
        plt.annotate(f'Best Val: {best_val:.4f}',
                    xy=(best_iter, best_val),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python grade_submissions.py <run_directory>")
        print("\nExample:")
        print("  python grade_submissions.py runs/2025-11-04T13-00-00-UTC_run-group_co-datascientist/random-acts-of-pizza_abc123/")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"❌ Run directory does not exist: {run_dir}")
        sys.exit(1)
    
    grade_submissions(run_dir)

