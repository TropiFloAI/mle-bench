#!/bin/bash
# Monitor all running competitions with pretty table

# Auto-detect most recent run directory
RUN_DIR=$(ls -td runs/*_run-group_co-datascientist 2>/dev/null | head -1)

if [ -z "$RUN_DIR" ]; then
  echo "❌ No run directories found!"
  echo "Looking for: runs/*_run-group_co-datascientist"
  exit 1
fi

echo "🎯 CHONK-CPU BENCHMARK RUN - REAL-TIME MONITOR"
echo "==============================================="
echo ""

# Use Python to parse JSON and create nice table
python3 << PYTHON_SCRIPT
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone

RUN_DIR = "$RUN_DIR"

def format_runtime(seconds):
    """Format runtime in human-readable format"""
    if seconds is None:
        return "-"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"

# Collect data from all competitions
competitions = []

for task_dir in sorted(Path(RUN_DIR).glob("*")):
    if not task_dir.is_dir():
        continue
    
    comp_name = task_dir.name.rsplit('_', 1)[0]
    tracking_file = task_dir / "submission" / "iteration_tracking.json"
    
    if not tracking_file.exists():
        competitions.append({
            'name': comp_name,
            'status': 'initializing',
            'iterations': 0,
            'runtime': None,
            'baseline_val': None,
            'baseline_test': None,
            'best_val': None,
            'best_test': None,
            'medal': '',
            'has_plot': False
        })
        continue
    
    try:
        with open(tracking_file, 'r') as f:
            data = json.load(f)
        
        iterations = data.get('iterations', [])
        num_iters = len(iterations)
        
        # Calculate runtime from start to NOW (both in UTC)
        runtime_seconds = None
        started_at_str = data.get('started_at')
        if started_at_str:
            try:
                # Parse the UTC timestamp (no timezone info in the string, but it's UTC)
                started_at = datetime.fromisoformat(started_at_str)
                # Get current UTC time (timezone-aware)
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                runtime_seconds = (now_utc - started_at).total_seconds()
            except Exception as e:
                pass
        
        # Get baseline (first iteration) KPIs
        baseline_val = None
        baseline_test = None
        if iterations:
            first_iter = iterations[0]
            baseline_val = first_iter.get('val_kpi')
            baseline_test = first_iter.get('test_kpi')
        
        # Find best val and test KPIs
        best_val = None
        best_test = None
        medal = ''
        
        # Determine if lower is better
        is_lower_better = data.get('is_lower_better', False)
        
        for iteration in iterations:
            val_kpi = iteration.get('val_kpi')
            test_kpi = iteration.get('test_kpi')
            
            if val_kpi is not None:
                if best_val is None:
                    best_val = val_kpi
                elif is_lower_better:
                    best_val = min(best_val, val_kpi)
                else:
                    best_val = max(best_val, val_kpi)
            
            if test_kpi is not None:
                if best_test is None:
                    best_test = test_kpi
                elif is_lower_better:
                    best_test = min(best_test, test_kpi)
                else:
                    best_test = max(best_test, test_kpi)
                
                # Check medal status for best test KPI
                score_details = iteration.get('test_score_details', {})
                if test_kpi == best_test:
                    if score_details.get('gold_medal'):
                        medal = '🥇 GOLD'
                    elif score_details.get('silver_medal'):
                        medal = '🥈 SILVER'
                    elif score_details.get('bronze_medal'):
                        medal = '🥉 BRONZE'
                    elif score_details.get('above_median'):
                        medal = '📊 Above Median'
                    else:
                        medal = '⚪ Below Median'
        
        plot_file = task_dir / "submission" / "kpi_progression.png"
        has_plot = plot_file.exists()
        
        competitions.append({
            'name': comp_name,
            'status': 'running' if num_iters > 0 else 'initializing',
            'iterations': num_iters,
            'runtime': runtime_seconds,
            'baseline_val': baseline_val,
            'baseline_test': baseline_test,
            'best_val': best_val,
            'best_test': best_test,
            'medal': medal,
            'has_plot': has_plot
        })
    
    except Exception as e:
        competitions.append({
            'name': comp_name,
            'status': 'error',
            'iterations': 0,
            'runtime': None,
            'baseline_val': None,
            'baseline_test': None,
            'best_val': None,
            'best_test': None,
            'medal': '',
            'has_plot': False
        })

# Print header
print("📊 COMPETITION PROGRESS & PERFORMANCE")
print("=" * 175)
print(f"{'Competition':<43} {'Iter':<6} {'Runtime':<10} {'Base Val':<13} {'Best Val':<13} {'Base Test':<13} {'Best Test':<13} {'Medal':<20}")
print("-" * 175)

# Print each competition
for comp in competitions:
    name_display = comp['name'][:41] if len(comp['name']) > 41 else comp['name']
    
    if comp['status'] == 'initializing':
        runtime_str = format_runtime(comp['runtime'])
        print(f"{name_display:<43} {'⏳':<6} {runtime_str:<10} {'-':<13} {'-':<13} {'-':<13} {'-':<13} {'Initializing...':<20}")
    else:
        iters_str = str(comp['iterations'])
        if comp['has_plot']:
            iters_str += "📊"
        
        runtime_str = format_runtime(comp['runtime'])
        base_val_str = f"{comp['baseline_val']:.6f}" if comp['baseline_val'] is not None else "-"
        best_val_str = f"{comp['best_val']:.6f}" if comp['best_val'] is not None else "-"
        base_test_str = f"{comp['baseline_test']:.6f}" if comp['baseline_test'] is not None else "pending..."
        best_test_str = f"{comp['best_test']:.6f}" if comp['best_test'] is not None else "pending..."
        medal_str = comp['medal'] if comp['medal'] else "-"
        
        print(f"{name_display:<43} {iters_str:<6} {runtime_str:<10} {base_val_str:<13} {best_val_str:<13} {base_test_str:<13} {best_test_str:<13} {medal_str:<20}")

print("=" * 175)

PYTHON_SCRIPT

echo ""
echo "📂 Active Run: $RUN_DIR"
echo ""
echo "🐳 Docker Containers: $(docker ps -q | grep -c . || echo 0) running"
docker ps --format "  • {{.Names}}" | grep competition | sed 's/competition-//' | sed 's/-2025.*//' | sort

echo ""
echo "💡 Quick Commands:"
echo "  View plot:  open $RUN_DIR/<competition>*/submission/kpi_progression.png"
echo "  Watch log:  tail -f $RUN_DIR/<competition>*/run.log"
echo ""
