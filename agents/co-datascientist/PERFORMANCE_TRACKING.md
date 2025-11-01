# Performance Tracking & Overfitting Analysis

This document explains the automatic performance tracking system that helps you understand model selection quality and detect overfitting.

---

## Overview

The system automatically tracks:
1. **Baseline Validation KPI** - Starting point performance
2. **Evolution Statistics** - How many versions, iterations, best validation KPI achieved
3. **Top 3 Ensemble** - The 3 best models selected and their validation KPIs
4. **Final Test Score** - Official test set performance (added after grading)
5. **Overfitting Analysis** - Gap between validation and test performance

All tracked in a single JSON file: `submission/performance_summary.json`

---

## What Gets Tracked Automatically

### During the Run (Automatic)

When the benchmark completes, `handle_results.py` creates `performance_summary.json`:

```json
{
  "timestamp": "2025-10-31T18:30:00",
  "baseline": {
    "val_kpi": 0.68935,
    "description": "Initial baseline performance on validation set"
  },
  "evolution": {
    "iterations": 100,
    "total_versions_generated": 487,
    "best_val_kpi": 0.756547,
    "description": "Best validation KPI achieved during evolution"
  },
  "ensemble": {
    "top3_versions": ["uuid-1", "uuid-2", "uuid-3"],
    "top3_val_kpis": [0.756547, 0.752103, 0.748992],
    "ensemble_strategy": "mean for numeric, mode for categorical"
  },
  "final": {
    "test_kpi": null,  // Filled after grading
    "test_metric": null
  },
  "analysis": {
    "val_test_gap": null,  // Computed after grading
    "overfitting_detected": null
  }
}
```

### After Grading (Semi-Automatic)

Run the helper script to add test scores:

```bash
cd /home/ozkilim/Co-DataScientist_/mle-bench

# After grading completes
python agents/co-datascientist/utils/update_test_scores.py \
    runs/2025-10-31T17-54-52-GMT_run-group_co-datascientist
```

This updates the JSON with:
```json
{
  "final": {
    "test_kpi": 0.71487,
    "test_metric": "roc_auc"
  },
  "analysis": {
    "val_test_gap": 0.04166,  // 0.756547 - 0.71487
    "overfitting_detected": false  // Gap < 0.05
  }
}
```

---

## How to Use This

### 1. Run Benchmark (Automatic Tracking)

```bash
cd /home/ozkilim/Co-DataScientist_/mle-bench
./agents/co-datascientist/run_benchmark.sh experiments/splits/my-comps.txt 1 1
```

After completion, check for the summary file:
```bash
RUN_DIR=$(ls -td runs/*/ | head -1)
cat $RUN_DIR/*/submission/performance_summary.json
```

### 2. After Grading, Add Test Scores

```bash
# Update with actual test scores
python agents/co-datascientist/utils/update_test_scores.py $RUN_DIR
```

### 3. Analyze Overfitting

```bash
# View complete analysis
cat $RUN_DIR/*/submission/performance_summary.json | jq .

# Quick check for overfitting
cat $RUN_DIR/*/submission/performance_summary.json | jq '.analysis'
```

---

## Interpreting Results

### Example 1: Good Generalization ✅

```json
{
  "baseline": {"val_kpi": 0.6893},
  "evolution": {"best_val_kpi": 0.7200},
  "ensemble": {"top3_val_kpis": [0.7200, 0.7150, 0.7100]},
  "final": {"test_kpi": 0.7148},
  "analysis": {
    "val_test_gap": 0.0052,
    "overfitting_detected": false
  }
}
```

**Analysis:**
- ✅ Val KPI (0.72) ≈ Test KPI (0.7148)
- ✅ Small gap (0.0052)
- ✅ Model generalizes well
- ✅ Evolution was successful!

### Example 2: Mild Overfitting ⚠️

```json
{
  "baseline": {"val_kpi": 0.6893},
  "evolution": {"best_val_kpi": 0.7800},
  "ensemble": {"top3_val_kpis": [0.7800, 0.7750, 0.7700]},
  "final": {"test_kpi": 0.7250},
  "analysis": {
    "val_test_gap": 0.0550,
    "overfitting_detected": true
  }
}
```

**Analysis:**
- ⚠️ Val KPI (0.78) >> Test KPI (0.725)
- ⚠️ Significant gap (0.055)
- ⚠️ Models overfit to validation set
- 📝 Consider: fewer iterations, regularization, different validation split

### Example 3: Great Evolution! 🎯

```json
{
  "baseline": {"val_kpi": 0.6893},
  "evolution": {"best_val_kpi": 0.7565},
  "ensemble": {"top3_val_kpis": [0.7565, 0.7521, 0.7490]},
  "final": {"test_kpi": 0.7600},
  "analysis": {
    "val_test_gap": -0.0035,
    "overfitting_detected": false
  }
}
```

**Analysis:**
- 🎯 Test KPI (0.76) > Val KPI (0.7565)!
- ✅ Negative gap = under-optimized validation
- ✅ Model generalizes beyond what evolution saw
- 🎉 Excellent result!

---

## Advanced Analysis

### Compare Multiple Runs

```bash
# Extract key metrics from multiple runs
for run in runs/*/; do
    echo "Run: $(basename $run)"
    cat $run/*/submission/performance_summary.json | \
        jq '{baseline: .baseline.val_kpi, best_val: .evolution.best_val_kpi, test: .final.test_kpi, gap: .analysis.val_test_gap}'
    echo ""
done
```

### Create Comparison Table

```python
import json
from pathlib import Path
import pandas as pd

runs_dir = Path("runs")
data = []

for run_dir in runs_dir.glob("*/"):
    perf_files = list(run_dir.glob("*/submission/performance_summary.json"))
    
    for perf_file in perf_files:
        with open(perf_file) as f:
            summary = json.load(f)
        
        data.append({
            "run": run_dir.name,
            "iterations": summary["evolution"]["iterations"],
            "versions": summary["evolution"]["total_versions_generated"],
            "baseline_val": summary["baseline"]["val_kpi"],
            "best_val": summary["evolution"]["best_val_kpi"],
            "test": summary["final"]["test_kpi"],
            "gap": summary["analysis"]["val_test_gap"],
            "overfit": summary["analysis"]["overfitting_detected"]
        })

df = pd.DataFrame(data)
print(df.to_string(index=False))
```

---

## File Locations

After a run completes, you'll find:

```
runs/TIMESTAMP_run-group_co-datascientist/
└── competition-id_UUID/
    ├── submission/
    │   ├── submission.csv              # Final ensemble predictions
    │   ├── uuid-1.csv                  # Top version 1 predictions
    │   ├── uuid-2.csv                  # Top version 2 predictions  
    │   ├── uuid-3.csv                  # Top version 3 predictions
    │   └── performance_summary.json    # ⭐ METRICS FILE
    ├── code/
    │   └── *.py                        # All generated code versions
    └── logs/
        └── run.log                     # Complete execution log
```

---

## Integration with Existing Workflow

The tracking is **fully automatic** - no changes needed to your workflow!

**Before (Old Workflow)**:
```bash
1. Run benchmark
2. Check grading report
3. ??? (No validation KPIs tracked)
```

**After (New Workflow)**:
```bash
1. Run benchmark
   → performance_summary.json created automatically
   
2. Check grading report

3. Update test scores:
   python utils/update_test_scores.py <run_dir>
   
4. Analyze overfitting:
   cat */submission/performance_summary.json | jq .analysis
```

---

## Benefits

### 1. **Detect Overfitting**
Immediately see if validation scores don't translate to test performance.

### 2. **Tune Iterations**
Find the sweet spot: enough iterations to improve, not so many you overfit.

### 3. **Compare Strategies**
Test different configurations and see which generalizes best.

### 4. **Debug Issues**
If test score << val score, you know the problem is overfitting, not code bugs.

### 5. **Report Results**
Professional tracking of all metrics for papers/reports.

---

## FAQ

**Q: Does this slow down the benchmark?**  
A: No! Minimal overhead (~0.1 seconds to write JSON).

**Q: What if I don't run the update script?**  
A: The validation KPIs are still tracked. You just won't have the test score comparison.

**Q: Can I customize the overfitting threshold?**  
A: Yes! Edit `performance_tracker.py`, line ~60:
```python
self.summary["analysis"]["overfitting_detected"] = gap > 0.05  # Change 0.05 to your threshold
```

**Q: What if baseline isn't tracked?**  
A: It will be null in the JSON. The system still tracks evolution and ensemble KPIs.

**Q: Does this work for all competitions?**  
A: Yes! It's competition-agnostic. Tracks whatever metric the engine optimizes.

---

## Example Session

```bash
# 1. Run benchmark
cd /home/ozkilim/Co-DataScientist_/mle-bench
./agents/co-datascientist/run_benchmark.sh experiments/splits/test.txt 1 1

# 2. Check automatic tracking (during run)
tail -f runs/$(ls -t runs | head -1)/*/logs/run.log | grep "Current best KPI"

# 3. After completion, view validation metrics
RUN_DIR=$(ls -td runs/*/ | head -1)
cat $RUN_DIR/*/submission/performance_summary.json | jq .

# 4. After grading, add test scores
python agents/co-datascientist/utils/update_test_scores.py $RUN_DIR

# 5. Analyze
cat $RUN_DIR/*/submission/performance_summary.json | jq '.analysis'

# Output:
# {
#   "val_test_gap": 0.04166,
#   "overfitting_detected": false,
#   "description": "Computed after test results available"
# }
```

---

## Next Steps

For the **current 100-iteration run**, the performance tracking will work automatically on the next run! The current run (already started) won't have it, but you can manually check logs for validation KPIs.

**For future runs**, everything is automatic! 🎉

---

**Questions?** Check the code:
- `adapter/performance_tracker.py` - Core tracking logic
- `adapter/handle_results.py` - Integration with ensemble
- `utils/update_test_scores.py` - Post-grading updater

