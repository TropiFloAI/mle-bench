# Contains functions for handling results from cli runs and preparing submissions
from pathlib import Path
import sys
import shutil
import os
import pandas as pd
import numpy as np
import logging

# Import from the installed co_datascientist_engine package
from co_datascientist_engine.models import CodeResult

# Import run_python_code from local copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "local_runner"))
from code_runner import run_python_code

# Import performance tracker
sys.path.insert(0, str(Path(__file__).parent))
from performance_tracker import PerformanceTracker

logger = logging.getLogger(__name__)


def handle_results(results: list[tuple[id: str, CodeResult]], result_file_paths: dict[id:str, Path], interpreter_path="python"):

    logger.info(f"Handling {len(results)} code results:")
    logger.info(results)
    
    # Initialize performance tracker
    submission_dir = os.environ.get("SUBMISSION_DIR")
    tracker = PerformanceTracker(output_dir=submission_dir)
    
    # Track evolution statistics
    total_versions = len(results)
    iterations = int(os.environ.get("CLI_MAX_ITERATIONS", 0))
    
    # Get baseline (first valid result) if available
    baseline_result = next((r for r in results if r[1].kpi is not None), None)
    if baseline_result:
        tracker.set_baseline(val_kpi=baseline_result[1].kpi)
        logger.info(f"📊 Baseline validation KPI: {baseline_result[1].kpi}")
    
    # Get best validation KPI from all results
    valid_results = [(r[0], r[1]) for r in results if r[1].kpi is not None]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x[1].kpi)
        best_val_kpi = best_result[1].kpi
        tracker.set_evolution_stats(
            iterations=iterations,
            total_versions=total_versions,
            best_val_kpi=best_val_kpi
        )
        logger.info(f"🧬 Best validation KPI: {best_val_kpi} (from {total_versions} versions)")
    
    # Get files to ensemble and track top 3
    files, top3_info = get_files_to_ensemble(results, result_file_paths, interpreter_path)
    
    # Track ensemble selection
    tracker.set_ensemble(
        top3_ids=top3_info['ids'],
        top3_kpis=top3_info['kpis']
    )
    logger.info(f"🎯 Top 3 ensemble - Val KPIs: {top3_info['kpis']}")
    
    # Create ensemble
    ensemble_predictions(files, os.path.join(submission_dir, "submission.csv"))
    
    # Print summary
    tracker.print_summary()
    logger.info(f"📁 Performance summary saved to: {submission_dir}/performance_summary.json")

def get_files_to_ensemble(results, result_file_paths, interpreter_path="python"):

    # Number of files to ensemble is defined as an env variable
    best_code_ids = get_best_codes(results, int(os.environ.get("ENSEMBLE_N",3)))
    
    # Get KPIs for the top codes
    results_dict = {r[0]: r[1] for r in results}
    top3_kpis = [results_dict[code_id].kpi for code_id in best_code_ids 
                 if code_id in results_dict and results_dict[code_id].kpi is not None]

    # Naive and inefficient solution - run each top solution sequentially and rename the submission file for ensembling
    # TODO: set baselines to always output with code name rather than 'submission' - though this would hugely increase storage (1 file per code version)
    # Could also include code to delete file
    # Need to parallelise!

    files = []
    for code_id in best_code_ids:
        # Generate submission.csv
        with open(result_file_paths[code_id]) as file:
            code = file.read()

        logger.info(f"Running code at {result_file_paths[code_id]}...")
        run_python_code(code, interpreter_path)
        # Rename
        file_name = f"{code_id}.csv"
        original_output_path = os.path.join(os.environ.get("SUBMISSION_DIR"), "submission.csv")
        new_output_path =  os.path.join(os.environ.get("SUBMISSION_DIR"), file_name)
        shutil.move(original_output_path, new_output_path)
        files.append(new_output_path)
    
    # Return files and top3 info for tracking
    top3_info = {
        'ids': best_code_ids,
        'kpis': top3_kpis
    }
    return files, top3_info


def ensemble_predictions(files, output_file):
    
    logger.info(f"Generating ensemble of {len(files)} files: {files}")

    # Load all DataFrames
    dfs = [pd.read_csv(f) for f in files]

    # First column is the ID
    id_col = dfs[0].columns[0]

    # Merge all on the ID
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=id_col, suffixes=("", "_dup"))

    result = merged[[id_col]].copy()

    # For each prediction column
    for col in dfs[0].columns[1:]:
        # Collect matching columns from all files
        cols = [col] + [c for c in merged.columns if c.startswith(col + "_")]

        # Numeric → mean
        if np.issubdtype(dfs[0][col].dtype, np.number):
            result[col] = merged[cols].mean(axis=1)

        # Categorical → mode (tie → take value from first file)
        else:
            def resolve_mode(row):
                counts = row.value_counts()
                max_count = counts.max()
                top = counts[counts == max_count].index
                if len(top) == 1:
                    return top[0]
                return row.iloc[0]  # tie → first file’s value

            result[col] = merged[cols].apply(resolve_mode, axis=1)

    result.to_csv(output_file, index=False)
    print(f"Ensembled predictions written to {output_file}")



# Return ids of highest kpi code results
def get_best_codes(results: list[tuple[id: str, CodeResult]], n=3):
    # Get ids and kpis, make sure to omit failed results (i.e. kpi is None)
    kpis = [(r[0], r[1].kpi) for r in results if r[1].kpi is not None ]
    kpis.sort(key=lambda k: k[1], reverse=True)
    return [k[0] for k in kpis][:n]



# test_files = [
#     "/home/harry/co-datascientist-engine/co_datascientist_output/38d12b2c-049f-47cd-af66-279a02d28090.csv",
#     "/home/harry/co-datascientist-engine/co_datascientist_output/43a189a7-2ac9-4ff9-9ffb-be84f6b1bfa7.csv",
#     "/home/harry/co-datascientist-engine/co_datascientist_output/e32cc8fe-0713-46df-8f5d-11d061bed059.csv"
# ]

# ensemble_files(test_files, "/home/harry/co-datascientist-engine/co_datascientist_output/submission.csv")