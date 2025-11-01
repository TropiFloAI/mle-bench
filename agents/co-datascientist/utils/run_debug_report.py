# This script takes a run directory and a text file with a competition list, and creates a debug_report.json file, with the reported KPI and actual scores of the files used for the ensemble, compared with the baselines

import os
import json
import sys
from pathlib import Path
import subprocess
import re

if len(sys.argv) != 3:
    print("Usage: python script.py <directory> <competition_ids.txt>")
    sys.exit(1)

run_dir = sys.argv[1]
txt_path = sys.argv[2]

# Generate a json of stats from a given run directory
def collect_stats(run_dir, comp_ids):
    results = {}
    for comp_id in comp_ids:

        results[comp_id] = {}

        # Find competition subfolder starting with comp_id
        comp_folder = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith(comp_id)][0]
        submission_dir = comp_folder / "submission"

        if submission_dir.is_dir():
            # Collect submission.csv and other CSVs
            for csv_file in submission_dir.glob("*.csv"):
                if csv_file.name.__contains__("submission"):
                    continue
                else:
                    # Extract someid (filename without .csv)
                    code_id = csv_file.stem
                    # Look up lineage result file
                    lineage_result_file = comp_folder / "code" / "lineage" / code_id / f"{code_id}_result.json"
                    with open(lineage_result_file, "r") as file:
                        results[comp_id][code_id] = json.load(file)# Pretty-print summary

                    

                # Get the actual score for that submission by running mlebench grade-sample
                print(f"Running mlebench grade-sample on {csv_file}")
                grade = subprocess.run(
                    ["/home/harry/mle-bench/venv/bin/mlebench", "grade-sample", csv_file, comp_id],
                    capture_output=True,
                    text=True
                )
                output = grade.stdout + grade.stderr
                match = re.search(r"\{[\s\S]*\}", output)
                json_output = match.group(0)
                report = json.loads(json_output)
                results[comp_id][code_id]["actual_score"] = report["score"]

    return results


with open(txt_path, 'r') as f:
    comp_ids = [line.strip() for line in f if line.strip()]

results = collect_stats(run_dir, comp_ids)

# Get baseline from the pre-existing baseline report
with open("/home/harry/co-datascientist-engine/mle-bench-agent/utils/baseline_debug_report.json", "r") as f:
    baseline_grades = json.load(f)

for comp in results:
    results[comp]["baseline"] = baseline_grades[comp]["baseline"]    

with open(run_dir / "debug_report.json", "w") as file:
    json.dump(results, file, indent=2)



# Generate a succinct report without stderr or stdout
for comp in results:
        for code in results[comp]:
            results[comp][code].pop("stderr")
            results[comp][code].pop("stdout")
with open(run_dir / "debug_report_succinct.json", "w") as file:
    json.dump(results, file, indent=2)