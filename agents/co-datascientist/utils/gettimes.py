# AI-generated script - pulls out the time from all the run.log files in a given run folder
import os
import re
import csv

# Root directory (change this if needed)
root_dir = "/home/harry/runs/2025-10-16T15-03-41-GMT_run-group_co-datascientist"

# Regex to capture "Run completed in X seconds."
pattern = re.compile(r"Run completed in ([\d.]+) seconds")

results = []

for subdir, dirs, files in os.walk(root_dir):
    if "run.log" in files:
        log_path = os.path.join(subdir, "run.log")
        with open(log_path, "r") as f:
            content = f.read()
            match = pattern.search(content)
            if match:
                seconds = float(match.group(1))
                folder_name = os.path.basename(subdir)
                results.append((folder_name, seconds))

# Sort results by runtime (optional)
results.sort(key=lambda x: x[1])

# Print summary
print(f"{'Folder':<40} {'Seconds':>10}")
print("-" * 52)
for folder, seconds in results:
    print(f"{folder:<40} {seconds:>10.2f}")

# Write to CSV
csv_path = os.path.join(root_dir, "run_times.csv")
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Folder", "Seconds"])
    writer.writerows(results)

print(f"\n✅ Results saved to {csv_path}")
