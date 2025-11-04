#!/usr/bin/env python3
"""
Auto-grader watcher that monitors for grading trigger files and grades automatically.

This script watches the submission directory for .grade_trigger_iterN files created
by the container, then automatically runs grading when detected.

Usage:
    python auto_grader_watcher.py <run_directory>

Example:
    python auto_grader_watcher.py runs/2025-11-04T12-00-00-UTC_run-group_co-datascientist/random-acts-of-pizza_abc123/
"""

import sys
import time
from pathlib import Path
import subprocess

def watch_and_grade(run_dir: Path, check_interval: int = 5):
    """
    Watch for grading trigger files and run grading automatically.
    
    Args:
        run_dir: Path to the run directory
        check_interval: How often to check for new triggers (seconds)
    """
    submission_dir = run_dir / "submission"
    graded_iterations = set()
    grading_script = Path(__file__).parent / "grade_submissions.py"
    
    print(f"🔍 Watching for grading triggers in: {submission_dir}")
    print(f"📊 Will grade automatically when new iterations complete")
    print(f"⏱️  Check interval: {check_interval} seconds")
    print(f"🛑 Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            # Find all trigger files
            trigger_files = list(submission_dir.glob(".grade_trigger_iter*"))
            
            for trigger_file in trigger_files:
                # Extract iteration number from filename
                try:
                    iter_num = int(trigger_file.name.replace(".grade_trigger_iter", ""))
                except ValueError:
                    continue
                
                # Skip if already graded
                if iter_num in graded_iterations:
                    continue
                
                print(f"🎯 Detected new iteration {iter_num} - triggering grading...")
                
                # Run grading script
                try:
                    result = subprocess.run(
                        [sys.executable, str(grading_script), str(run_dir)],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if result.returncode == 0:
                        print(f"   ✅ Grading completed for iteration {iter_num}")
                        graded_iterations.add(iter_num)
                        # Clean up trigger file
                        trigger_file.unlink()
                    else:
                        print(f"   ⚠️  Grading failed for iteration {iter_num}:")
                        print(f"       {result.stderr}")
                except subprocess.TimeoutExpired:
                    print(f"   ⚠️  Grading timeout for iteration {iter_num}")
                except Exception as e:
                    print(f"   ❌ Error grading iteration {iter_num}: {e}")
            
            # Wait before next check
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print()
        print("🛑 Watcher stopped by user")
        print(f"📊 Graded {len(graded_iterations)} iteration(s): {sorted(graded_iterations)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python auto_grader_watcher.py <run_directory>")
        print("\nExample:")
        print("  python auto_grader_watcher.py runs/2025-11-04T13-00-00-UTC_run-group_co-datascientist/random-acts-of-pizza_abc123/")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"❌ Run directory does not exist: {run_dir}")
        sys.exit(1)
    
    watch_and_grade(run_dir)

