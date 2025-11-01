import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
import time
import pandas as pd

from dotenv import load_dotenv

# Import from the installed co_datascientist_engine package
from co_datascientist_engine.models import CodeResult
from co_datascientist_engine import EngineType, CodeVersion, SystemInfo
import co_datascientist_engine

# Import run_python_code from local copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "local_runner"))
from code_runner import run_python_code

load_dotenv()
OUTPUT_FOLDER = os.environ.get("CODE_DIR","co_datascientist_output")


def load_task_description():
    """
    Load the task description markdown file from the data directory.
    This provides context about the competition/task to the agent.
    
    MLE-bench mounts each competition's specific public_dir to /home/data,
    so description.md is always task-specific for the running competition.
    """
    data_dir = os.environ.get("DATA_DIR", "/home/data")
    competition_id = os.environ.get("COMPETITION_ID", "unknown")
    description_path = Path(data_dir) / "description.md"
    
    try:
        if description_path.exists():
            with open(description_path, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"✓ Loaded task description for '{competition_id}' from {description_path} ({len(content)} chars)")
            return content
        else:
            print(f"⚠ Task description not found at {description_path} for competition '{competition_id}'")
            return None
    except Exception as e:
        print(f"⚠ Error loading task description for '{competition_id}': {e}")
        return None


def main():
    """
    Main routine to run the hypothesis evolution engine locally.
    
    This mimics the modern frontend's batch-based approach:
    1. Starts a workflow (with optional preflight)
    2. Runs the baseline
    3. Gets batches of hypotheses to test
    4. Executes each batch and reports results
    5. Continues until workflow is finished
    
    Enhanced with frontend CLI-like UX and better error handling.
    """
    setup_logging()
    args = parse_args()
    
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        sys.exit(1)
    except Exception as e:
        if args.verbosity > 1:
            import traceback
            traceback.print_exc()
        else:
            print(f"\nError: {e}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hypothesis evolution experiments locally (mimics modern frontend)."
    )

    parser.add_argument(
        "--script-path",
        type=str,
        required=True,
        help="Absolute path to the script to analyze and improve.",
    )
    parser.add_argument(
        "--interpreter-path",
        type=str,
        default="python",
        help="Path to the python interpreter executable.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of hypotheses to test in parallel (default: 1 for sequential)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the preflight Q&A phase and start directly",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("CLI_DEFAULT_TIMEOUT", "3600")),
        help="Timeout in seconds for experiment runs (default: from environment or 3600 seconds = 1 hour).",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        help="Logging verbosity level",
    )

    return parser.parse_args()


def print_header():
    """Print clean header"""
    print("Co-DataScientist Engine - Local Runner")
    print("=" * 50)
    print("Running locally without backend/frontend")
    print()


def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/cli_local.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )


async def run(args):
    """Main execution function matching the modern frontend approach"""
    code = Path(args.script_path).read_text()
    
    # Print enhanced header
    print_header()
    
    engine = await co_datascientist_engine.create_engine(EngineType.EVOLVE_HYPOTHESIS)
    print("Starting hypothesis evolution workflow")
    print(f"Script: {args.script_path}")
    print(f"Python: {args.interpreter_path}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 50)
    
    system_info = get_system_info(args.interpreter_path)
    
    # Start workflow (like the frontend)
    workflow, questions, observation, raw_text = await engine.start_workflow_preflight(
        user_id="local_runner_user",
        code={"code":code},
        system_info=system_info
    )
    print(f"Generated {len(questions)} preflight questions - for local runner, skipping Q&A")
    # In a real implementation, you'd ask the questions, but for local runner we skip
    workflow = await engine.complete_preflight(workflow.workflow_id, [])
    
    # Load task description from markdown file and inject as context
    task_description = load_task_description()
    if task_description:
        workflow.user_context_summary = task_description
        print(f"✓ Task description injected into workflow context")
    else:
        print(f"⚠ No task description available - agent will work without competition context")
    
    print(f"Workflow started: {workflow.workflow_id}")
    print("-" * 50)
    
    # Run the evolution loop using batch system
    iteration = 0
    start_time = time.time()
    all_results = []
    all_result_file_paths = {}

    while not workflow.finished and time.time()-start_time < args.timeout:
        iteration += 1
        print(f"\nIteration {iteration}")
        
        # Get batch of code to run (like frontend's get_batch_to_run)
        batch_to_run = await engine.get_batch_to_run(workflow.workflow_id, batch_size=args.batch_size)
        
        if batch_to_run is None:
            print("No code ready, waiting...")
            await asyncio.sleep(1)
            workflow = await engine.get_workflow(workflow.workflow_id)
            continue
            
        if len(batch_to_run) == 1:
            cv = batch_to_run[0]
            if cv.name == "baseline":
                print("Running baseline...")
            else:
                hypothesis = getattr(cv, 'hypothesis', None) or cv.idea
                print(f"Testing hypothesis: {hypothesis}")
        else:
            print(f"Testing {len(batch_to_run)} hypotheses in parallel:")
            for i, code_version in enumerate(batch_to_run, 1):
                hypothesis = getattr(code_version, 'hypothesis', None) or code_version.idea
                print(f"   {i}. {hypothesis[:80]}{'...' if len(hypothesis) > 80 else ''}")
        
        # Execute the batch in parallel
        async def run_code_version_async(code_version):
            """Run a single code version asynchronously"""
            if len(batch_to_run) > 1:
                print(f"\nTesting: {code_version.name}")
            
            # Run the code in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                run_python_code, 
                code_version.code["code"], 
                args.interpreter_path
            )
            
            # Extract KPI
            kpi_value = extract_kpi_from_result(result)
            result.kpi = kpi_value
            
            # Update code version
            code_version.result = result
            code_version.is_final = True
            
            # Display result with better formatting
            if result.return_code == 0:
                if kpi_value is not None:
                    if code_version.name == "baseline":
                        print(f"   SUCCESS: Baseline completed | KPI = {kpi_value:.6f}")
                    else:
                        # Compare with baseline if available
                        baseline_kpi = get_baseline_kpi(workflow)
                        if baseline_kpi is not None:
                            improved = kpi_value > baseline_kpi
                            status = "Improvement" if improved else "No improvement"
                            print(f"   {status} | KPI = {kpi_value:.6f}")
                        else:
                            print(f"   KPI = {kpi_value:.6f}")
                        
                    # If kpi value is returned then update workflow best kpi 
                    if workflow.best_kpi is None or result.kpi > workflow.best_kpi:
                        workflow.best_kpi = result.kpi
                        workflow.best_code_version = code_version
                else:
                    print(f"   WARNING: No KPI found in output")
            else:
                print(f"   ERROR: Failed with exit code {result.return_code}")
                if result.stderr:
                    print(f"      Error: {result.stderr[:100]}{'...' if len(result.stderr) > 100 else ''}")
            
            # Save results to files (like frontend)
            test_dir = "/home/ozkilim/Co-DataScientist_/mle-bench/runs/test_dir"
            result_file_path = save_code_version_to_file(code_version, result, test_dir)
            
            return (code_version.code_version_id, result, result_file_path)
        
        # Run all code versions in parallel
        results = await asyncio.gather(*[run_code_version_async(cv) for cv in batch_to_run])
        
        # Collect batch results
        batch_results = []
        for code_version_id, result, result_file_path in results:
            batch_results.append((code_version_id, result))
            all_result_file_paths[code_version_id] = result_file_path

        # TODO: some kpis are nan they should not be added
        
        # Report batch results back to engine (like frontend's finished_running_batch)
        try:
            # Use the batch API method if available
            if hasattr(engine, 'batch_finished_running'):
                await engine.batch_finished_running(workflow.workflow_id, batch_results)
            else:
                # Fallback to individual reporting
                for code_version_id, result in batch_results:
                    # Find the matching code version
                    code_version = next(cv for cv in batch_to_run if cv.code_version_id == code_version_id)
                    await engine.code_finished_running(
                        workflow.workflow_id, 
                        code_version, 
                        result, 
                        result.kpi
                    )
        except Exception as e:
            print(f"⚠️  Error reporting results: {e}")
            break
        
        # Get updated workflow state
        workflow = await engine.get_workflow(workflow.workflow_id)
        
        # Show progress
        if workflow.best_kpi is not None:
            print(f"Current best KPI: {workflow.best_kpi:.6f}")
        
        print("-" * 50)

        all_results = all_results + batch_results


        print("Generating runtime and KPI report")
        code_report = pd.DataFrame([{
            "id": row[0],
            "name": "baseline" if row[0]==workflow.baseline_code.code_version_id else "",
            "file": all_result_file_paths[row[0]],
            "kpi": row[1].kpi,
            "runtime_s": row[1].runtime_ms/1000,
            "error": row[1].stderr}
             for row in all_results] )
        code_report["file"] = code_report["file"].astype(str)
        code_report.map(str).to_json(Path(OUTPUT_FOLDER) / "code_report.json", orient="records", indent=2)        
        # Safety check to avoid infinite loops
        max_iterations = int(os.environ.get("CLI_MAX_ITERATIONS",100))
        if iteration > max_iterations:
            print(f"WARNING: Reached maximum iterations ({max_iterations}), stopping")
            await engine.stop_workflow(workflow.workflow_id)
            break


    
    print(f"\nEvolution complete!")
    if workflow.best_kpi is not None:
        print(f"Final best KPI: {workflow.best_kpi:.6f}")
    print(f"Results saved to: {OUTPUT_FOLDER}/")

    return all_results, all_result_file_paths


def get_baseline_kpi(workflow):
    """Get baseline KPI for comparison"""
    if (hasattr(workflow, 'baseline_code') and 
        workflow.baseline_code and 
        workflow.baseline_code.result):
        return extract_kpi_from_result(workflow.baseline_code.result)
    return None


def get_system_info(python_path: str) -> SystemInfo:
    return SystemInfo(
        python_libraries=_get_python_libraries(python_path),
        python_version=_get_python_version(python_path),
        os=sys.platform
    )


def _get_python_libraries(python_path: str) -> list[str]:
    try:
        installed_libraries = subprocess.check_output(
            [python_path, "-m", "pip", "freeze"],
            universal_newlines=True
        ).strip()
        return [lib.strip() for lib in installed_libraries.split("\n")]
    except subprocess.CalledProcessError:
        return []


def _get_python_version(python_path: str) -> str:
    try:
        return subprocess.check_output(
            [python_path, "--version"],
            universal_newlines=True
        ).strip()
    except subprocess.CalledProcessError:
        return "Unknown"


def extract_kpi_from_result(result: CodeResult) -> float:
    """Extract KPI value from code execution result"""
    if result.return_code != 0:
        return None
    
    # Look for KPI in stdout
    if result.stdout:
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line.startswith('KPI:'):
                try:
                    return float(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    continue
    return None


def save_code_version_to_file(code_version: CodeVersion, result: CodeResult, output_path: str):
    """Save code version and results to files"""
    
    timestamp = code_version.timestamp.strftime("%Y_%m_%d__%H_%M_%S")

    folder_name = _make_filesystem_safe(f"{code_version.code_version_id}")
    
    python_file_name = _make_filesystem_safe(f"{code_version.code_version_id}.py")
    info_file_name = _make_filesystem_safe(f"{code_version.code_version_id}_info.json")
    result_file_name = _make_filesystem_safe(f"{code_version.code_version_id}_result.json")
    
    folder_path = Path(output_path) / OUTPUT_FOLDER / "lineage" / folder_name
    python_file_path = folder_path / python_file_name
    info_file_path = folder_path / info_file_name
    result_file_path = folder_path / result_file_name

    folder_path.mkdir(parents=True, exist_ok=True)
    
    if not python_file_path.exists():
        python_file_path.write_text(code_version.code["code"])
    if not info_file_path.exists():
        info_file_path.write_text(json.dumps(code_version.info, indent=4) + "\n")
    if not result_file_path.exists():
        result_file_path.write_text(json.dumps(result.model_dump(), indent=4) + "\n")
    
    return python_file_path

def _make_filesystem_safe(name):
    return re.sub(r"[/\\?%*:|\"<>\x7F\x00-\x1F]", '_', name)


if __name__ == "__main__":
    main()