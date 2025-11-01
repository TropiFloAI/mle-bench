"""
The orchestration script. This script gets a baseline model, and kicks off the agent.
"""

import getpass
import os
import sys
from pathlib import Path
import asyncio
import argparse
import importlib.util
import os
import logging
import shutil
import handle_results

from dotenv import load_dotenv
load_dotenv()

import cli_mle as cli

# Get the current user's username
username = getpass.getuser()

# Check if the current user ID is 0 (root user ID on Unix-like systems)
if os.getuid() == 0:
    print(f"You are running this script as root. Your username is '{username}'.")
else:
    print(f"You do not have root access. Your username is {username}.")

print("The script is being run with the following python interpreter:")
print(sys.executable)




cli.setup_logging()
logger = logging.getLogger(__name__)

# 1. Use the cli to start the process

args = argparse.Namespace(
    script_path = (Path(os.environ["BASELINE_DIR"]) / os.environ.get("COMPETITION_ID")).with_suffix(".py"),
    interpreter_path="python",  
    batch_size=int(os.environ["BATCH_SIZE"]),
    skip_preflight=True,       
    timeout=float(os.environ["CLI_TIMEOUT"]),
    verbosity=1                 
)   

# For comparing with benchmarks
if os.environ.get("RUN_ONLY_BASELINES","False") == "True":
    print("RUNNING BASELINES ONLY")
    results, result_file_paths = args.script_path
else:
    results, result_file_paths = asyncio.run(cli.run(args))

# TODO: Save summary of output, put in to code file?
handle_results.handle_results(results, result_file_paths)
