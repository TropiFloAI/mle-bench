#!/bin/bash
set -x # Print commands and their arguments as they are executed

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate agent

# Run the adapter's main entry point
# The co-datascientist-engine is now installed as a package!
python ${AGENT_DIR}/adapter/main.py


# Check that the agent doesn't have permissions to read private dir
ls /private
# ls: cannot open directory '/private': Permission denied

# Check that the agent does have permissions to read/write everything in /home
ls /home/data
touch $CODE_DIR/code.py
touch $LOGS_DIR/run.log
touch $AGENT_DIR/agent_was_here.txt
cat /home/instructions.txt

# Use the environment-provided grading server to validate our submission
bash /home/validate_submission.sh /home/submission/submission.csv
