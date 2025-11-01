#!/bin/bash
set -euo pipefail

# Set up the required files for a custom task
competition_id=$1


######### Set up mle-bench config files
# This is required to register the competition with the tool
mle_path=~/mle-bench/mlebench/competitions/$competition_id

mkdir -p $mle_path
# Data prep script
echo "from pathlib import Path
def prepare(raw: Path, public: Path, private: Path):
    pass" \
    > $mle_path/prepare.py

# Grading script
echo "import pandas as pd
def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    pass" \
    > $mle_path/grade.py

# Config
echo "id: custom-task
name: CUSTOM TASK NAME
competition_type: simple
awards_medals: false
prizes: null
description: $HOME/.cache/mle-bench/data/$competition_id/description.md

dataset:
  answers: $competition_id/prepared/private/test.csv
  sample_submission: $competition_id/prepared/public/sample_submission.csv

grader:
  name: accuracy # This can be changed
  grade_fn: mlebench.competitions.$competition_id.grade:grade

preparer: mlebench.competitions.$competition_id.prepare:prepare" \
    > $mle_path/config.yaml

######### Set up data files
# Place data files in default mle-bench data file location
data_path=~/.cache/mle-bench/data/$competition_id

mkdir -p $data_path/prepared/public
mkdir -p $data_path/prepared/private

touch $data_path/description.md

# Create the ground truth answers file.
# Can be empty if not grading
touch $data_path/prepared/private/test.csv

# Labelled training data
touch $data_path/prepared/public/train.csv

# Unlabelled test data
touch $data_path/prepared/public/test.csv

# Example submission file
touch $data_path/prepared/public/sample_submission.csv


######### Create baseline file
touch ~/co-datascientist-engine/mle-bench-agent/baselines/$competition_id.py