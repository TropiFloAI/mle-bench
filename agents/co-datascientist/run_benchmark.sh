#!/bin/bash
# Unified benchmark runner for co-datascientist MLE-bench agent
# Run from anywhere - script will find mle-bench root

set -e  # Exit on any error

# Color output for better visibility
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Determine script location and navigate to mle-bench root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLEBENCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$MLEBENCH_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Co-DataScientist MLE-Bench Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Running from: $MLEBENCH_ROOT${NC}"
echo ""

# Default values
COMPETITION_SET="${1:-experiments/splits/trial.txt}"
N_WORKERS="${2:-1}"
N_SEEDS="${3:-1}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Competition set: $COMPETITION_SET"
echo "  Workers: $N_WORKERS"
echo "  Seeds per task: $N_SEEDS"
echo ""

# Step 1: Build base mlebench-env if needed
if [[ "$(docker images -q mlebench-env 2> /dev/null)" == "" ]]; then
    echo -e "${YELLOW}Step 1/4: Building mlebench-env base image...${NC}"
    docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .
else
    echo -e "${GREEN}Step 1/4: mlebench-env base image already exists ✓${NC}"
fi

# Step 2: Build co-datascientist agent image
echo -e "${YELLOW}Step 2/4: Building co-datascientist agent image...${NC}"
# Build from Co-DataScientist_ root to access both engine and mle-bench
cd "$MLEBENCH_ROOT/../"
docker build --platform=linux/amd64 \
    -t co-datascientist \
    -f mle-bench/agents/co-datascientist/Dockerfile \
    .
cd "$MLEBENCH_ROOT"

echo -e "${GREEN}Agent image built successfully ✓${NC}"

# Clean up dangling images
echo -e "${YELLOW}Cleaning up dangling images...${NC}"
docker image prune -f > /dev/null 2>&1 || true

# Step 3: Activate venv and run benchmarks
echo -e "${YELLOW}Step 3/4: Running benchmarks...${NC}"
source venv/bin/activate

python run_agent.py \
    --competition-set "$COMPETITION_SET" \
    --agent-id co-datascientist \
    --n-workers "$N_WORKERS" \
    --n-seeds "$N_SEEDS"

echo -e "${GREEN}Benchmark run complete ✓${NC}"
echo ""

# Step 4: Grade results
echo -e "${YELLOW}Step 4/4: Grading results...${NC}"
RUN_DIR=$(ls -td runs/*/ | head -1)
echo -e "${BLUE}Results directory: $RUN_DIR${NC}"

python experiments/make_submission.py \
    --metadata "$RUN_DIR/metadata.json" \
    --output "$RUN_DIR/submission.jsonl"

mlebench grade \
    --submission "$RUN_DIR/submission.jsonl" \
    --output-dir "$RUN_DIR"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Benchmarking complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Results saved to: ${BLUE}$RUN_DIR${NC}"
echo -e "Grading report: ${BLUE}$RUN_DIR/*_grading_report.json${NC}"
echo ""

