# Co-DataScientist MLE-Bench Agent

This directory contains the **complete, self-contained benchmarking setup** for running the co-datascientist engine on [OpenAI's MLE-Bench testing suite](https://github.com/openai/mle-bench/).

**Key Design**: The co-datascientist-engine is installed as a Python package (like a library), and this directory contains only the MLE-bench adapter/orchestration code. This keeps the engine code unchanged while providing a clean, consolidated benchmarking solution.

---

## 🏗️ Architecture

```
mle-bench/agents/co-datascientist/
├── adapter/                    # Orchestration layer
│   ├── main.py                # Entry point for benchmark runs
│   ├── cli_mle.py             # CLI wrapper for the engine
│   └── handle_results.py      # Result processing & ensembling
├── baselines/                 # Competition-specific baseline scripts
│   ├── spaceship-titanic.py
│   ├── random-acts-of-pizza.py
│   └── ...
├── Dockerfile                 # Agent Docker image (extends mlebench-env)
├── pyproject.toml             # Package dependencies (installs engine!)
├── config.yaml                # Runtime configuration
├── start.sh                   # Container entry point
├── run_benchmark.sh           # 🚀 Main runner script
├── container.env.example      # Environment variables template
└── README.md                  # This file
```

### How It Works

1. **Base Image**: `mlebench-env` provides the MLE-bench infrastructure
2. **Agent Image**: `co-datascientist` extends base, installs engine as package
3. **Orchestration**: `run_agent.py` (from MLE-bench) spawns containers
4. **Execution**: Containers run `start.sh` → `adapter/main.py` → engine workflow
5. **Results**: Submissions saved to `mle-bench/runs/`, graded automatically

---

## 📋 Prerequisites

- **Docker**: For containerized execution
- **Python 3.12**: With `uv` or `pip` for package management
- **Git LFS**: For MLE-bench competition data
- **API Keys**: Azure OpenAI API key for the engine

---

## 🚀 Quick Start

### 1. Clone and Setup MLE-Bench

```bash
# Clone this repository (if not already done)
cd /path/to/mle-bench

# Install MLE-bench in a virtual environment
python3 -m venv venv
source venv/bin/activate

# Pull competition data
git lfs fetch --all
git lfs pull

# Install MLE-bench
pip install -e .
```

### 2. Prepare Competition Data

```bash
# Prepare specific competitions
mlebench prepare -c spaceship-titanic
mlebench prepare -c random-acts-of-pizza

# Or prepare the full Lite dataset
mlebench prepare --lite
```

### 3. Configure API Keys

```bash
cd agents/co-datascientist

# Copy the example and add your API key
cp container.env.example container.env

# Edit container.env and add:
# AZURE_OPENAI_API_KEY=your_key_here
```

### 4. Build Docker Images

```bash
# From mle-bench root
cd /path/to/mle-bench

# Build base MLE-bench environment (first time only, ~10-15 min)
docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .

# Build co-datascientist agent image (~2-5 min)
docker build --platform=linux/amd64 \
    -t co-datascientist \
    -f agents/co-datascientist/Dockerfile \
    agents/co-datascientist
```

### 5. Run Benchmarks! 🎯

```bash
# Option A: Use the unified runner script (RECOMMENDED)
./agents/co-datascientist/run_benchmark.sh

# Option B: Manual execution with more control
source venv/bin/activate
python run_agent.py \
    --competition-set experiments/splits/trial.txt \
    --agent-id co-datascientist \
    --n-workers 1 \
    --n-seeds 1
```

---

## 📖 Detailed Usage

### Running Specific Competitions

Create a text file with competition IDs (one per line):

```bash
# experiments/splits/my-competitions.txt
spaceship-titanic
random-acts-of-pizza
```

Run with the custom competition set:

```bash
./agents/co-datascientist/run_benchmark.sh experiments/splits/my-competitions.txt 2 1
#                                          ^competition file  ^workers ^seeds
```

### Runner Script Arguments

```bash
./run_benchmark.sh [COMPETITION_SET] [N_WORKERS] [N_SEEDS]
```

- `COMPETITION_SET`: Path to txt file with competition IDs (default: `experiments/splits/trial.txt`)
- `N_WORKERS`: Number of parallel Docker containers (default: 1)
- `N_SEEDS`: Number of times to run each task (default: 1)

**Examples:**

```bash
# Run trial competitions with 1 worker
./run_benchmark.sh

# Run custom set with 4 parallel workers, 3 trials each
./run_benchmark.sh experiments/splits/custom.txt 4 3

# Run lite set with 10 workers
./run_benchmark.sh experiments/splits/lite.txt 10 1
```

### Manual Execution

For more control, run directly with `run_agent.py`:

```bash
source venv/bin/activate

python run_agent.py \
    --competition-set experiments/splits/my-comps.txt \
    --agent-id co-datascientist \
    --n-workers 4 \
    --n-seeds 1 \
    --retain  # Optional: keep containers for debugging
```

### Grading Results

Results are automatically graded by the runner script. To manually grade:

```bash
# Find the most recent run
RUN_DIR=$(ls -td runs/*/ | head -1)

# Create submission file
python experiments/make_submission.py \
    --metadata $RUN_DIR/metadata.json \
    --output $RUN_DIR/submission.jsonl

# Grade it
mlebench grade \
    --submission $RUN_DIR/submission.jsonl \
    --output-dir $RUN_DIR

# View results
cat $RUN_DIR/*_grading_report.json
```

---

## ⚙️ Configuration

### Runtime Configuration (`config.yaml`)

Adjust these values to control agent behavior:

```yaml
vars:
  script_execution_timeout: 7200      # Max time for baseline execution (2 hours)
  cli_timeout: 43200                  # Max time for entire workflow (12 hours)
  cli_max_iterations: 2               # Number of hypothesis evolution cycles
  batch_size: 5                       # Hypotheses generated per iteration
  ensemble_n: 3                       # Number of top models to ensemble
```

### Environment Variables (`container.env`)

Required variables:

```bash
AZURE_OPENAI_API_KEY=your_api_key_here
```

Optional (set via config.yaml instead):

```bash
SCRIPT_EXECUTION_TIMEOUT=7200
CLI_TIMEOUT=43200
CLI_MAX_ITERATIONS=2
BATCH_SIZE=5
ENSEMBLE_N=3
```

### Package Dependencies (`pyproject.toml`)

The engine is installed as a package! Two modes:

**Production (default):** Install from GitHub
```toml
[tool.uv.sources]
co-datascientist-engine = { git = "https://github.com/TropiFloAI/co-datascientist-engine.git", branch = "master" }
```

**Local Development:** Use editable install
```toml
[tool.uv.sources]
co-datascientist-engine = { path = "../../../co-datascientist-engine", editable = true }
```

---

## 🔧 Development Workflow

### Making Code Changes

**Engine Changes:**
1. Edit code in `/path/to/co-datascientist-engine/`
2. Changes are automatically used if using local editable install
3. Rebuild Docker image: `docker build -t co-datascientist -f agents/co-datascientist/Dockerfile agents/co-datascientist`

**Adapter Changes:**
1. Edit code in `agents/co-datascientist/adapter/`
2. Rebuild Docker image (same command as above)

**Baseline Changes:**
1. Edit/add files in `agents/co-datascientist/baselines/`
2. Rebuild Docker image

### Using Local Engine During Development

Edit `pyproject.toml`:

```toml
[tool.uv.sources]
# Comment out the GitHub line
# co-datascientist-engine = { git = "https://...", branch = "master" }

# Uncomment the local path
co-datascientist-engine = { path = "../../../co-datascientist-engine", editable = true }
```

Then rebuild the Docker image.

### Adding New Baselines

Create a new baseline script in `baselines/`:

```bash
# baselines/my-competition.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# ... your baseline code ...
```

The script filename must match the competition ID: `{competition-id}.py`

Rebuild the Docker image after adding baselines.

---

## 🐛 Troubleshooting

### Docker Build Fails

**Issue**: Cannot find co-datascientist-engine package

**Solution**: Check `pyproject.toml` - ensure Git URL is correct or local path exists

```bash
# Test package installation locally
cd agents/co-datascientist
pip install -e .
```

### Container Crashes Immediately

**Issue**: Missing API key

**Solution**: Ensure `container.env` exists and has valid `AZURE_OPENAI_API_KEY`

### Competition Not Found

**Issue**: `mlebench prepare` didn't download data

**Solution**: Prepare the competition data first:

```bash
mlebench prepare -c competition-id
```

### Out of Memory

**Issue**: Too many workers or large models

**Solution**: Reduce `--n-workers` or increase Docker memory limits

```bash
# Check Docker memory
docker info | grep -i memory

# Run with fewer workers
./run_benchmark.sh experiments/splits/lite.txt 1 1
```

### Image Rebuilds Too Slow

**Issue**: Every rebuild takes 5+ minutes

**Solution**: Use layer caching - only copy what changed:

```dockerfile
# Copy package definition first (rarely changes)
COPY pyproject.toml ${AGENT_DIR}/pyproject.toml
RUN pip install -e ${AGENT_DIR}

# Copy code last (changes frequently)
COPY adapter/ ${AGENT_DIR}/adapter/
```

---

## 📁 Results Structure

After running benchmarks, results are saved in `mle-bench/runs/`:

```
runs/
└── 2025-10-31T13-19-06-GMT_run-group_co-datascientist/
    ├── metadata.json              # Run metadata
    ├── submission.jsonl           # Submission file
    ├── *_grading_report.json      # Grading results ⭐
    └── competition-id_uuid/       # Per-competition results
        ├── run.log                # Execution log
        ├── submission/
        │   └── submission.csv     # Competition submission
        ├── code/                  # Generated code versions
        └── logs/                  # Agent logs
```

Key files:
- **Grading Report**: `*_grading_report.json` - Final scores and metrics
- **Submission CSV**: `submission/submission.csv` - Model predictions
- **Run Log**: `run.log` - Full execution trace
- **Code Versions**: `code/*.py` - All generated hypothesis code

---

## 🧪 Testing & Validation

### Quick Smoke Test

```bash
# Run a single fast competition
./agents/co-datascientist/run_benchmark.sh experiments/splits/trial.txt 1 1
```

### Validate Docker Setup

```bash
# Check images exist
docker images | grep -E "mlebench-env|co-datascientist"

# Test container can start
docker run --rm co-datascientist /bin/bash -c "echo 'Container works!'"
```

### Verify Package Installation

```bash
# Enter container interactively
docker run -it --rm co-datascientist /bin/bash

# Inside container:
conda activate agent
python -c "import co_datascientist_engine; print('Engine installed!')"
```

---

## 🤝 Contributing

### Adding New Competitions

1. Get the competition ID from MLE-bench
2. Create baseline in `baselines/{competition-id}.py`
3. Test locally:
   ```bash
   python baselines/competition-id.py
   ```
4. Add to a splits file (e.g., `experiments/splits/custom.txt`)
5. Rebuild image and run

### Improving Baselines

Edit existing baseline files and rebuild:

```bash
vim agents/co-datascientist/baselines/spaceship-titanic.py
docker build -t co-datascientist -f agents/co-datascientist/Dockerfile agents/co-datascientist
```

---

## 📚 Additional Resources

- **MLE-Bench Docs**: https://github.com/openai/mle-bench
- **Co-DataScientist Engine**: https://github.com/TropiFloAI/co-datascientist-engine
- **Competition Data**: Stored in `~/.cache/mle-bench/data/`
- **Results Analysis**: See `experiments/make_submission.py` for custom reports

---

## 🎯 Summary Cheat Sheet

```bash
# ONE-TIME SETUP
cd /path/to/mle-bench
python3 -m venv venv && source venv/bin/activate
pip install -e .
git lfs fetch --all && git lfs pull
mlebench prepare --lite
docker build -t mlebench-env -f environment/Dockerfile .

# CONFIGURE
cd agents/co-datascientist
cp container.env.example container.env
# Edit container.env with API key

# BUILD AGENT
cd /path/to/mle-bench
docker build -t co-datascientist -f agents/co-datascientist/Dockerfile agents/co-datascientist

# RUN
./agents/co-datascientist/run_benchmark.sh [competitions.txt] [workers] [seeds]

# VIEW RESULTS
cat runs/$(ls -t runs | head -1)/*_grading_report.json
```

---

## 💡 Design Rationale

### Why Package-Based Architecture?

**Before**: Split code across two repos with symlinks  
**After**: Engine as package, adapter as consumer

**Benefits:**
- ✅ Engine repo unchanged (single source of truth)
- ✅ No symlink management
- ✅ Clean separation of concerns
- ✅ Easy version control (pin engine versions)
- ✅ Consistent with backend architecture
- ✅ Standard Python packaging practices
- ✅ Everything runs from one place (mle-bench)

### Why Two Docker Images?

**`mlebench-env` (base):**
- Shared by all agents (AIDE, dummy, co-datascientist)
- Rarely changes
- Cached effectively

**`co-datascientist` (agent):**
- Extends base
- Installs engine + adapter
- Rebuilt frequently during development

This 2-layer approach:
- Faster rebuilds (only agent layer changes)
- Consistent with MLE-bench standards
- Better layer caching

---

**Questions? Issues?** Check the troubleshooting section or open an issue on GitHub.

**Happy Benchmarking! 🚀**

