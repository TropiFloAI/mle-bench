# Task Description Loading

## Overview

The co-datascientist agent automatically loads task-specific competition descriptions for every MLE-bench task. This provides the agent with full context about:
- The problem to solve
- Dataset details
- Evaluation metrics
- Competition rules

## How It Works

### 1. Competition Structure

Each MLE-bench competition has its own description file:
```
mlebench/competitions/
├── random-acts-of-pizza/
│   ├── description.md          # Task-specific description
│   ├── config.yaml
│   └── ...
├── spaceship-titanic/
│   ├── description.md          # Different task description
│   ├── config.yaml
│   └── ...
└── ...
```

### 2. Data Preparation

When you run `mlebench prepare -c <competition_id>`, the description is copied:
```bash
# From source
mlebench/competitions/<competition_id>/description.md

# To prepared directory
~/.cache/mle-bench/data/<competition_id>/prepared/public/description.md
```

### 3. Docker Volume Mounting

When a benchmark run starts, MLE-bench mounts the **competition-specific** public directory:

```python
# In agents/run.py
volumes_config = {
    competition.public_dir.resolve().as_posix(): {
        "bind": "/home/data",  # Mounted here
        "mode": "ro",
    },
    ...
}
```

**Key Point**: Each Docker container gets a DIFFERENT `/home/data` mount depending on which competition is running!

### 4. Agent Loading

The agent's `cli_mle.py` loads the description at startup:

```python
def load_task_description():
    """
    Loads /home/data/description.md which is the task-specific
    description for whichever competition is currently running.
    """
    data_dir = os.environ.get("DATA_DIR", "/home/data")
    competition_id = os.environ.get("COMPETITION_ID", "unknown")
    description_path = Path(data_dir) / "description.md"
    
    with open(description_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return content
```

### 5. Context Injection

The loaded description is injected into the workflow:

```python
# After starting the workflow
task_description = load_task_description()
if task_description:
    workflow.user_context_summary = task_description
    print(f"✓ Task description injected into workflow context")
```

## Verification

### Log Output

When a run starts, you'll see:
```
✓ Loaded task description for 'random-acts-of-pizza' from /home/data/description.md (7157 chars)
✓ Task description injected into workflow context
Workflow started: <workflow_id>
```

### Hypothesis Generation

The agent's generated hypotheses will reflect the task context:

**For random-acts-of-pizza:**
- "Including a feature capturing the length of the Reddit username..."
- "Adding text length for pizza request descriptions..."
- "Day of the week patterns for successful requests..."

**For a different task (e.g., image classification):**
- "Adding data augmentation with random rotations..."
- "Using pre-trained ResNet features..."
- "Ensemble of different CNN architectures..."

## Multi-Competition Support

The system automatically handles any competition:

| Competition Run | Mounted Directory | Description Loaded |
|----------------|-------------------|-------------------|
| random-acts-of-pizza | `~/.cache/mle-bench/data/random-acts-of-pizza/prepared/public/` | Pizza competition description |
| spaceship-titanic | `~/.cache/mle-bench/data/spaceship-titanic/prepared/public/` | Spaceship competition description |
| dog-breed-identification | `~/.cache/mle-bench/data/dog-breed-identification/prepared/public/` | Dog breed competition description |
| custom-task | `~/.cache/mle-bench/data/custom-task/prepared/public/` | Custom task description |

**No configuration needed** - it's automatic!

## Environment Variables

The agent uses these environment variables (automatically set by MLE-bench):

- `DATA_DIR`: Path to mounted data directory (default: `/home/data`)
- `COMPETITION_ID`: Current competition identifier (for logging)

## Testing

To verify task description loading for any competition:

```bash
cd /home/ozkilim/Co-DataScientist_/mle-bench

# 1. Set iterations to 1 for quick test
vim agents/co-datascientist/config.yaml  # Set cli_max_iterations: 1

# 2. Run benchmark on any competition
./agents/co-datascientist/run_benchmark.sh experiments/splits/test-<competition>.txt 1 1

# 3. Check the run log
tail -f runs/<latest-run>/<competition>_<uuid>/run.log | grep "Loaded task description"
```

You should see:
```
✓ Loaded task description for '<competition-id>' from /home/data/description.md (XXXX chars)
```

## Troubleshooting

### Description Not Found

If you see:
```
⚠ Task description not found at /home/data/description.md for competition '<competition-id>'
```

**Solution**: Prepare the dataset first:
```bash
cd /home/ozkilim/Co-DataScientist_/mle-bench
mlebench prepare -c <competition-id>
```

### Wrong Competition Description

If the loaded description doesn't match the competition:
1. Check `COMPETITION_ID` in the logs
2. Verify the volume mount in Docker container
3. Re-prepare the dataset: `mlebench prepare -c <competition-id>`

## Summary

✅ **Task-specific**: Each competition gets its own description  
✅ **Automatic**: No manual configuration needed  
✅ **Docker-safe**: Works through volume mounts  
✅ **Verified**: Tested and working in production runs  
✅ **Multi-competition**: Handles any MLE-bench competition  

The agent always has the correct task context! 🎯

