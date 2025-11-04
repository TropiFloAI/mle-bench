# Co-DataScientist Agent for MLE-Bench

Production-ready agent with real-time KPI tracking and automatic grading.

## Quick Start

### 1. Setup Environment

```bash
cd /Users/ozkilim/Documents/mle-bench

# Activate virtual environment
source venv/bin/activate

# Ensure data is symlinked (don't move!)
# If data is in ~/Downloads/competitions/:
ln -s ~/Downloads/competitions ~/.mlebench/competitions

# Prepare a competition (example)
mlebench prepare -c random-acts-of-pizza
```

### 2. Build Docker Image

```bash
# Build base environment (one time, ~5-10 minutes)
docker build --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .

# Build co-datascientist agent
cd /Users/ozkilim/Documents
docker build --platform=linux/amd64 -t co-datascientist -f mle-bench/agents/co-datascientist/Dockerfile .
```

### 3. Run with Automatic Grading

```bash
cd /Users/ozkilim/Documents/mle-bench
source venv/bin/activate

# Create competition list
echo "random-acts-of-pizza" > /tmp/competitions.txt

# Start agent run
python run_agent.py \
  --agent-id co-datascientist \
  --competition-set /tmp/competitions.txt \
  --n-workers 1 \
  --n-seeds 1 \
  > agent_run.log 2>&1 &

# Wait ~20 seconds for run to start, then get run directory
sleep 20
LATEST=$(ls -td runs/*co-datascientist*/*/ 2>/dev/null | head -1)
echo "Run directory: $LATEST"

# Start auto-grader watcher for real-time grading
python agents/co-datascientist/utils/auto_grader_watcher.py "$LATEST" &
echo "Auto-grader started - will grade each iteration automatically"
```

### 4. Monitor Progress

```bash
# Watch tracking data update in real-time
watch -n 10 "cat $LATEST/submission/iteration_tracking.json | python3 -m json.tool | head -50"

# Or check the plot periodically (no auto-popup)
open "$LATEST/submission/kpi_progression.png"

# Check agent logs
tail -f $LATEST/run.log
```

## What You Get

### Real-Time Tracking Plot

Located at: `runs/<run-group>/<competition>/submission/kpi_progression.png`

The plot shows:
- **Blue line (•)**: Validation KPI for each iteration
- **Orange line (■)**: Test KPI for each graded iteration
- **Gold/Silver/Bronze lines**: Medal thresholds
- **X-axis**: Iteration number + timestamp (e.g., "#1\n14:50:52")
- **Yellow annotation**: Best validation KPI

### Tracking Data

Located at: `runs/<run-group>/<competition>/submission/iteration_tracking.json`

Contains:
```json
{
  "competition_id": "random-acts-of-pizza",
  "started_at": "2025-11-04T14:50:49.828701",
  "iterations": [
    {
      "iteration": 1,
      "timestamp": "2025-11-04T14:50:52.025157",
      "val_kpi": 0.7304,
      "test_kpi": 0.70502,  ← Automatically graded
      "test_score_details": {
        "bronze_medal": true  ← Medal achieved!
      },
      "code_version_id": "..."
    }
  ],
  "medal_thresholds": {
    "gold": 0.97908,
    "silver": 0.76482,
    "bronze": 0.6921
  }
}
```

### Output Files

All files saved to: `runs/<run-group>/<competition>/submission/`

- `iteration_tracking.json` - Complete tracking data
- `kpi_progression.png` - Live-updating plot
- `submission_iterN.csv` - Submission for iteration N
- `submission.csv` - Current best submission
- `performance_summary.json` - Final results

## How It Works

### Architecture

```
┌─────────────────────────────────────┐
│     CONTAINER (Agent)               │
│                                     │
│  Each iteration:                    │
│   1. Generate code                  │
│   2. Evaluate (get val_kpi)        │
│   3. Save submission_iterN.csv     │
│   4. Update tracking JSON          │
│   5. Create .grade_trigger_iterN   │◄─── Trigger file
└─────────────────────────────────────┘
              ↓
    (Host detects trigger)
              ↓
┌─────────────────────────────────────┐
│     HOST (Auto-Grader)              │
│                                     │
│  Every 5 seconds:                   │
│   1. Check for trigger files        │
│   2. Grade submission_iterN.csv     │
│   3. Update tracking JSON           │
│   4. Generate complete plot         │◄─── Plot with val + test
│   5. Delete trigger file            │
└─────────────────────────────────────┘
```

### Key Features

1. **Automatic Grading**: Grading happens automatically after each iteration via trigger files
2. **Smart Data Preservation**: Container reloads JSON before updating to preserve grading results
3. **Single Plot Source**: Only grading script generates plots (simpler, more elegant)
4. **Real-Time Updates**: All files written directly to host's runs directory
5. **No Popups**: Plots saved silently using matplotlib Agg backend

## Agent Configuration

### Environment Variables

Set in `container.env`:
```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Config Parameters

Edit `config.yaml`:
```yaml
name: co-datascientist
version: 1.0.0
privileged: true  # Required for Docker-in-Docker
timeout: 14400    # 4 hours
env_vars:
  OPENAI_API_KEY: ${OPENAI_API_KEY}
  OPENAI_MODEL: ${OPENAI_MODEL}
  OPENAI_BASE_URL: ${OPENAI_BASE_URL}
```

## Advanced Usage

### Multiple Competitions

```bash
# Create competition list
cat > competitions.txt << EOF
random-acts-of-pizza
spaceship-titanic
house-prices-advanced-regression-techniques
EOF

# Run on all competitions
python run_agent.py \
  --agent-id co-datascientist \
  --competition-set competitions.txt \
  --n-workers 1 \
  --n-seeds 1
```

### Custom Iterations

Edit `adapter/cli_mle.py`:
```python
max_evolution_iterations = 5  # Default, change as needed
```

### Manual Grading

If auto-grader isn't running:
```bash
LATEST=$(ls -td runs/*co-datascientist*/*/ | head -1)
python agents/co-datascientist/utils/grade_submissions.py "$LATEST"
```

### Stop Everything

```bash
# Stop containers
docker stop $(docker ps -q)

# Stop agent processes
pkill -f "run_agent.py"

# Stop auto-grader
pkill -f "auto_grader_watcher.py"
```

## Troubleshooting

### Auto-grading not working?

1. **Check watcher is running**:
   ```bash
   ps aux | grep auto_grader_watcher
   ```

2. **Check for trigger files** (should be deleted quickly):
   ```bash
   ls -la $LATEST/submission/.grade_trigger*
   ```

3. **Check container created triggers**:
   ```bash
   grep "trigger" $LATEST/run.log
   ```

4. **Manually grade if needed**:
   ```bash
   python agents/co-datascientist/utils/grade_submissions.py "$LATEST"
   ```

### Docker issues?

```bash
# Clean up old containers and images
docker stop $(docker ps -aq)
docker rm $(docker ps -aq)
docker system prune -af

# Rebuild from scratch
docker build --no-cache --platform=linux/amd64 -t mlebench-env -f environment/Dockerfile .
cd /Users/ozkilim/Documents
docker build --no-cache --platform=linux/amd64 -t co-datascientist -f mle-bench/agents/co-datascientist/Dockerfile .
```

### Data not found?

```bash
# Check symlink
ls -la ~/.mlebench/competitions

# Re-symlink if needed
ln -sf ~/Downloads/competitions ~/.mlebench/competitions

# Prepare competition
mlebench prepare -c <competition-id>
```

## File Structure

```
agents/co-datascientist/
├── README.md                 # This file
├── Dockerfile                # Agent Docker image
├── config.yaml               # Agent configuration
├── container.env             # Environment variables
├── start.sh                  # Container entrypoint
├── adapter/
│   ├── main.py              # Main orchestration
│   ├── cli_mle.py           # Evolution loop
│   ├── handle_results.py    # Result processing & ensembling
│   └── iteration_tracker.py # Tracking data management
├── baselines/
│   └── ...                  # Baseline models for each competition
└── utils/
    ├── grade_submissions.py      # Grading script (generates plot)
    └── auto_grader_watcher.py    # Watches for triggers, runs grading
```

## Dependencies

- **Python 3.12+**
- **Docker** with platform=linux/amd64
- **mlebench** (installed in venv)
- **OpenAI API key** (or compatible endpoint)

Python packages (in container):
- smolagents
- openai, litellm
- pandas, numpy, scikit-learn
- matplotlib (for plotting)
- torch, torchvision

## Performance Tips

1. **Use overnight runs**: Agent takes 2-4 hours per competition
2. **Monitor GPU/CPU**: Container uses significant resources
3. **Check disk space**: Submissions and logs can grow large
4. **Set realistic timeouts**: Default 4h, adjust in config.yaml
5. **Use auto-grader**: No need to manually check results

## Results

After run completes, check:

```bash
# View final summary
cat $LATEST/submission/performance_summary.json

# View all tracked iterations
cat $LATEST/submission/iteration_tracking.json | python3 -m json.tool

# View final plot
open $LATEST/submission/kpi_progression.png
```

## Known Issues & Limitations

- **Requires prepared dataset**: Must run `mlebench prepare` first
- **Docker-in-Docker**: Needs privileged mode
- **OpenAI API**: Requires valid API key and sufficient credits
- **Platform specific**: Built for linux/amd64 (runs via Rosetta on M1/M2 Macs)

## Citation

If you use this agent, please cite:

```
@article{chan2024mlebench,
  title={MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering},
  author={Chan, Jun Shern and Pister, Neil and Wang, Jinu and Ostrow, Evan and Treutlein, Johannes and Bale, Michael and Li, Belinda and Huang, Stephanie and Phung, Jennifer and Scheurer, Jérémy and others},
  journal={arXiv preprint arXiv:TODO},
  year={2024}
}
```

## Support

For issues or questions:
- Check the [main MLE-bench README](../../README.md)
- Review logs in `runs/<run-group>/<competition>/run.log`
- Inspect tracking data in `submission/iteration_tracking.json`

---

**Status**: ✅ Production Ready  
**Last Updated**: November 2025  
**Version**: 2.0 (with auto-grading)
