# piops-repro

CLI tool for incident triage. Helps ingest robot run logs, cluster similar failures, extract minimal reproducible examples, and evaluate policies.

## Features

- **Mock Data Generation**: Create synthetic robot run logs with configurable failure rates
- **Log Ingestion**: Parse JSONL logs and extract failure context
- **Failure Clustering**: Group similar failures using error signatures and semantic embeddings
- **Bundle Creation**: Package failures into reproducible examples for eval
- **Policy Evaluation**: Test stubbed policies against reproducible bundles

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the full pipeline with one command:

```bash
python cli.py pipeline --num-runs 100
```

This will:
1. Generate 100 mock robot runs with ~30% failure rate
2. Ingest logs and extract failures
3. Cluster similar failures using embeddings
4. Create reproducible bundles
5. Evaluate baseline vs improved policies

## CLI Commands

### Generate Mock Data

```bash
python cli.py generate --num-runs 100 --failure-prob 0.3 --output mockdata/robot_logs.jsonl
```

Options:
- `--num-runs`: Number of robot runs (default: 100)
- `--run-length`: Timesteps per run (default: 50)
- `--failure-prob`: Failure probability 0.0-1.0 (default: 0.3)
- `--seed`: Random seed for reproducibility (default: 42)

### Ingest Logs

```bash
python cli.py ingest mockdata/robot_logs.jsonl --output output/failures.json
```

Options:
- `--context-window`: Number of preceding actions to include (default: 5)

### Cluster Failures

```bash
python cli.py cluster output/failures.json --method embedding --output output/clusters.json
```

Options:
- `--method`: Clustering method: `signature`, `embedding`, or `both` (default: embedding)
- `--eps`: DBSCAN eps parameter (default: 0.5)
- `--min-samples`: DBSCAN min_samples (default: 2)

### Create Bundles

```bash
python cli.py bundle output/failures.json --clusters output/clusters.json --output-dir output/bundles
```

Options:
- `--clusters`: Optional clusters file (creates cluster-based bundles)
- `--minimize`: Minimize action sequences in bundles

### Evaluate Policies

```bash
python cli.py evaluate --bundles-dir output/bundles --compare
```

Options:
- `--compare`: Compare multiple policies (baseline vs improved)
- `--seed`: Random seed (default: 42)

## Project Structure

```
piops-repro/
├── cli.py              # Main CLI entry point
├── ingest.py           # Log parsing and failure extraction
├── cluster.py          # Failure clustering logic
├── bundle.py           # Reproducible bundle creation
├── eval_harness.py     # Policy evaluation framework
├── mockdata/
│   └── generate.py     # Mock data generator
├── output/             # Generated artifacts (created at runtime)
│   ├── failures.json
│   ├── clusters.json
│   ├── bundles/
│   └── eval_results.json
└── requirements.txt    # Python dependencies
```

## Module Usage

Each module can be used independently:

### Generate Mock Data

```python
from mockdata.generate import RobotLogGenerator

generator = RobotLogGenerator(seed=42)
generator.generate_dataset(
    num_runs=100,
    run_length=50,
    failure_prob=0.3,
    output_file="robot_logs.jsonl"
)
```

### Ingest Logs

```python
from ingest import LogIngestor

ingestor = LogIngestor(context_window=5)
ingestor.load_logs("robot_logs.jsonl")
failures = ingestor.extract_failures()
ingestor.save_failures("failures.json")
```

### Cluster Failures

```python
from cluster import FailureClusterer

clusterer = FailureClusterer()
clusterer.load_failures("failures.json")
clusters = clusterer.cluster_by_embedding(eps=0.5, min_samples=2)
clusterer.save_clusters("clusters.json")
```

### Create Bundles

```python
from bundle import BundleCreator

creator = BundleCreator(output_dir="bundles")
creator.create_bundles_from_clusters("clusters.json", "failures.json")
creator.create_manifest()
```

### Evaluate Policies

```python
from eval_harness import EvalHarness, StubPolicy, ImprovedPolicy

harness = EvalHarness(bundles_dir="bundles")
harness.load_bundles()

baseline = StubPolicy("baseline", success_rate=0.5)
improved = ImprovedPolicy("improved")

comparison = harness.compare_policies([baseline, improved])
harness.save_results("eval_results.json")
```

## Data Format

### JSONL Log Format

Each log entry contains:

```json
{
  "run_id": "run_00001",
  "timestamp": "2024-01-15T10:30:45.123456",
  "timestep": 0,
  "observation": {
    "camera": {
      "rgb_shape": [224, 224, 3],
      "detected_objects": ["cup", "plate"]
    },
    "robot_state": {
      "joint_positions": [0.1, -0.5, ...],
      "gripper_open": true,
      "end_effector_pose": {...}
    },
    "force_torque": [0.0, 0.0, ...]
  },
  "action": {
    "type": "grasp_object",
    "object": "cup",
    "force": 10.5
  },
  "status": "success",
  "error": {  // Only present on failure
    "type": "gripper_slip",
    "message": "Object cup slipped from gripper",
    "severity": "error",
    "recoverable": false
  }
}
```

### Bundle Format

Each bundle contains a minimal reproducible example:

```json
{
  "bundle_id": "bundle_cluster_0",
  "cluster_id": 0,
  "error_type": "gripper_slip",
  "error_message": "Object cup slipped from gripper",
  "initial_state": {...},
  "action_sequence": [
    {"type": "move_to_position", "target": [0.1, 0.2, 0.3]},
    {"type": "grasp_object", "object": "cup", "force": 10.5}
  ],
  "expected_outcome": "Action sequence completes successfully",
  "actual_outcome": "Failure at step 2: Object cup slipped from gripper",
  "difficulty": "medium",
  "source_failures": ["run_00042", "run_00087"]
}
```

## Example Workflow

```bash
# 1. Generate 200 robot runs
python cli.py generate --num-runs 200 --failure-prob 0.4

# 2. Ingest and extract failures
python cli.py ingest mockdata/robot_logs.jsonl

# 3. Cluster failures by embedding similarity
python cli.py cluster output/failures.json --method embedding --eps 0.4

# 4. Create minimal reproducible bundles
python cli.py bundle output/failures.json --clusters output/clusters.json

# 5. Evaluate policies
python cli.py evaluate --compare

# 6. View results
cat output/eval_results.json
```

## Development

### Running Tests

```bash
# Run individual modules
python ingest.py mockdata/robot_logs.jsonl
python cluster.py output/failures.json
python bundle.py --failures output/failures.json
python eval_harness.py --bundles-dir output/bundles --compare
```

### Extending

To add a new policy:

```python
from eval_harness import StubPolicy, EvalResult

class MyPolicy(StubPolicy):
    def execute(self, bundle):
        # Custom policy logic
        # Return EvalResult
        pass
```

## License

MIT
