#!/usr/bin/env python3
"""
Log ingestion module for parsing JSONL robot run logs.
Extracts failures and relevant context.
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RunFailure:
    """Represents a failed robot run with context."""
    run_id: str
    error_type: str
    error_message: str
    failure_timestep: int
    total_timesteps: int
    action_at_failure: Dict[str, Any]
    observation_at_failure: Dict[str, Any]
    preceding_actions: List[Dict[str, Any]]
    error_metadata: Dict[str, Any]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_id": self.run_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "failure_timestep": self.failure_timestep,
            "total_timesteps": self.total_timesteps,
            "action_at_failure": self.action_at_failure,
            "observation_at_failure": self.observation_at_failure,
            "preceding_actions": self.preceding_actions,
            "error_metadata": self.error_metadata,
            "timestamp": self.timestamp
        }

    def get_signature(self) -> str:
        """Get a signature for this failure (error type + action type)."""
        action_type = self.action_at_failure.get("type", "unknown")
        return f"{self.error_type}::{action_type}"


class LogIngestor:
    """Ingests and parses robot run logs from JSONL format."""

    def __init__(self, context_window: int = 5):
        """
        Initialize log ingestor.

        Args:
            context_window: Number of preceding actions to include in failure context
        """
        self.context_window = context_window
        self.failures: List[RunFailure] = []
        self.runs_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def load_logs(self, filepath: str) -> None:
        """Load JSONL logs from file."""
        print(f"Loading logs from {filepath}...")

        log_count = 0
        with jsonlines.open(filepath) as reader:
            for log_entry in reader:
                run_id = log_entry.get("run_id")
                self.runs_by_id[run_id].append(log_entry)
                log_count += 1

        print(f"✓ Loaded {log_count} log entries from {len(self.runs_by_id)} runs")

    def extract_failures(self) -> List[RunFailure]:
        """Extract all failures from loaded logs."""
        print("Extracting failures from runs...")

        failures = []

        for run_id, logs in self.runs_by_id.items():
            # Sort by timestep
            logs = sorted(logs, key=lambda x: x.get("timestep", 0))

            # Find failure point
            for idx, log_entry in enumerate(logs):
                if log_entry.get("status") == "failed" and "error" in log_entry:
                    error_info = log_entry["error"]

                    # Get preceding actions for context
                    start_idx = max(0, idx - self.context_window)
                    preceding_actions = [
                        logs[i]["action"] for i in range(start_idx, idx)
                    ]

                    failure = RunFailure(
                        run_id=run_id,
                        error_type=error_info.get("type", "unknown"),
                        error_message=error_info.get("message", ""),
                        failure_timestep=log_entry["timestep"],
                        total_timesteps=len(logs),
                        action_at_failure=log_entry["action"],
                        observation_at_failure=log_entry["observation"],
                        preceding_actions=preceding_actions,
                        error_metadata={
                            "severity": error_info.get("severity"),
                            "recoverable": error_info.get("recoverable")
                        },
                        timestamp=log_entry.get("timestamp", "")
                    )

                    failures.append(failure)
                    break  # Only one failure per run

        self.failures = failures
        print(f"✓ Extracted {len(failures)} failures")

        # Print failure statistics
        error_counts = defaultdict(int)
        for failure in failures:
            error_counts[failure.error_type] += 1

        print("\nFailure breakdown:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {error_type}: {count}")

        return failures

    def get_run_logs(self, run_id: str) -> List[Dict[str, Any]]:
        """Get all logs for a specific run."""
        return self.runs_by_id.get(run_id, [])

    def save_failures(self, output_file: str) -> None:
        """Save extracted failures to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        failures_data = [f.to_dict() for f in self.failures]

        with open(output_path, 'w') as f:
            json.dump(failures_data, f, indent=2)

        print(f"✓ Saved {len(failures_data)} failures to {output_path}")

    def filter_by_error_type(self, error_type: str) -> List[RunFailure]:
        """Filter failures by error type."""
        return [f for f in self.failures if f.error_type == error_type]

    def filter_by_action_type(self, action_type: str) -> List[RunFailure]:
        """Filter failures by action type at failure."""
        return [
            f for f in self.failures
            if f.action_at_failure.get("type") == action_type
        ]


def main():
    """Example usage of LogIngestor."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest robot run logs")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output", default="output/failures.json",
                       help="Output JSON file for failures")
    parser.add_argument("--context-window", type=int, default=5,
                       help="Number of preceding actions to include")

    args = parser.parse_args()

    ingestor = LogIngestor(context_window=args.context_window)
    ingestor.load_logs(args.input)
    ingestor.extract_failures()
    ingestor.save_failures(args.output)


if __name__ == "__main__":
    main()
