#!/usr/bin/env python3
"""
Bundle module for creating minimal reproducible examples from failures.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from ingest import RunFailure


@dataclass
class ReproBundle:
    """A minimal repro bundle."""
    bundle_id: str
    cluster_id: Optional[int]
    error_type: str
    error_message: str
    failure_run_id: str

    # Minimal repro scenario
    initial_state: Dict[str, Any]
    action_sequence: List[Dict[str, Any]]
    expected_outcome: str
    actual_outcome: str

    # Metadata
    created_at: str
    source_failures: List[str]
    difficulty: str  # "easy", "medium", "hard"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BundleCreator:
    """Creates minimal reproducible bundles from failure data."""

    def __init__(self, output_dir: str = "output/bundles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bundles: List[ReproBundle] = []

    def create_bundle_from_failure(self, failure: RunFailure,
                                   cluster_id: Optional[int] = None) -> ReproBundle:
        """
        Create a minimal reproducible bundle from a single failure.

        Extracts the minimal context needed to reproduce the failure.
        """
        # Extract initial state from observation at failure
        obs = failure.observation_at_failure
        initial_state = {
            "detected_objects": obs.get("camera", {}).get("detected_objects", []),
            "robot_state": obs.get("robot_state", {}),
            "timestep": failure.failure_timestep
        }

        # Create action sequence (preceding + failure action)
        action_sequence = failure.preceding_actions.copy()
        action_sequence.append(failure.action_at_failure)

        # Determine difficulty based on sequence length and error type. Would play around with this to see what counts for each
        seq_length = len(action_sequence)
        if seq_length <= 3:
            difficulty = "easy"
        elif seq_length <= 7:
            difficulty = "medium"
        else:
            difficulty = "hard"

        # Adjust difficulty based on error recoverability
        if not failure.error_metadata.get("recoverable", True):
            difficulty = "hard" if difficulty == "medium" else difficulty

        bundle = ReproBundle(
            bundle_id=f"bundle_{failure.run_id}",
            cluster_id=cluster_id,
            error_type=failure.error_type,
            error_message=failure.error_message,
            failure_run_id=failure.run_id,
            initial_state=initial_state,
            action_sequence=action_sequence,
            expected_outcome="Action sequence completes successfully",
            actual_outcome=f"Failure at step {len(action_sequence)}: {failure.error_message}",
            created_at=datetime.now().isoformat(),
            source_failures=[failure.run_id],
            difficulty=difficulty
        )

        return bundle

    def create_bundle_from_cluster(self, cluster_failures: List[RunFailure],
                                   cluster_id: int) -> ReproBundle:
        """
        Create a bundle from a cluster of similar failures.

        Uses the exemplar failure but includes metadata about all failures.
        """
        # Use the first failure as exemplar
        exemplar = cluster_failures[0]

        bundle = self.create_bundle_from_failure(exemplar, cluster_id=cluster_id)

        # Update source failures to include all in cluster
        bundle.source_failures = [f.run_id for f in cluster_failures]

        # Update bundle_id to reflect cluster
        bundle.bundle_id = f"bundle_cluster_{cluster_id}"

        return bundle

    def minimize_action_sequence(self, bundle: ReproBundle) -> ReproBundle:
        """
        Attempt to minimize the action sequence while preserving the failure.

        This is a simplified heuristic - in practice, you'd need to test each minimization.
        """
        # For now, just keep the last 3 actions if sequence is longer
        if len(bundle.action_sequence) > 3:
            bundle.action_sequence = bundle.action_sequence[-3:]

        return bundle

    def save_bundle(self, bundle: ReproBundle) -> Path:
        """Save a bundle to disk as JSON."""
        bundle_path = self.output_dir / f"{bundle.bundle_id}.json"

        with open(bundle_path, 'w') as f:
            json.dump(bundle.to_dict(), f, indent=2)

        return bundle_path

    def create_bundles_from_failures(self, failures: List[RunFailure],
                                    minimize: bool = False) -> List[ReproBundle]:
        """Create bundles from a list of failures."""
        print(f"Creating bundles from {len(failures)} failures...")

        bundles = []
        for failure in failures:
            bundle = self.create_bundle_from_failure(failure)

            if minimize:
                bundle = self.minimize_action_sequence(bundle)

            bundles.append(bundle)
            self.save_bundle(bundle)

        self.bundles = bundles
        print(f"Created {len(bundles)} bundles in {self.output_dir}")

        return bundles

    def create_bundles_from_clusters(self, clusters_file: str,
                                    failures_file: str) -> List[ReproBundle]:
        """Create bundles from cluster data."""
        print("Loading clusters and failures...")

        # Load clusters
        with open(clusters_file, 'r') as f:
            cluster_data = json.load(f)

        # Load failures
        with open(failures_file, 'r') as f:
            failures_data = json.load(f)

        # Convert to RunFailure objects
        failures_by_id = {}
        for f_data in failures_data:
            failure = RunFailure(
                run_id=f_data["run_id"],
                error_type=f_data["error_type"],
                error_message=f_data["error_message"],
                failure_timestep=f_data["failure_timestep"],
                total_timesteps=f_data["total_timesteps"],
                action_at_failure=f_data["action_at_failure"],
                observation_at_failure=f_data["observation_at_failure"],
                preceding_actions=f_data["preceding_actions"],
                error_metadata=f_data["error_metadata"],
                timestamp=f_data["timestamp"]
            )
            failures_by_id[failure.run_id] = failure

        # Create bundles for each cluster
        bundles = []
        for cluster in cluster_data.get("clusters", []):
            cluster_id = cluster["cluster_id"]
            failure_ids = cluster["failure_ids"]

            cluster_failures = [
                failures_by_id[fid] for fid in failure_ids
                if fid in failures_by_id
            ]

            if cluster_failures:
                bundle = self.create_bundle_from_cluster(cluster_failures, cluster_id)
                bundles.append(bundle)
                self.save_bundle(bundle)

        self.bundles = bundles
        print(f"Created {len(bundles)} cluster bundles in {self.output_dir}")

        return bundles

    def create_manifest(self) -> Path:
        """Create a manifest file listing all bundles."""
        manifest_path = self.output_dir / "manifest.json"

        manifest = {
            "created_at": datetime.now().isoformat(),
            "total_bundles": len(self.bundles),
            "bundles": [
                {
                    "bundle_id": b.bundle_id,
                    "error_type": b.error_type,
                    "difficulty": b.difficulty,
                    "num_actions": len(b.action_sequence),
                    "source_failures_count": len(b.source_failures)
                }
                for b in self.bundles
            ]
        }

        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        print(f"✓ Created manifest at {manifest_path}")
        return manifest_path


def main():
    """Example usage of BundleCreator."""
    import argparse

    parser = argparse.ArgumentParser(description="Create reproducible bundles from failures")
    parser.add_argument("--failures", required=True, help="Failures JSON file")
    parser.add_argument("--clusters", help="Optional clusters JSON file")
    parser.add_argument("--output-dir", default="output/bundles",
                       help="Output directory for bundles")
    parser.add_argument("--minimize", action="store_true",
                       help="Minimize action sequences")

    args = parser.parse_args()

    creator = BundleCreator(output_dir=args.output_dir)

    if args.clusters:
        # Create bundles from clusters
        creator.create_bundles_from_clusters(args.clusters, args.failures)
    else:
        # Create individual bundles from failures
        with open(args.failures, 'r') as f:
            failures_data = json.load(f)

        failures = [
            RunFailure(
                run_id=f["run_id"],
                error_type=f["error_type"],
                error_message=f["error_message"],
                failure_timestep=f["failure_timestep"],
                total_timesteps=f["total_timesteps"],
                action_at_failure=f["action_at_failure"],
                observation_at_failure=f["observation_at_failure"],
                preceding_actions=f["preceding_actions"],
                error_metadata=f["error_metadata"],
                timestamp=f["timestamp"]
            )
            for f in failures_data
        ]

        creator.create_bundles_from_failures(failures, minimize=args.minimize)

    creator.create_manifest()


if __name__ == "__main__":
    main()
