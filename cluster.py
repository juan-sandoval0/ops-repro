#!/usr/bin/env python3
"""
Clustering module for grouping similar failures using error signatures and embeddings.
Two-stage clustering: first by signature, then HDBSCAN within each signature group.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

import hdbscan
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

from ingest import RunFailure


@dataclass
class FailureCluster:
    """Represents a cluster of similar failures."""
    cluster_id: int
    signature: str
    subcluster_id: int  # Subcluster ID within signature group
    failure_ids: List[str]
    exemplar_id: str  # Representative failure for this cluster
    size: int
    error_types: List[str]
    action_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "signature": self.signature,
            "subcluster_id": int(self.subcluster_id) if isinstance(self.subcluster_id, (np.integer, np.int64)) else self.subcluster_id,
            "failure_ids": self.failure_ids,
            "exemplar_id": self.exemplar_id,
            "size": self.size,
            "error_types": list(set(self.error_types)),
            "action_types": list(set(self.action_types))
        }


class FailureClusterer:
    """Clusters similar failures using signatures and semantic embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize clusterer.

        Args:
            model_name: SentenceTransformer model name for embeddings
        """
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.failures: List[RunFailure] = []
        self.clusters: List[FailureCluster] = []

    def load_failures(self, failures_file: str) -> None:
        """Load failures from JSON file."""
        print(f"Loading failures from {failures_file}...")

        with open(failures_file, 'r') as f:
            failures_data = json.load(f)

        # Convert back to RunFailure objects
        self.failures = [
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

        print(f"Loaded {len(self.failures)} failures")

    def _create_failure_text(self, failure: RunFailure) -> str:
        """Create text representation of failure for embedding."""
        action_sequence = " -> ".join([
            a.get("type", "unknown") for a in failure.preceding_actions
        ])
        current_action = failure.action_at_failure.get("type", "unknown")

        text = (
            f"Error: {failure.error_type}. "
            f"Message: {failure.error_message}. "
            f"Action sequence: {action_sequence} -> {current_action}. "
            f"Failed at timestep {failure.failure_timestep}."
        )

        return text

    def cluster_by_signature(self) -> Dict[str, List[RunFailure]]:
        """Simple clustering by error signature (error_type::action_type)."""
        print("Clustering by signature...")

        signature_clusters = defaultdict(list)

        for failure in self.failures:
            sig = failure.get_signature()
            signature_clusters[sig].append(failure)

        print(f"Found {len(signature_clusters)} signature-based clusters")

        return signature_clusters

    def _extract_numeric_features(self, failure: RunFailure) -> np.ndarray:
        """Extract numeric features from a failure for clustering."""
        features = []

        # Timestep features
        features.append(failure.failure_timestep)
        features.append(failure.total_timesteps)
        features.append(failure.failure_timestep / max(failure.total_timesteps, 1))  # normalized position

        # Action sequence length
        features.append(len(failure.preceding_actions))

        # Robot state features (if available)
        robot_state = failure.observation_at_failure.get("robot_state", {})
        if "gripper_open" in robot_state:
            features.append(1.0 if robot_state["gripper_open"] else 0.0)

        # Force/torque magnitude (if available)
        force_torque = failure.observation_at_failure.get("force_torque", [])
        if force_torque:
            features.append(np.linalg.norm(force_torque))
        else:
            features.append(0.0)

        # Number of detected objects
        detected_objects = failure.observation_at_failure.get("camera", {}).get("detected_objects", [])
        features.append(len(detected_objects))

        return np.array(features)

    def cluster_two_stage(self, min_cluster_size: int = 2, min_samples: int = 1) -> List[FailureCluster]:
        """
        Two-stage clustering:
        1. Group by signature (error_type::action_type)
        2. Within each signature group, run HDBSCAN on embeddings + numeric features

        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN core points
        """
        print("Starting two-stage clustering...")

        # Stage 1: Group by signature
        print("\nStage 1: Grouping by signature (error_type::action_type)...")
        signature_groups = self.cluster_by_signature()

        print(f"\nFound {len(signature_groups)} signature groups:")
        for sig, failures in sorted(signature_groups.items(), key=lambda x: -len(x[1])):
            print(f"  {sig}: {len(failures)} failures")

        # Stage 2: HDBSCAN within each signature group
        print("\nStage 2: Running HDBSCAN within each signature group...")

        all_clusters = []
        global_cluster_id = 0

        for signature, sig_failures in signature_groups.items():
            if len(sig_failures) < min_cluster_size:
                # Too few failures - create single cluster
                print(f"\n  {signature}: {len(sig_failures)} failures (too few, creating single cluster)")

                cluster = FailureCluster(
                    cluster_id=global_cluster_id,
                    signature=signature,
                    subcluster_id=0,
                    failure_ids=[f.run_id for f in sig_failures],
                    exemplar_id=sig_failures[0].run_id,
                    size=len(sig_failures),
                    error_types=[f.error_type for f in sig_failures],
                    action_types=[f.action_at_failure.get("type", "unknown") for f in sig_failures]
                )
                all_clusters.append(cluster)
                global_cluster_id += 1
                continue

            print(f"\n  {signature}: {len(sig_failures)} failures")

            # Generate embeddings for this signature group
            failure_texts = [self._create_failure_text(f) for f in sig_failures]
            embeddings = self.model.encode(failure_texts, show_progress_bar=False)

            # Extract numeric features
            numeric_features = np.array([self._extract_numeric_features(f) for f in sig_failures])

            # Normalize both feature sets
            scaler_emb = StandardScaler()
            scaler_num = StandardScaler()

            embeddings_scaled = scaler_emb.fit_transform(embeddings)
            numeric_scaled = scaler_num.fit_transform(numeric_features)

            # Combine embeddings and numeric features
            combined_features = np.hstack([embeddings_scaled, numeric_scaled])

            # Run HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            labels = clusterer.fit_predict(combined_features)

            # Count clusters (excluding noise)
            unique_labels = set(labels)
            n_clusters = len([l for l in unique_labels if l != -1])
            n_noise = sum(1 for l in labels if l == -1)

            print(f"    → {n_clusters} subclusters, {n_noise} noise points")

            # Create FailureCluster objects for each subcluster
            subcluster_groups = defaultdict(list)
            for idx, label in enumerate(labels):
                subcluster_groups[label].append(idx)

            for subcluster_label, indices in subcluster_groups.items():
                failures_in_subcluster = [sig_failures[i] for i in indices]

                cluster = FailureCluster(
                    cluster_id=global_cluster_id,
                    signature=signature,
                    subcluster_id=subcluster_label,
                    failure_ids=[f.run_id for f in failures_in_subcluster],
                    exemplar_id=failures_in_subcluster[0].run_id,
                    size=len(failures_in_subcluster),
                    error_types=[f.error_type for f in failures_in_subcluster],
                    action_types=[f.action_at_failure.get("type", "unknown") for f in failures_in_subcluster]
                )
                all_clusters.append(cluster)
                global_cluster_id += 1

        self.clusters = all_clusters

        # Print summary
        total_subclusters = len(all_clusters)
        noise_clusters = len([c for c in all_clusters if c.subcluster_id == -1])

        print(f"\n✓ Two-stage clustering complete:")
        print(f"  Stage 1: {len(signature_groups)} signature groups")
        print(f"  Stage 2: {total_subclusters} total clusters ({noise_clusters} noise clusters)")

        return all_clusters

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary statistics of clusters."""
        if not self.clusters:
            return {}

        cluster_sizes = [c.size for c in self.clusters]

        return {
            "total_clusters": len(self.clusters),
            "total_failures_clustered": sum(cluster_sizes),
            "avg_cluster_size": float(np.mean(cluster_sizes)),
            "median_cluster_size": float(np.median(cluster_sizes)),
            "max_cluster_size": int(max(cluster_sizes)),
            "min_cluster_size": int(min(cluster_sizes)),
            "clusters": [c.to_dict() for c in self.clusters]
        }

    def save_clusters(self, output_file: str) -> None:
        """Save clusters to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.get_cluster_summary()

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved {len(self.clusters)} clusters to {output_path}")

    def get_failures_in_cluster(self, cluster_id: int) -> List[RunFailure]:
        """Get all failures in a specific cluster."""
        if cluster_id >= len(self.clusters):
            return []

        cluster = self.clusters[cluster_id]
        return [f for f in self.failures if f.run_id in cluster.failure_ids]


def main():
    """Example usage of FailureClusterer."""
    import argparse

    parser = argparse.ArgumentParser(description="Cluster similar failures")
    parser.add_argument("input", help="Input failures JSON file")
    parser.add_argument("--output", default="output/clusters.json",
                       help="Output JSON file for clusters")
    parser.add_argument("--method", choices=["signature", "two-stage"],
                       default="two-stage", help="Clustering method")
    parser.add_argument("--min-cluster-size", type=int, default=2,
                       help="HDBSCAN min_cluster_size parameter")
    parser.add_argument("--min-samples", type=int, default=1,
                       help="HDBSCAN min_samples parameter")

    args = parser.parse_args()

    clusterer = FailureClusterer()
    clusterer.load_failures(args.input)

    if args.method == "signature":
        sig_clusters = clusterer.cluster_by_signature()
        print("\nSignature-based clusters:")
        for sig, failures in sorted(sig_clusters.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {sig}: {len(failures)} failures")
    else:
        # Two-stage clustering
        clusterer.cluster_two_stage(
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples
        )
        clusterer.save_clusters(args.output)

        summary = clusterer.get_cluster_summary()
        print(f"\nCluster summary:")
        print(f"  Total clusters: {summary['total_clusters']}")
        print(f"  Average cluster size: {summary['avg_cluster_size']:.1f}")


if __name__ == "__main__":
    main()
