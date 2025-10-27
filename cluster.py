#!/usr/bin/env python3
"""
Clustering module for grouping similar failures using error signatures and embeddings.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

from ingest import RunFailure


@dataclass
class FailureCluster:
    """Represents a cluster of similar failures."""
    cluster_id: int
    signature: str
    failure_ids: List[str]
    exemplar_id: str  # Representative failure for this cluster
    size: int
    error_types: List[str]
    action_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "signature": self.signature,
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

    def cluster_by_embedding(self, eps: float = 0.5, min_samples: int = 2) -> List[FailureCluster]:
        """Cluster failures using semantic embeddings and DBSCAN."""
        print("Generating embeddings for failures...")

        # Create text representations
        failure_texts = [self._create_failure_text(f) for f in self.failures]

        # Generate embeddings
        embeddings = self.model.encode(failure_texts, show_progress_bar=True)

        # Normalize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        print(f"Clustering with DBSCAN (eps={eps}, min_samples={min_samples})...")

        # Cluster with DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        # Group failures by cluster
        cluster_groups = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_groups[label].append(idx)

        print(f"Found {len([l for l in cluster_groups.keys() if l != -1])} clusters")
        print(f"Outliers (unclustered): {len(cluster_groups.get(-1, []))}")

        # Create FailureCluster objects
        clusters = []
        cluster_id = 0

        for label, indices in cluster_groups.items():
            if label == -1:  # Skip outliers
                continue

            failures_in_cluster = [self.failures[i] for i in indices]

            # Determine most common signature
            signatures = [f.get_signature() for f in failures_in_cluster]
            most_common_sig = max(set(signatures), key=signatures.count)

            # Select exemplar (first failure in cluster for simplicity)
            exemplar = failures_in_cluster[0]

            cluster = FailureCluster(
                cluster_id=cluster_id,
                signature=most_common_sig,
                failure_ids=[f.run_id for f in failures_in_cluster],
                exemplar_id=exemplar.run_id,
                size=len(failures_in_cluster),
                error_types=[f.error_type for f in failures_in_cluster],
                action_types=[f.action_at_failure.get("type", "unknown")
                             for f in failures_in_cluster]
            )

            clusters.append(cluster)
            cluster_id += 1

        self.clusters = clusters
        return clusters

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary statistics of clusters."""
        if not self.clusters:
            return {}

        cluster_sizes = [c.size for c in self.clusters]

        return {
            "total_clusters": len(self.clusters),
            "total_failures_clustered": sum(cluster_sizes),
            "avg_cluster_size": np.mean(cluster_sizes),
            "median_cluster_size": np.median(cluster_sizes),
            "max_cluster_size": max(cluster_sizes),
            "min_cluster_size": min(cluster_sizes),
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
    parser.add_argument("--method", choices=["signature", "embedding", "both"],
                       default="embedding", help="Clustering method")
    parser.add_argument("--eps", type=float, default=0.5,
                       help="DBSCAN eps parameter")
    parser.add_argument("--min-samples", type=int, default=2,
                       help="DBSCAN min_samples parameter")

    args = parser.parse_args()

    clusterer = FailureClusterer()
    clusterer.load_failures(args.input)

    if args.method in ["signature", "both"]:
        sig_clusters = clusterer.cluster_by_signature()
        print("\nSignature-based clusters:")
        for sig, failures in sorted(sig_clusters.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {sig}: {len(failures)} failures")

    if args.method in ["embedding", "both"]:
        clusterer.cluster_by_embedding(eps=args.eps, min_samples=args.min_samples)
        clusterer.save_clusters(args.output)

        summary = clusterer.get_cluster_summary()
        print(f"\nCluster summary:")
        print(f"  Total clusters: {summary['total_clusters']}")
        print(f"  Average cluster size: {summary['avg_cluster_size']:.1f}")


if __name__ == "__main__":
    main()
