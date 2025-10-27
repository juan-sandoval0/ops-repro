#!/usr/bin/env python3
"""
Evaluation harness for testing policies against reproducible bundles.
Runs stubbed policies and tracks success/failure rates.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict


@dataclass
class EvalResult:
    """Result of evaluating a policy on a bundle."""
    bundle_id: str
    policy_name: str
    success: bool
    steps_executed: int
    failure_step: Optional[int]
    error_encountered: Optional[str]
    execution_time_ms: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "policy_name": self.policy_name,
            "success": self.success,
            "steps_executed": self.steps_executed,
            "failure_step": self.failure_step,
            "error_encountered": self.error_encountered,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }


class StubPolicy:
    """
    Stubbed policy for testing.
    Simulates policy execution with configurable failure rates.
    """

    def __init__(self, name: str, success_rate: float = 0.5):
        self.name = name
        self.success_rate = success_rate

    def execute(self, bundle: Dict[str, Any]) -> EvalResult:
        """Execute policy on a bundle."""
        start_time = datetime.now()

        action_sequence = bundle.get("action_sequence", [])
        steps_executed = 0
        success = False
        failure_step = None
        error_encountered = None

        # Simulate execution
        for idx, action in enumerate(action_sequence):
            steps_executed += 1

            # Randomly fail based on success rate
            if random.random() > self.success_rate:
                failure_step = idx
                error_encountered = f"Policy failed at action: {action.get('type', 'unknown')}"
                break

        # If we got through all steps, it's a success
        if failure_step is None:
            success = True

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return EvalResult(
            bundle_id=bundle.get("bundle_id", "unknown"),
            policy_name=self.name,
            success=success,
            steps_executed=steps_executed,
            failure_step=failure_step,
            error_encountered=error_encountered,
            execution_time_ms=execution_time,
            metadata={"success_rate": self.success_rate}
        )


class ImprovedPolicy(StubPolicy):
    """
    A 'better' policy that handles certain error types.
    """

    def __init__(self, name: str = "improved_policy"):
        super().__init__(name, success_rate=0.7)  # Higher success rate

    def execute(self, bundle: Dict[str, Any]) -> EvalResult:
        """Execute with special handling for known error types."""
        start_time = datetime.now()

        action_sequence = bundle.get("action_sequence", [])
        error_type = bundle.get("error_type", "")

        steps_executed = 0
        success = False
        failure_step = None
        error_encountered = None

        # Special handling for certain error types
        handled_errors = ["gripper_slip", "grasp_failure", "timeout"]

        if error_type in handled_errors:
            # Higher success rate for handled errors
            local_success_rate = 0.9
        else:
            local_success_rate = self.success_rate

        for idx, action in enumerate(action_sequence):
            steps_executed += 1

            if random.random() > local_success_rate:
                failure_step = idx
                error_encountered = f"Policy failed at action: {action.get('type', 'unknown')}"
                break

        if failure_step is None:
            success = True

        execution_time = (datetime.now() - start_time).total_seconds() * 1000

        return EvalResult(
            bundle_id=bundle.get("bundle_id", "unknown"),
            policy_name=self.name,
            success=success,
            steps_executed=steps_executed,
            failure_step=failure_step,
            error_encountered=error_encountered,
            execution_time_ms=execution_time,
            metadata={
                "success_rate": local_success_rate,
                "handled_error": error_type in handled_errors
            }
        )


class EvalHarness:
    """Evaluation harness for testing policies against bundles."""

    def __init__(self, bundles_dir: str = "output/bundles"):
        self.bundles_dir = Path(bundles_dir)
        self.bundles: List[Dict[str, Any]] = []
        self.results: List[EvalResult] = []

    def load_bundles(self) -> None:
        """Load all bundles from the bundles directory."""
        print(f"Loading bundles from {self.bundles_dir}...")

        bundle_files = list(self.bundles_dir.glob("bundle_*.json"))

        for bundle_file in bundle_files:
            with open(bundle_file, 'r') as f:
                bundle = json.load(f)
                self.bundles.append(bundle)

        print(f"✓ Loaded {len(self.bundles)} bundles")

    def evaluate_policy(self, policy: StubPolicy,
                       filter_difficulty: Optional[str] = None,
                       filter_error_type: Optional[str] = None) -> List[EvalResult]:
        """
        Evaluate a policy on all bundles (or filtered subset).

        Args:
            policy: The policy to evaluate
            filter_difficulty: Optional difficulty filter ("easy", "medium", "hard")
            filter_error_type: Optional error type filter
        """
        print(f"\nEvaluating policy: {policy.name}")

        # Filter bundles
        bundles_to_eval = self.bundles

        if filter_difficulty:
            bundles_to_eval = [
                b for b in bundles_to_eval
                if b.get("difficulty") == filter_difficulty
            ]
            print(f"  Filtered to {len(bundles_to_eval)} {filter_difficulty} bundles")

        if filter_error_type:
            bundles_to_eval = [
                b for b in bundles_to_eval
                if b.get("error_type") == filter_error_type
            ]
            print(f"  Filtered to {len(bundles_to_eval)} bundles with error type '{filter_error_type}'")

        # Run evaluation
        results = []
        for bundle in bundles_to_eval:
            result = policy.execute(bundle)
            results.append(result)

        self.results.extend(results)

        # Print summary
        successes = sum(1 for r in results if r.success)
        success_rate = successes / len(results) if results else 0

        print(f"  Results: {successes}/{len(results)} successful ({success_rate:.1%})")

        return results

    def compare_policies(self, policies: List[StubPolicy]) -> Dict[str, Any]:
        """Compare multiple policies on the same bundle set."""
        print("\n" + "="*60)
        print("POLICY COMPARISON")
        print("="*60)

        comparison = {}

        for policy in policies:
            results = self.evaluate_policy(policy)

            successes = sum(1 for r in results if r.success)
            success_rate = successes / len(results) if results else 0
            avg_execution_time = sum(r.execution_time_ms for r in results) / len(results) if results else 0

            comparison[policy.name] = {
                "success_rate": success_rate,
                "successes": successes,
                "total": len(results),
                "avg_execution_time_ms": avg_execution_time
            }

        # Print comparison table
        print("\n{:<20} {:<15} {:<20}".format("Policy", "Success Rate", "Avg Time (ms)"))
        print("-" * 60)
        for name, stats in comparison.items():
            print("{:<20} {:<15} {:<20.2f}".format(
                name,
                f"{stats['success_rate']:.1%}",
                stats['avg_execution_time_ms']
            ))

        return comparison

    def analyze_by_error_type(self, policy_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze results broken down by error type."""
        # Filter results by policy if specified
        results_to_analyze = self.results
        if policy_name:
            results_to_analyze = [r for r in self.results if r.policy_name == policy_name]

        # Group by error type (need to load bundles to get error types)
        bundle_error_types = {b["bundle_id"]: b["error_type"] for b in self.bundles}

        error_type_results = defaultdict(list)
        for result in results_to_analyze:
            error_type = bundle_error_types.get(result.bundle_id, "unknown")
            error_type_results[error_type].append(result)

        # Calculate statistics per error type
        analysis = {}
        for error_type, results in error_type_results.items():
            successes = sum(1 for r in results if r.success)
            success_rate = successes / len(results) if results else 0

            analysis[error_type] = {
                "success_rate": success_rate,
                "successes": successes,
                "total": len(results)
            }

        return analysis

    def save_results(self, output_file: str) -> None:
        """Save evaluation results to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_data = {
            "evaluated_at": datetime.now().isoformat(),
            "total_bundles": len(self.bundles),
            "total_evaluations": len(self.results),
            "results": [r.to_dict() for r in self.results]
        }

        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"\n✓ Saved results to {output_path}")


def main():
    """Example usage of EvalHarness."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate policies on reproducible bundles")
    parser.add_argument("--bundles-dir", default="output/bundles",
                       help="Directory containing bundles")
    parser.add_argument("--output", default="output/eval_results.json",
                       help="Output file for results")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple policies")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    random.seed(args.seed)

    harness = EvalHarness(bundles_dir=args.bundles_dir)
    harness.load_bundles()

    if args.compare:
        # Compare baseline vs improved policy
        baseline = StubPolicy("baseline", success_rate=0.5)
        improved = ImprovedPolicy("improved")

        comparison = harness.compare_policies([baseline, improved])

        # Analyze by error type for improved policy
        print("\n" + "="*60)
        print("ERROR TYPE ANALYSIS (Improved Policy)")
        print("="*60)
        analysis = harness.analyze_by_error_type(policy_name="improved")

        print("\n{:<25} {:<15}".format("Error Type", "Success Rate"))
        print("-" * 45)
        for error_type, stats in sorted(analysis.items(), key=lambda x: -x[1]['success_rate']):
            print("{:<25} {:<15}".format(
                error_type,
                f"{stats['success_rate']:.1%} ({stats['successes']}/{stats['total']})"
            ))

    else:
        # Just run baseline
        baseline = StubPolicy("baseline", success_rate=0.5)
        harness.evaluate_policy(baseline)

    harness.save_results(args.output)


if __name__ == "__main__":
    main()
