#!/usr/bin/env python3
"""
piops-repro: CLI tool for incident triage.

Ingests robot run logs, clusters similar failures, and creates
minimal repros for evaluation.
"""

import click
from pathlib import Path

from mockdata.generate import RobotLogGenerator
from ingest import LogIngestor
from cluster import FailureClusterer
from bundle import BundleCreator
from eval_harness import EvalHarness, StubPolicy, ImprovedPolicy


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    ops-repro: Incident triage tool for robot run failures.

    Helps you ingest logs, cluster failures, create reproducible bundles,
    and evaluate policies.
    """
    pass


@cli.command()
@click.option('--num-runs', default=100, help='Number of robot runs to generate')
@click.option('--run-length', default=50, help='Timesteps per run')
@click.option('--failure-prob', default=0.3, help='Probability of failure (0.0-1.0)')
@click.option('--output', default='mockdata/robot_logs.jsonl', help='Output JSONL file')
@click.option('--seed', default=42, help='Random seed')
def generate(num_runs, run_length, failure_prob, output, seed):
    """Generate mock robot run logs."""
    click.echo(click.style("Generating mock robot logs...", fg='cyan', bold=True))

    generator = RobotLogGenerator(seed=seed)
    generator.generate_dataset(
        num_runs=num_runs,
        run_length=run_length,
        failure_prob=failure_prob,
        output_file=output
    )

    click.echo(click.style(f"\n Logs saved to {output}", fg='green'))


@cli.command()
@click.argument('input_file')
@click.option('--output', default='output/failures.json', help='Output failures file')
# Look at function for changing this
@click.option('--context-window', default=5, help='Number of preceding actions to include')
def ingest(input_file, output, context_window):
    """Ingest JSONL logs and extract failures."""
    click.echo(click.style("Ingesting logs...", fg='cyan', bold=True))

    ingestor = LogIngestor(context_window=context_window)
    ingestor.load_logs(input_file)
    ingestor.extract_failures()
    ingestor.save_failures(output)

    click.echo(click.style(f"\n Failures saved to {output}", fg='green'))


@cli.command()
@click.argument('failures_file')
@click.option('--output', default='output/clusters.json', help='Output clusters file')
@click.option('--method', type=click.Choice(['signature', 'embedding', 'both']),
              default='embedding', help='Clustering method')
@click.option('--eps', default=0.5, help='DBSCAN eps parameter')
@click.option('--min-samples', default=2, help='DBSCAN min_samples parameter')
def cluster(failures_file, output, method, eps, min_samples):
    """Cluster similar failures."""
    click.echo(click.style("Clustering failures...", fg='cyan', bold=True))

    clusterer = FailureClusterer()
    clusterer.load_failures(failures_file)

    if method in ['signature', 'both']:
        sig_clusters = clusterer.cluster_by_signature()
        click.echo("\nTop signature-based clusters:")
        for sig, failures in sorted(sig_clusters.items(), key=lambda x: -len(x[1]))[:10]:
            click.echo(f"  {sig}: {len(failures)} failures")

    if method in ['embedding', 'both']:
        clusterer.cluster_by_embedding(eps=eps, min_samples=min_samples)
        clusterer.save_clusters(output)

        summary = clusterer.get_cluster_summary()
        click.echo(f"\nCluster summary:")
        click.echo(f"  Total clusters: {summary['total_clusters']}")
        click.echo(f"  Average cluster size: {summary['avg_cluster_size']:.1f}")

    click.echo(click.style(f"\n Clusters saved to {output}", fg='green'))


@cli.command()
@click.argument('failures_file')
@click.option('--clusters', help='Optional clusters JSON file')
@click.option('--output-dir', default='output/bundles', help='Output directory for bundles')
@click.option('--minimize', is_flag=True, help='Minimize action sequences')
def bundle(failures_file, clusters, output_dir, minimize):
    """Create reproducible bundles from failures."""
    click.echo(click.style("Creating reproducible bundles...", fg='cyan', bold=True))

    creator = BundleCreator(output_dir=output_dir)

    if clusters:
        creator.create_bundles_from_clusters(clusters, failures_file)
    else:
        import json
        with open(failures_file, 'r') as f:
            failures_data = json.load(f)

        from ingest import RunFailure
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

        creator.create_bundles_from_failures(failures, minimize=minimize)

    creator.create_manifest()

    click.echo(click.style(f"\n✓ Done! Bundles saved to {output_dir}", fg='green'))


@cli.command()
@click.option('--bundles-dir', default='output/bundles', help='Directory containing bundles')
@click.option('--output', default='output/eval_results.json', help='Output results file')
@click.option('--compare', is_flag=True, help='Compare multiple policies')
@click.option('--seed', default=42, help='Random seed')
def evaluate(bundles_dir, output, compare, seed):
    """Evaluate policies on reproducible bundles."""
    click.echo(click.style("Evaluating policies...", fg='cyan', bold=True))

    import random
    random.seed(seed)

    harness = EvalHarness(bundles_dir=bundles_dir)
    harness.load_bundles()

    if compare:
        baseline = StubPolicy("baseline", success_rate=0.5)
        improved = ImprovedPolicy("improved")

        comparison = harness.compare_policies([baseline, improved])

        click.echo(click.style("\nError Type Analysis (Improved Policy)", fg='cyan', bold=True))
        analysis = harness.analyze_by_error_type(policy_name="improved")

        click.echo("\n{:<25} {:<15}".format("Error Type", "Success Rate"))
        click.echo("-" * 45)
        for error_type, stats in sorted(analysis.items(), key=lambda x: -x[1]['success_rate']):
            click.echo("{:<25} {:<15}".format(
                error_type,
                f"{stats['success_rate']:.1%} ({stats['successes']}/{stats['total']})"
            ))
    else:
        baseline = StubPolicy("baseline", success_rate=0.5)
        harness.evaluate_policy(baseline)

    harness.save_results(output)

    click.echo(click.style(f"\n Results saved to {output}", fg='green'))


@cli.command()
@click.option('--num-runs', default=100, help='Number of runs to generate')
@click.option('--seed', default=42, help='Random seed')
def pipeline(num_runs, seed):
    """
    Run the full pipeline: generate -> ingest -> cluster -> bundle -> evaluate.

    This is a convenience command that runs all steps in sequence.
    """
    click.echo(click.style("Running full pipeline...\n", fg='cyan', bold=True))

    import random
    random.seed(seed)

    # Step 1: Generate
    click.echo(click.style("1/5: Generating mock data...", fg='yellow', bold=True))
    generator = RobotLogGenerator(seed=seed)
    generator.generate_dataset(
        num_runs=num_runs,
        run_length=50,
        failure_prob=0.3,
        output_file='mockdata/robot_logs.jsonl'
    )

    # Step 2: Ingest
    click.echo(click.style("\n2/5: Ingesting logs...", fg='yellow', bold=True))
    ingestor = LogIngestor(context_window=5)
    ingestor.load_logs('mockdata/robot_logs.jsonl')
    ingestor.extract_failures()
    ingestor.save_failures('output/failures.json')

    # Step 3: Cluster
    click.echo(click.style("\n3/5: Clustering failures...", fg='yellow', bold=True))
    clusterer = FailureClusterer()
    clusterer.load_failures('output/failures.json')
    clusterer.cluster_by_embedding(eps=0.5, min_samples=2)
    clusterer.save_clusters('output/clusters.json')

    # Step 4: Bundle
    click.echo(click.style("\n4/5: Creating bundles...", fg='yellow', bold=True))
    creator = BundleCreator(output_dir='output/bundles')
    creator.create_bundles_from_clusters('output/clusters.json', 'output/failures.json')
    creator.create_manifest()

    # Step 5: Evaluate
    click.echo(click.style("\n5/5: Evaluating policies...", fg='yellow', bold=True))
    harness = EvalHarness(bundles_dir='output/bundles')
    harness.load_bundles()

    baseline = StubPolicy("baseline", success_rate=0.5)
    improved = ImprovedPolicy("improved")

    comparison = harness.compare_policies([baseline, improved])
    harness.save_results('output/eval_results.json')

    click.echo(click.style("\n" + "="*60, fg='green'))
    click.echo(click.style("✓ Pipeline complete!", fg='green', bold=True))
    click.echo(click.style("="*60, fg='green'))
    click.echo("\nGenerated artifacts:")
    click.echo("mockdata/robot_logs.jsonl - Raw logs")
    click.echo("output/failures.json - Extracted failures")
    click.echo("output/clusters.json - Failure clusters")
    click.echo("output/bundles/ - Reproducible bundles")
    click.echo("output/eval_results.json - Evaluation results")


if __name__ == '__main__':
    cli()
