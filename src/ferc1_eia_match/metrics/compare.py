"""Script to compare sets of metrics."""
import argparse
import sys
from pathlib import Path

import mlflow  # type: ignore
import pandas as pd

import pudl

# Default dir is the base of repo
MLRUNS_DEFAULT = Path(__file__).parent.parent.parent.parent / "mlruns/"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two sets of metrics from a specified experiment and git hashes."
    )
    parser.add_argument(
        "base_commit",
        type=str,
        help="Commit hash associated with baseline metrics.",
    )
    parser.add_argument(
        "experiment_commit",
        type=str,
        help="Commit hash associated with experimental metrics.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment to compare.",
        default="blocking",
    )

    parser.add_argument(
        "--mlrun_uri",
        type=str,
        help="Path to MLflow tracking backend location.",
        default=MLRUNS_DEFAULT,
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Path to pudl.sqlite DB.",
        default=pudl.workspace.setup.get_defaults()["pudl_db"],
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to file where formatted markdown results are written.",
        default="./experiment_results.md",
    )

    return parser.parse_args()


def _get_columns(columns: list[str]) -> list[str]:
    columns = [c for c in columns if c.startswith(("params", "metrics"))]
    columns.insert(0, "source")
    return columns


def main():
    """Search experiment results desired runs and write results to markdown."""
    args = _parse_args()
    metrics = mlflow.search_runs(experiment_names=[args.experiment])

    baseline_metrics = metrics[
        metrics["tags.mlflow.source.git.commit"] == args.base_commit
    ]
    experimental_metrics = metrics[
        metrics["tags.mlflow.source.git.commit"] == args.experiment_commit
    ]

    baseline_metrics["source"] = "baseline"
    experimental_metrics["source"] = "experimental"

    columns = _get_columns(metrics.columns)
    metrics = pd.concat([baseline_metrics, experimental_metrics])[columns]
    metrics.sort_values(by=["params.k"])

    metrics.to_markdown(buf=args.output_file)


if __name__ == "__main__":
    sys.exit(main())
