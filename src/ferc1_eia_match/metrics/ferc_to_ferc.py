"""Helper functions to compute and output various metrics."""
from __future__ import annotations

import logging
from pathlib import Path

import mlflow  # type: ignore
import pandas as pd

logger = logging.getLogger(__name__)


def measure_matching(
    plants_labels_df: pd.DataFrame,
    distance_thresh: float,
    mlruns: Path = Path("./mlruns/"),
    run_tags: dict | None = None,
):
    """Measure ferc-ferc matching."""
    logger.info(f"Measuring FERC-FERC matching and saving results at {mlruns}")

    with mlflow.start_run(tags=run_tags):
        mlflow.set_tracking_uri(f"file:{str(mlruns)}")
        mlflow.set_experiment(experiment_name="ferc_to_ferc")
        mlflow.log_params({"distance_thresh": distance_thresh})

        year_counts = (
            plants_labels_df.groupby(by=["id", "report_year"]).size().value_counts()
        )
        mlflow.log_metric("ratio_single_year", year_counts[1] / len(plants_labels_df))
        mlflow.log_metric(
            "avg_cluster_size", plants_labels_df.groupby("id").size().mean()
        )
