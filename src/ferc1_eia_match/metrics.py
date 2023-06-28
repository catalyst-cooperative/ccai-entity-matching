"""Helper functions to compute and output various metrics."""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from ferc1_eia_match import config

logger = logging.getLogger(__name__)

# Default dir is the base of repo
MLRUNS_DEFAULT = Path(__file__).parent.parent.parent / "mlruns/"


def measure_blocking(
    candidate_matcher: Callable,
    ks: list[int],
    train_df: pd.DataFrame,
    ferc_left: pd.DataFrame,
    eia_right: pd.DataFrame,
    model: config.Model,
    mlruns: Path = MLRUNS_DEFAULT,
):
    """Record important metrics from blocking step using mlflow."""
    mlflow.set_tracking_uri(f"file:{str(mlruns)}")
    train_df_with_idx = train_df.merge(
        ferc_left.reset_index(names="ferc_index")[["record_id_ferc1", "ferc_index"]],
        how="inner",
        on="record_id_ferc1",
    )
    train_df_with_idx = train_df_with_idx.merge(
        eia_right.reset_index(names="eia_index")[["record_id_eia", "eia_index"]],
        how="inner",
        on="record_id_eia",
    )
    ferc_train_idx = train_df_with_idx.ferc_index
    eia_train_idx = train_df_with_idx.eia_index

    metric = model.similarity_search.distance_metric
    for k in ks:
        with mlflow.start_run():
            # Log model config and parameters
            mlflow.log_dict(model.dict(), "blocking_model_config")
            mlflow.log_params({"k": k})

            # Run blocking
            candidate_set = candidate_matcher(k, metric)

            # Compute % that capture match in candidate set
            pair_is_correct = np.in1d(eia_train_idx, candidate_set[ferc_train_idx])
            n_correct_pairs = np.sum(pair_is_correct)
            mlflow.log_metric(
                "percent_capture", n_correct_pairs / len(train_df_with_idx)
            )
            logger.info(
                f"k: {k}, metric: {metric} percent_capture: {n_correct_pairs/len(train_df_with_idx)}"
            )

            # Compute % that predict match as first value in candidate set
            first_match_is_correct = np.in1d(
                eia_train_idx, candidate_set[ferc_train_idx][:, 0]
            )
            n_first_match_correct = np.sum(first_match_is_correct)
            mlflow.log_metric(
                "percent_first_match", n_first_match_correct / len(train_df_with_idx)
            )
            logger.info(
                f"k: {k}, metric: {metric} percent_first: {n_first_match_correct/len(train_df_with_idx)}"
            )
