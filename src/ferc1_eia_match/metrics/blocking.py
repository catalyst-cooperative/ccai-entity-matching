"""Helper functions to compute and output various metrics."""
from __future__ import annotations

import importlib.resources
import logging
from collections.abc import Callable
from pathlib import Path

import mlflow  # type: ignore
import numpy as np
import pandas as pd
import sqlalchemy as sa

from ferc1_eia_match import candidate_set_creation, config, inputs

logger = logging.getLogger(__name__)


def measure_blocking(
    candidate_matcher: Callable,
    ks: list[int],
    train_df: pd.DataFrame,
    ferc_left: pd.DataFrame,
    eia_right: pd.DataFrame,
    model: config.Model,
    mlruns: Path = Path("./mlruns/"),
    run_tags: dict | None = None,
):
    """Record a set of runs of the blocking step for each requested k value.

    Args:
        candidate_matcher: A function to generate a set of candidate matches.
        ks: List of k values to test.
        train_df: Dataframe of training data.
        ferc_left: FERC input data.
        eia_right: EIA input data.
        model: Configuration of model to be logged by mlflow.
        mlruns: Path to mlflow tracking directory.
        run_tags: Tags to help filter mlflow runs.
    """
    logger.info(f"Starting blocking experiment and saving results at {mlruns}")
    mlflow.set_tracking_uri(f"file:{str(mlruns)}")
    mlflow.set_experiment(experiment_name="blocking")

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
        logger.info(f"Run blocking with k={k}")
        with mlflow.start_run(tags=run_tags):
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


def run_blocking_tests(
    pudl_engine: sa.engine.Engine,
    mlruns: Path = Path("./mlruns/"),
    run_tags: dict | None = None,
    config_dict: dict | None = None,
    config_file: str | None = None,
    ks: list[int] = [5, 10, 15, 20, 25, 30, 40, 50],
):
    """Set up and measure blocking step.

    This function provides a repeatable test of the blocking step. It will prepare
    all input data and call measure_blocking.

    Args:
        pudl_engine: PUDL DB connection.
        mlruns: Path to mlruns directory.
        run_tags: Tags for filtering experiments.
        config_dict: Dictionary of model config will override a config_file.
        config_file: Dictionary of model config will override a config_file.
        ks: List of k values to test.
    """
    # set configuration for model
    config_source = importlib.resources.files("ferc1_eia_match.package_data").joinpath(
        "blocking_config.json"
    )
    with importlib.resources.as_file(config_source) as json_file:
        model_config = config.Model.from_json(json_file)

    if config_file:
        model_config = config.Model.from_json(json_file)

    if config_dict:
        model_config = config.Model(**config_dict)  # type: ignore

    # Prep inputs
    logger.info("Prepping inputs for blocking")
    model_inputs = inputs.InputManager(
        pudl_engine=pudl_engine,
        start_report_year=model_config.inputs.start_year,
        end_report_year=model_config.inputs.end_year,
    )

    ferc_df = model_inputs.get_ferc_input()
    eia_df = model_inputs.get_eia_input()

    ferc_left = ferc_df[model_config.embedding.matching_cols].reset_index()
    eia_right = eia_df[model_config.embedding.matching_cols].reset_index()

    logger.info("Embed dataframes")
    embedder = candidate_set_creation.DataframeEmbedder(
        left_df=ferc_left,
        right_df=eia_right,
        embedding_map=model_config.embedding.embedding_map,
    )
    embedder.embed_dataframes(blocking_col=model_config.embedding.blocking_col)

    logger.info("Create similarity searcher for blocking")
    searcher = candidate_set_creation.SimilaritySearcher(
        query_embedding_matrix=embedder.left_embedding_matrix,
        menu_embedding_matrix=embedder.right_embedding_matrix,
        query_blocks_dict=embedder.left_blocks_dict,
        menu_blocks_dict=embedder.right_blocks_dict,
    )

    # read in training data
    pkg_source = importlib.resources.files("ferc1_eia_match.package_data").joinpath(
        "ferc1_eia_train.csv"
    )
    with importlib.resources.as_file(pkg_source) as csv_file:
        train_df = pd.read_csv(csv_file)
    measure_blocking(
        searcher.run_candidate_pair_search,
        ks,
        train_df,
        ferc_left,
        eia_right,
        model_config,
        mlruns=mlruns,
        run_tags=run_tags,
    )
