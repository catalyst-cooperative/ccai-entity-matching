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

# Default config
DEFAULT_CONFIG = {
    "inputs": {
        "start_year": 2019,
        "end_year": 2020,
    },
    "embedding": {
        "embedding_map": {
            "plant_name": {"embedding_type": "tfidf_vectorize"},
            "utility_name": {"embedding_type": "tfidf_vectorize"},
            "fuel_type_code_pudl": {"embedding_type": "tfidf_vectorize"},
            "installation_year": {"embedding_type": "min_max_scale"},
            "construction_year": {"embedding_type": "min_max_scale"},
            "capacity_mw": {"embedding_type": "min_max_scale"},
        },
        "matching_cols": [
            "plant_name",
            "utility_name",
            "installation_year",
            "construction_year",
            "fuel_type_code_pudl",
            "capacity_mw",
            "report_year",
        ],
        "blocking_col": "report_year",
    },
    "similarity_search": {"distance_metric": "l2_distance_search"},
}


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
    """Record important metrics from blocking step using mlflow."""
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


def execute_blocking(
    pudl_engine: sa.engine.Engine,
    mlruns: Path = Path("./mlruns/"),
    run_tags: dict | None = None,
):
    """Set up and measure blocking step."""
    # set configuration for model
    model_config = config.Model(**DEFAULT_CONFIG)  # type: ignore

    model_inputs = inputs.InputManager(
        pudl_engine=pudl_engine,
        start_report_year=model_config.inputs.start_year,
        end_report_year=model_config.inputs.end_year,
    )

    ferc_df = model_inputs.get_ferc_input()
    eia_df = model_inputs.get_eia_input()

    ferc_left = ferc_df[model_config.embedding.matching_cols].reset_index()
    eia_right = eia_df[model_config.embedding.matching_cols].reset_index()

    embedder = candidate_set_creation.DataframeEmbedder(
        left_df=ferc_left,
        right_df=eia_right,
        embedding_map=model_config.embedding.embedding_map,
    )
    embedder.embed_dataframes(blocking_col=model_config.embedding.blocking_col)

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
    ks = [5, 10, 15, 20, 25, 30, 40, 50]
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
