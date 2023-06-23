"""Dagster definitions for the entity matching steps."""

from dagster import (
    Definitions,
    EnvVar,
    RunConfig,
    define_asset_job,
    load_assets_from_modules,
)

from ferc1_eia_match import candidate_set_creation, inputs, resources

default_assets = (
    *load_assets_from_modules([inputs], group_name="inputs"),
    *load_assets_from_modules(
        [candidate_set_creation], group_name="candidate_set_creation"
    ),
)


defs: Definitions = Definitions(
    assets=default_assets,
    resources={
        "pudl_input_resource": resources.PudlInputs(
            start_year=None,
            end_year=None,
            pudl_db_path=EnvVar("PUDL_OUTPUT"),
        ),
    },
    jobs=[
        define_asset_job(
            name="inputs",
            config=RunConfig(
                ops={
                    "eia_input": inputs.EiaInputConfig(
                        update=False, eia_plant_part=None
                    ),
                    "embed_dataframes": candidate_set_creation.EmbeddingConfig(
                        embedding_map={
                            "plant_name": candidate_set_creation.ColumnEmbedding(
                                embedding_type="tfidf_vectorize",
                            ),
                            "utility_name": candidate_set_creation.ColumnEmbedding(
                                embedding_type="tfidf_vectorize",
                            ),
                            "fuel_type_code_pudl": candidate_set_creation.ColumnEmbedding(
                                embedding_type="tfidf_vectorize",
                            ),
                            "installation_year": candidate_set_creation.ColumnEmbedding(
                                embedding_type="min_max_scale",
                            ),
                            "construction_year": candidate_set_creation.ColumnEmbedding(
                                embedding_type="min_max_scale",
                            ),
                            "capacity_mw": candidate_set_creation.ColumnEmbedding(
                                embedding_type="min_max_scale",
                            ),
                        },
                        matching_cols=[
                            "plant_name",
                            "utility_name",
                            "installation_year",
                            "construction_year",
                            "fuel_type_code_pudl",
                            "capacity_mw",
                            "report_year",
                        ],
                        blocking_col="report_year",
                    ),
                },
            ),
            description="Job to generate input assets.",
        )
    ],
)
