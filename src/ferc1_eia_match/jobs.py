"""Dagster definitions for the entity matching steps."""
import importlib.resources

from dagster import (
    Definitions,
    EnvVar,
    RunConfig,
    define_asset_job,
    load_assets_from_modules,
)

from ferc1_eia_match import candidate_set_creation, config, inputs, resources

default_assets = (
    *load_assets_from_modules([inputs], group_name="inputs"),
    *load_assets_from_modules(
        [candidate_set_creation], group_name="candidate_set_creation"
    ),
)

pkg_source = importlib.resources.files("ferc1_eia_match.package_data").joinpath(
    "blocking_config.json"
)
with importlib.resources.as_file(pkg_source) as json_file:
    blocking_config = config.Model.from_json(json_file)


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
                    "embed_dataframes": blocking_config.embedding,
                },
            ),
            description="Job to generate input assets.",
        )
    ],
)
