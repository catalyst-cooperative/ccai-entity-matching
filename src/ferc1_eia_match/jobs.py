"""Dagster definitions for the entity matching steps."""

from dagster import (
    Definitions,
    EnvVar,
    RunConfig,
    define_asset_job,
    load_assets_from_modules,
)

from ferc1_eia_match import inputs, resources

default_assets = (*load_assets_from_modules([inputs], group_name="inputs"),)


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
                    "get_eia_input": inputs.EiaInputConfig(
                        update=False, eia_plant_part=None
                    )
                },
            ),
            description="Job to generate input assets.",
        )
    ],
)
