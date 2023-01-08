"""Prepare the FERC and EIA input data for matching according to the needs of the specified entity resolution method."""

from typing import List, Literal

import pandas as pd

import pudl


class InputManager:
    """Class to prepare FERC1 and EIA data for matching."""

    def __init__(
        self,
        pudl_out: pudl.output.pudltabl.PudlTabl,
        matching_tool: Literal["panda", "splink"] = "panda",
        report_years: List[int] | None = None,
        eia_plant_part: str | None = None,
    ) -> None:
        """Initialize class that gets FERC 1 input for matching with EIA.

        Args:
            pudl_out: PUDL output object to retrieve FERC 1 data.
            matching_tool: The tool that will be used for matching. The idea is
                to use this parameter for specific tool dependent filtering/cleaning
                that needs to happen. Maybe it won't be needed.
            report_years: A list of the report years to filter the data to. Year values
                should be integers. None by default, and all years of data will be included.
            eia_plant_part: The plant part to filter the EIA data by. None by default,
                and all plant parts will be included in the EIA data.
        """
        self.pudl_out = pudl_out
        # TODO: need matching_tool?
        self.matching_tool = matching_tool
        self.report_years = report_years
        if (
            eia_plant_part is not None
            and eia_plant_part not in pudl.analysis.plant_parts_eia.PLANT_PARTS
        ):
            raise AssertionError(f"{eia_plant_part} is not a valid EIA plant part.")
        else:
            self.eia_plant_part = eia_plant_part

    def get_ferc_input(self) -> pd.DataFrame:
        """Get the FERC 1 plants data from PUDL and prepare for matching.

        Function adapted from RMI and Catalyst EIA and FERC repo.
        Merge FERC 1 fuel usage by plant attributes onto all FERC 1 plant records,
        add potentially helpful comparison attributes like `installation_year`,
        rename columns to match EIA side, set the index to `record_id_ferc1`,
        and filter to desired report years.
        """
        fbp_cols_to_use = [
            "report_year",
            "utility_id_ferc1",
            "plant_name_ferc1",
            "utility_id_pudl",
            "fuel_cost",
            "fuel_mmbtu",
            "primary_fuel_by_mmbtu",
        ]
        plants_ferc1_df = (
            self.pudl_out.plants_all_ferc1()
            .merge(
                self.pudl_out.fbp_ferc1()[fbp_cols_to_use],
                on=[
                    "report_year",
                    "utility_id_ferc1",
                    "utility_id_pudl",
                    "plant_name_ferc1",
                ],
                how="left",
            )
            .pipe(pudl.helpers.convert_cols_dtypes, "ferc1")
            .assign(
                installation_year=lambda x: (
                    x.installation_year.astype("float")
                ),  # need for comparison vectors
                fuel_cost_per_mmbtu=lambda x: (x.fuel_cost / x.fuel_mmbtu),
                heat_rate_mmbtu_mwh=lambda x: (x.fuel_mmbtu / x.net_generation_mwh),
            )
            .rename(
                columns={
                    "record_id": "record_id_ferc1",
                    "opex_plants": "opex_plant",
                    "fuel_cost": "total_fuel_cost",
                    "fuel_mmbtu": "total_mmbtu",
                    "opex_fuel_per_mwh": "fuel_cost_per_mwh",
                    "primary_fuel_by_mmbtu": "fuel_type_code_pudl",
                    "plant_name_ferc1": "plant_name",  # rename so column names match EIA side
                    "utility_name_ferc1": "utility_name",
                }
            )
            .set_index("record_id_ferc1")
        )
        plants_ferc1_df = plants_ferc1_df[
            plants_ferc1_df.report_year.isin(self.report_years)
        ]
        return plants_ferc1_df

    def get_eia_input(self) -> pd.DataFrame:
        """Get the distinct EIA plant parts list from PUDL and prepare for matching.

        The distinct plant parts list includes only the true granularities of plant part
        and non duplicate ownership. See the `pudl.analysis.plant_parts_eia` module
        for more explanation.
        Make the EIA plant parts distinct, filter by report year and plant part,
        add on utlity name.
        """
        plant_parts_eia = self.pudl_out.plant_parts_eia()
        # a little patch, this might not be needed anymore
        plant_parts_eia = plant_parts_eia[
            ~plant_parts_eia.index.duplicated(keep="first")
        ]
        # make plant_parts_eia distinct
        plant_parts_eia = plant_parts_eia[
            (plant_parts_eia["true_gran"]) & (~plant_parts_eia["ownership_dupe"])
        ]
        # filter by plant part and report years
        plant_parts_eia = plant_parts_eia[
            plant_parts_eia.report_year.isin(self.report_years)
        ]
        plant_parts_eia = plant_parts_eia[
            plant_parts_eia.plant_part == self.eia_plant_part
        ]
        # add on utility name
        # are these tables the same?
        # eia_util = pd.read_sql("utilities_eia", pudl_engine)
        eia_util = self.pudl_out.utils_eia860()
        eia_util = eia_util.set_index("utility_id_eia")["utility_name_eia"]
        non_null_df = plant_parts_eia[~(plant_parts_eia.utility_id_eia.isnull())]
        non_null_df = non_null_df.merge(
            eia_util,
            how="left",
            left_on="utility_id_eia",
            right_index=True,
            validate="m:1",
        )
        plant_parts_eia = pd.concat(
            [non_null_df, plant_parts_eia[plant_parts_eia.utility_id_eia.isnull()]]
        ).reindex(plant_parts_eia.index)

        return plant_parts_eia
