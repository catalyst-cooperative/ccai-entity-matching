"""Prepare the FERC and EIA input data for matching according to the needs of the specified entity resolution method."""

import logging
from typing import List, Literal

import pandas as pd

import pudl
from ferc_eia_match.helpers import drop_null_cols
from ferc_eia_match.name_cleaner import CompanyNameCleaner

logger = logging.getLogger(__name__)


def splink_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare EIA and FERC data for matching with splink.

    Dataframe should have ``record_id_eia`` or ``record_id_ferc1`` as the index.
    """
    df = drop_null_cols(df, threshold=0.8)
    # need to make the record id index a column for splink to have a unique id col
    df["record_id"] = df.index
    # keep all cols for now
    """
    splink_cols = [
        "report_year",
        "plant_name",
        "utility_name",
        "installation_year",
        "construction_year",
        "capacity_mw",
        "fuel_type_code_pudl",
        "utility_id_pudl",
        "plant_id_pudl",
    ]
    df = df[splink_cols]
    """
    return df


# the preprocessing function to use for the EIA and FERC datasets for each linking tool
LINKER_PREPROCESS_FUNCS = {
    "splink": {"eia": splink_preprocess, "ferc": splink_preprocess}
}


class InputManager:
    """Class to prepare FERC1 and EIA data for matching."""

    def __init__(
        self,
        pudl_out: pudl.output.pudltabl.PudlTabl,
        linking_tool: Literal["panda", "splink"] = "panda",
        report_years: List[int] | None = None,
        eia_plant_part: str | None = None,
    ) -> None:
        """Initialize class that gets FERC 1 input for matching with EIA.

        Args:
            pudl_out: PUDL output object to retrieve FERC 1 data.
            linking_tool: The tool that will be used for matching. The idea is
                to use this parameter for specific tool dependent filtering/cleaning
                that needs to happen. Maybe it won't be needed.
            report_years: A list of the report years to filter the data to. Year values
                should be integers. None by default, and all years of data will be included.
            eia_plant_part: The plant part to filter the EIA data by. None by default,
                and all plant parts will be included in the EIA data.
        """
        self.pudl_out = pudl_out
        self.pudl_out.freq = "AS"
        # TODO: need linking_tool?
        self.linking_tool = linking_tool
        # company name string cleaner, currently uses default rules
        self.utility_cleaner = CompanyNameCleaner()
        self.report_years = report_years
        self.plant_parts_eia = None
        if self.report_years:
            self.pudl_out.start_date = pd.to_datetime(
                str(min(self.report_years)) + "-01-01"
            )
            self.pudl_out.end_date = pd.to_datetime(
                str(max(self.report_years)) + "-12-31"
            )
        if (
            eia_plant_part is not None
            and eia_plant_part not in pudl.analysis.plant_parts_eia.PLANT_PARTS
        ):
            raise AssertionError(f"{eia_plant_part} is not a valid EIA plant part.")
        else:
            self.eia_plant_part = eia_plant_part

    def get_ferc_input(self) -> pd.DataFrame:
        """Get the FERC 1 plants data from PUDL and prepare for matching.

        Merge FERC 1 fuel usage by plant attributes onto all FERC 1 plant records,
        add potentially helpful comparison attributes like `installation_year`,
        rename columns to match EIA side, set the index to `record_id_ferc1`,
        and filter to desired report years.
        """
        logger.info("Creating FERC plants input.")
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
                }
            )
            .astype(
                {
                    "plant_name": "string",
                    "utility_name_ferc1": "string",
                    "fuel_type_code_pudl": "string",
                    "installation_year": "Int64",
                    "construction_year": "Int64",
                    "capacity_mw": "float64",
                }
            )
            .set_index("record_id_ferc1")
        )
        plants_ferc1_df = plants_ferc1_df[
            plants_ferc1_df.report_year.isin(self.report_years)
        ]
        # nullify negative capacity and round values
        plants_ferc1_df.loc[plants_ferc1_df.capacity_mw <= 0, "capacity_mw"] = None
        plants_ferc1_df = plants_ferc1_df.round({"capacity_mw": 2})
        # basic string cleaning
        str_cols = ["utility_name_ferc1", "plant_name"]
        plants_ferc1_df[str_cols] = plants_ferc1_df[str_cols].apply(
            lambda x: x.str.strip().str.lower()
        )
        plants_ferc1_df = self.utility_cleaner.get_clean_df(
            plants_ferc1_df, "utility_name_ferc1", "utility_name"
        )

        # apply tool specific preprocessing
        if self.linking_tool in LINKER_PREPROCESS_FUNCS:
            logger.info(f"Applying preprocess function for {self.linking_tool}.")
            plants_ferc1_df = LINKER_PREPROCESS_FUNCS[self.linking_tool]["ferc"](
                plants_ferc1_df
            )

        return plants_ferc1_df

    def get_eia_input(self, update: bool = False) -> pd.DataFrame:
        """Get the distinct EIA plant parts list from PUDL and prepare for matching.

        The distinct plant parts list includes only the true granularities of plant part
        and non duplicate ownership. See the `pudl.analysis.plant_parts_eia` module
        for more explanation.
        Make the EIA plant parts distinct, filter by report year and plant part,
        add on utlity name.
        """
        logger.info("Creating the EIA plant parts list input.")
        plant_parts_eia = self.pudl_out.plant_parts_eia(
            update=update, update_gens_mega=update
        )
        # a little patch, this might not be needed anymore
        plant_parts_eia = plant_parts_eia[
            ~plant_parts_eia.index.duplicated(keep="first")
        ]
        # make plant_parts_eia distinct
        plant_parts_eia = plant_parts_eia[
            (plant_parts_eia["true_gran"]) & (~plant_parts_eia["ownership_dupe"])
        ]
        # filter by plant part
        if self.eia_plant_part:
            plant_parts_eia = plant_parts_eia[
                plant_parts_eia.plant_part == self.eia_plant_part
            ]
        # add on utility name
        # TODO: use entity table or glue table for utility names?
        eia_util = pd.read_sql("utilities_eia", self.pudl_out.pudl_engine)
        # eia_util = self.pudl_out.utils_eia860()
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
        # utility_name_eia will be renamed in company name cleaning step
        plant_parts_eia = plant_parts_eia.rename(
            columns={"plant_name_eia": "plant_name"}
        ).astype(
            {
                "plant_name": "string",
                "utility_name_eia": "string",
                "fuel_type_code_pudl": "string",
                "technology_description": "string",
                "installation_year": "Int64",
                "construction_year": "Int64",
                "capacity_mw": "float64",
            }
        )
        # nullify negative capacity and round values
        plant_parts_eia.loc[plant_parts_eia.capacity_mw <= 0, "capacity_mw"] = None
        plant_parts_eia = plant_parts_eia.round({"capacity_mw": 2})
        # basic string cleaning
        str_cols = ["utility_name_eia", "plant_name"]
        plant_parts_eia[str_cols] = plant_parts_eia[str_cols].apply(
            lambda x: x.str.strip().str.lower()
        )
        plant_parts_eia = self.utility_cleaner.get_clean_df(
            plant_parts_eia, "utility_name_eia", "utility_name"
        )
        if self.linking_tool in LINKER_PREPROCESS_FUNCS:
            logger.info(f"Applying preprocess function for {self.linking_tool}.")
            plant_parts_eia = LINKER_PREPROCESS_FUNCS[self.linking_tool]["eia"](
                plant_parts_eia
            )
        self.plant_parts_eia = plant_parts_eia

        return self.plant_parts_eia
