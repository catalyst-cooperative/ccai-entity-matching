"""Prepare the FERC and EIA input data for matching according to the needs of the specified entity resolution method."""

import logging

import pandas as pd
import sqlalchemy as sa

import pudl
from ferc1_eia_match.name_cleaner import CompanyNameCleaner

logger = logging.getLogger(__name__)

TRAIN_CSV_FILENAME = "ferc1_eia_train.csv"


class InputManager:
    """Class to prepare FERC1 and EIA data for matching."""

    def __init__(
        self,
        pudl_engine: sa.engine.Engine,
        start_report_year: int | None = None,
        end_report_year: int | None = None,
        eia_plant_part: str | None = None,
    ) -> None:
        """Initialize class that gets FERC 1 input for matching with EIA.

        Args:
            pudl_engine: A connection to the PUDL DB.
            start_report_year: The first year of the report years that the data is filtered by.
                Year value should be an integer. None by default, and all years of data before
                ``end_report_year`` will be included.
            end_report_year: The last year of the report years that the data is filtered by.
                Year value should be an integer. None by default, and all years of data after
                ``start_report_year`` will be included.
            eia_plant_part: The plant part to filter the EIA data by. None by default,
                and all plant parts will be included in the EIA data.
        """
        self.pudl_engine = pudl_engine
        self.start_date = None
        self.end_date = None
        if start_report_year is not None:
            self.start_date = str(start_report_year) + "-01-01"
        if end_report_year is not None:
            self.end_date = str(end_report_year) + "-12-31"
        self.pudl_out = pudl.output.pudltabl.PudlTabl(
            pudl_engine,
            start_date=self.start_date,
            end_date=self.end_date,
            freq="AS",
        )
        # use default rules for utility_name
        self.utility_name_cleaner = CompanyNameCleaner()
        # default rules except keep words in parentheses
        self.plant_name_cleaner = CompanyNameCleaner(
            cleaning_rules_list=[
                "replace_amperstand_between_space_by_AND",
                "replace_hyphen_between_spaces_by_single_space",
                "replace_underscore_by_space",
                "replace_underscore_between_spaces_by_single_space",
                "remove_text_puctuation_except_dot",
                "remove_math_symbols",
                "add_space_before_opening_parentheses",
                "add_space_after_closing_parentheses",
                "remove_parentheses",
                "remove_brackets",
                "remove_curly_brackets",
                "enforce_single_space_between_words",
            ]
        )
        self.eia_df = None
        self.ferc1_df = None
        self.non_distinct_plant_parts_eia = None
        if (
            eia_plant_part is not None
            and eia_plant_part not in pudl.analysis.plant_parts_eia.PLANT_PARTS
        ):
            raise AssertionError(f"{eia_plant_part} is not a valid EIA plant part.")
        else:
            self.eia_plant_part = eia_plant_part

    def _extract_keyword_from_column(
        self, ser: pd.Series, keyword_list: list[str]
    ) -> pd.Series:
        """Extract keywords contained in a Pandas series with a regular expression."""
        pattern = r"(?:^|\s+)(" + "|".join(keyword_list) + r")(?:\s+|$)"
        return ser.str.extract(pattern, expand=False)

    def fill_fuel_type_from_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing fuel_type_code_pudl data from plant name.

        If a missing fuel type code is contained in the plant name, fill in the fuel
        type code PUDL for that record. E.g. "Washington Hydro"
        """
        if "fuel_type_code_pudl" not in df.columns:
            raise AssertionError("fuel_type_code_pudl is not in dataframe columns.")
        with self.pudl_engine.connect() as conn:
            ftcp = conn.execute(
                "SELECT DISTINCT fuel_type_code_pudl FROM energy_sources_eia"
            )
            fuel_type_list = [fuel[0] for fuel in ftcp]
        fuel_type_map = {fuel_type: fuel_type for fuel_type in fuel_type_list}
        fuel_type_map.update(
            {
                "pumped storage": "hydro",
                "peaker": "gas",
                "gt": "gas",
                "peaking": "gas",
                "river": "hydro",
                "falls": "hydro",
            }
        )
        # grab fuel type keywords that are within plant_name and fill in null FTCP
        df["fuel_type_code_pudl"] = df["fuel_type_code_pudl"].fillna(
            self._extract_keyword_from_column(
                df["plant_name"], list(fuel_type_map.keys())
            ).map(fuel_type_map)
        )
        return df

    def match_installation_construction_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Function to impute missing installation or construction yaers.

        It's likely better to fill an installation/construction year to be equal instead
        of imputing with an average. If one of installation year or construction year is null,
        fill with the other.
        """
        df.fillna(
            {
                "installaton_year": df["construction_year"],
                "construction_year": df["installation_year"],
            },
            inplace=True,
        )
        return df

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
                }
            )
            .astype(
                {
                    "plant_name_ferc1": "string",
                    "utility_name_ferc1": "string",
                    "fuel_type_code_pudl": "string",
                    "installation_year": "Int64",
                    "construction_year": "Int64",
                    "capacity_mw": "float64",
                }
            )
            .set_index("record_id_ferc1")
        )
        # nullify negative capacity and round values
        plants_ferc1_df.loc[plants_ferc1_df.capacity_mw <= 0, "capacity_mw"] = None
        plants_ferc1_df = plants_ferc1_df.round({"capacity_mw": 2})
        plants_ferc1_df = (
            plants_ferc1_df.pipe(
                self.utility_name_cleaner.get_clean_df,
                "utility_name_ferc1",
                "utility_name",
            )
            .pipe(
                self.plant_name_cleaner.get_clean_df, "plant_name_ferc1", "plant_name"
            )
            .pipe(self.fill_fuel_type_from_name)
            .pipe(self.match_installation_construction_year)
        )

        self.ferc1_df = plants_ferc1_df
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

        # TODO: don't save this variable once PPE is dagsterized
        # used for train connections
        self.non_distinct_plant_parts_eia = plant_parts_eia
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
        plant_parts_eia = plant_parts_eia.astype(
            {
                "plant_name_eia": "string",
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
        plant_parts_eia = (
            plant_parts_eia.pipe(
                self.utility_name_cleaner.get_clean_df,
                "utility_name_eia",
                "utility_name",
            )
            .pipe(self.plant_name_cleaner.get_clean_df, "plant_name_eia", "plant_name")
            .pipe(self.fill_fuel_type_from_name)
            .pipe(self.match_installation_construction_year)
        )
        self.eia_df = plant_parts_eia

        return plant_parts_eia

    def get_training_data(self, plant_parts_eia: pd.DataFrame | None = None):
        """Get FERC1 to EIA training data from PUDL analysis module.

        FERC1 records will be matched to their true granulity EIA match and training
        data will be restricted to the data range defined by ``start_report_year`` and
        ``end_report_year``.

        Arguments:
            plant_parts_eia: Optionally pass in the plant parts list (non distinct).
                Otherwise, it is generated in ``get_eia_input``.

        Returns:
            train_df: Training data connected at the true granularity with columns
                for ``record_id_eia`` and ``record_id_ferc1`` as well as a signature
                and notes for each match.
        """
        # TODO: when PPE is dagsterized just grab it from the DB
        # get training data connected at the true granularity
        if plant_parts_eia is not None:
            train_df = pudl.analysis.ferc1_eia.prep_train_connections(
                self.non_distinct_plant_parts_eia,
                start_date=self.start_date,
                end_date=self.end_date,
            ).reset_index()
        else:
            if self.non_distinct_plant_parts_eia is None:
                self.get_eia_input()
            train_df = pudl.analysis.ferc1_eia.prep_train_connections(
                self.non_distinct_plant_parts_eia,
                start_date=self.start_date,
                end_date=self.end_date,
            ).reset_index()

        train_df = pudl.analysis.ferc1_eia.restrict_train_connections_on_date_range(
            train_df,
            "record_id_eia",
            start_date=self.start_date,
            end_date=self.end_date,
        )
        return train_df
