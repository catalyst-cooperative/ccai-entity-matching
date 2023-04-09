"""Test that the EIA and FERC data inputs can be generated."""
import pytest
import sqlalchemy as sa

from ferc1_eia_match.inputs import InputManager


@pytest.mark.parametrize(
    "table_name",
    [
        "fuel_ferc1",
        "ownership_eia860",
        "plants_entity_eia",
        "fuel_receipts_costs_eia923",
        "utilities_pudl",
    ],
)
def test_pudl_engine(pudl_engine: sa.engine.Engine, table_name: str) -> None:
    """Test that the PUDL DB is actually available."""
    insp = sa.inspect(pudl_engine)
    if table_name not in insp.get_table_names():
        raise AssertionError(f"{table_name} not in PUDL DB.")


def test_ferc_input(pudl_engine: sa.engine.Engine) -> None:
    """Test that the FERC input data can be created."""
    ferc_df = InputManager(
        pudl_engine=pudl_engine, start_report_year=2020, end_report_year=2020
    ).get_ferc_input()
    if ferc_df.empty:
        raise AssertionError("FERC input dataframe is empty.")


def test_eia_input(pudl_engine: sa.engine.Engine) -> None:
    """Test that the EIA input data can be created."""
    eia_df = InputManager(
        pudl_engine=pudl_engine, start_report_year=2020, end_report_year=2020
    ).get_eia_input()
    if eia_df.empty:
        raise AssertionError("EIA input dataframe is empty.")
