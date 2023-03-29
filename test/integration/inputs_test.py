"""Test that the EIA and FERC data inputs can be generated."""
from typing import Any

import pytest
import sqlalchemy as sa

from ferc_eia_match.inputs import InputManager
from pudl.output.pudltabl import PudlTabl


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
def test_pudl_engine(pudl_engine: dict[Any, Any], table_name: str) -> None:
    """Test that the PUDL DB is actually available."""
    insp = sa.inspect(pudl_engine)
    if table_name not in insp.get_table_names():
        raise AssertionError(f"{table_name} not in PUDL DB.")


def test_ferc_input(pudl_out: PudlTabl) -> None:
    """Test that the FERC input data can be created."""
    ferc_df = InputManager(pudl_out=pudl_out, report_years=[2020]).get_ferc_input()
    if ferc_df.empty:
        raise AssertionError("FERC input dataframe is empty.")


def test_eia_input(pudl_out: PudlTabl) -> None:
    """Test that the EIA input data can be created."""
    eia_df = InputManager(pudl_out=pudl_out, eport_years=[2020]).get_eia_input()
    if eia_df.empty:
        raise AssertionError("EIA input dataframe is empty.")
