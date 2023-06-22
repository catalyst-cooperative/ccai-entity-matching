"""Define config resources."""

import sqlalchemy as sa
from dagster import ConfigurableResource

from pudl.output.pudltabl import PudlTabl


class PudlInputs(ConfigurableResource):
    """Resource to manage inputs from PUDL."""

    start_year: int | None
    end_year: int | None
    pudl_db_path: str

    def get_inputs(self) -> tuple[PudlTabl, sa.engine.Engine]:
        """Dagster resource producing PudlTabl object."""
        # TODO: Switch to using PUDL assets directly
        pudl_engine = sa.create_engine(f"sqlite:///{self.pudl_db_path}/pudl.sqlite")

        start_year = self.start_year
        end_year = self.end_year
        start_date = None
        end_date = None
        if start_year is not None:
            start_date = str(start_year) + "-01-01"
        if end_year is not None:
            end_date = str(end_year) + "-12-31"

        return (
            PudlTabl(
                pudl_engine,
                start_date=start_date,
                end_date=end_date,
                freq="AS",
            ),
            pudl_engine,
        )
