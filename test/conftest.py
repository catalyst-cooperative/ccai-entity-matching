"""PyTest configuration module. Defines useful fixtures, command line args."""
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

import pudl
from pudl.output.pudltabl import PudlTabl

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add package-specific command line options to pytest.

    This is slightly magical -- pytest has a hook that will run this function
    automatically, adding any options defined here to the internal pytest options that
    already exist.
    """
    parser.addoption(
        "--sandbox",
        action="store_true",
        default=False,
        help="Flag to indicate that the tests should use a sandbox.",
    )


@pytest.fixture(scope="session")
def test_dir() -> Path:
    """Return the path to the top-level directory containing the tests.

    This might be useful if there's test data stored under the tests directory that
    you need to be able to access from elsewhere within the tests.

    Mostly this is meant as an example of a fixture.
    """
    return Path(__file__).parent


@pytest.fixture(scope="session")
def pudl_input_dir() -> dict[Any, Any]:
    """Determine where the PUDL input/output dirs should be."""
    input_override = None

    # In CI we want a hard-coded path for input caching purposes:
    if os.environ.get("GITHUB_ACTIONS", False):
        # hard-code input dir for CI caching
        input_override = Path(os.environ["HOME"]) / "pudl-work/data"

    return {"input_dir": input_override}


@pytest.fixture(scope="session")
def pudl_settings_fixture(request, pudl_input_dir) -> Any:  # type: ignore
    """Determine some settings for the test session.

    * On a user machine, it should use their existing PUDL_DIR.
    * In CI, it should use PUDL_DIR=$HOME/pudl-work containing the
      downloaded PUDL DB.
    """
    logger.info("setting up the pudl_settings_fixture")
    pudl_settings = pudl.workspace.setup.get_defaults(**pudl_input_dir)
    pudl.workspace.setup.init(pudl_settings)

    pudl_settings["sandbox"] = request.config.getoption("--sandbox")

    pretty_settings = json.dumps(
        {str(k): str(v) for k, v in pudl_settings.items()}, indent=2
    )
    logger.info(f"pudl_settings being used: {pretty_settings}")
    return pudl_settings


@pytest.fixture(scope="session", name="pudl_engine")
def pudl_engine_fixture(pudl_settings_fixture: dict[Any, Any]) -> sa.engine.Engine:
    """Grab a connection to the PUDL Database.

    If we are using the test database, we initialize the PUDL DB from scratch.
    If we're using the live database, then we just make a conneciton to it.
    """
    logger.info("setting up the pudl_engine fixture")
    engine = sa.create_engine(pudl_settings_fixture["pudl_db"])
    logger.info("PUDL Engine: %s", engine)
    return engine


@pytest.fixture(scope="session", name="pudl_out")
def pudl_out_fixture(pudl_engine: sa.engine.Engine) -> PudlTabl:
    """Define PudlTabl output object fixture."""
    return PudlTabl(pudl_engine=pudl_engine, freq="AS")
