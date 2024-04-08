"""Script to execute experiments and save metrics."""

import argparse
import logging
import sys
from pathlib import Path

import sqlalchemy as sa

import pudl
from ferc1_eia_match.metrics import blocking

# Default dir is the base of repo
MLRUNS_DEFAULT = Path(__file__).parent.parent.parent.parent / "mlruns/"
logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser(description="Log experiment metrics.")
    parser.add_argument(
        "--experiments",
        type=list[str],
        nargs="+",
        help="Experiments to run.",
        default=["blocking"],
    )

    parser.add_argument(
        "--mlrun_uri",
        type=str,
        help="Path to MLflow tracking backend location.",
        default=MLRUNS_DEFAULT,
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Path to pudl.sqlite DB.",
        default=pudl.workspace.setup.get_defaults()["pudl_db"],
    )
    parser.add_argument(
        "--model-config",
        type=str,
        help="Path to JSON file containing model config",
        default=None,
    )

    return parser.parse_args()


def main():
    """Execute experiments."""
    args = _parse_args()
    pudl_engine = sa.create_engine(args.db)

    for experiment in args.experiments:
        match experiment:
            case "blocking":
                blocking.run_blocking_tests(pudl_engine, Path(args.mlrun_uri))


if __name__ == "__main__":
    sys.exit(main())
