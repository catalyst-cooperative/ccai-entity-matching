"""Script to execute experiments and save metrics."""
import argparse
import sys
from pathlib import Path

import sqlalchemy as sa

import pudl
from ferc1_eia_match.metrics import blocking

# Default dir is the base of repo
MLRUNS_DEFAULT = Path(__file__).parent.parent.parent.parent / "mlruns/"


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

    return parser.parse_args()


def main(experiments: list[str], mlrun_uri: str, db: str):
    """Execute experiments."""
    pudl_engine = sa.create_engine(db)

    for experiment in experiments:
        match experiment:
            case "blocking":
                blocking.execute_blocking(pudl_engine, Path(mlrun_uri))


if __name__ == "__main__":
    sys.exit(main(**vars(_parse_args())))
