"""A dummy unit test so pytest has something to do."""
import logging

import pandas as pd

from ferc_eia_match.name_cleaner import CompanyNameCleaner

logger = logging.getLogger(__name__)


def test_company_name_cleaner() -> None:
    """Test the company name cleaner."""
    dirty_df = pd.DataFrame(
        {
            "utility_name_dirty": [
                "Duke Energy, LLC",
                "  Duke Energy Limited Liability Company ",
                "Duke Energy - [LLC]",
                "Duke Energy l.l.c. (NC)",
                "Pacific Gas & Electric Co.",
                "Southern   Edison",
                "Southern_Edison+",
                r"{Southern} Edison:",
                pd.NA,
                0,
            ]
        }
    )
    expected_df = pd.DataFrame(
        {
            "utility_name_dirty": [
                "Duke Energy, LLC",
                "  Duke Energy Limited Liability Company ",
                "Duke Energy - [LLC]",
                "Duke Energy l.l.c. (NC)",
                "Pacific Gas & Electric Co.",
                "Southern   Edison",
                "Southern_Edison+",
                r"{Southern} Edison:",
                pd.NA,
                0,
            ],
            "utility_name_clean": ["duke energy limited liability company"] * 4
            + [
                "pacific gas and electric company",
                "southern edison",
                "southern edison",
                "southern edison",
                pd.NA,
                pd.NA,
            ],
        }
    )

    clean_df = CompanyNameCleaner().get_clean_df(
        dirty_df, "utility_name_dirty", "utility_name_clean"
    )
    pd.testing.assert_frame_equal(clean_df, expected_df)
