"""Helper functions for matching the FERC and EIA datasets."""

import pandas as pd


def drop_null_cols(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Drop columns with a specified percentage of null values.

    Args:
        df: The dataframe to drop null columns from.
        threshold: If the percentage of nulls in a column is greater
            than or equal to ``threshold`` then the column will be dropped.
            Default is 80%.
    """
    percent_null = df.isnull().sum() / len(df)
    cols_to_drop = list(set(percent_null[percent_null >= threshold].index))
    df = df.drop(columns=cols_to_drop)
    return df
