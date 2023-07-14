"""Blocking methods to create a candidate set of tuples matches."""
import logging

import faiss
import numpy as np
import pandas as pd
from dagster import AssetOut, multi_asset
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

from ferc1_eia_match.config import EmbeddingConfig

logger = logging.getLogger(__name__)


def tfidf_vectorize(
    left_df, right_df, column_name: str, kwargs: dict = {}
) -> tuple[np.ndarray, np.ndarray]:
    """Column embedding function: Use TF-IDF to create vector embeddings for a column.

    Arguments:
        left_df: The left dataframe to be embedded. If no ``right_df`` is given, then this
            is the only dataframe that will be embedded. The index should be the default,
            numeric index.
        right_df: The right dataframe to be embedded. If it's an empty dataframe, then
            only ``left_df`` will be embedded. The index should be the default, numeric
            index.
        column_name: Name of the column in the dataframe to vectorize.
        kwargs: Keyword arguments for the sklearn TfidfVectorizer object.
    """
    # fill nulls with the empty string so they become 0 vectors
    logger.info(f"Converting {column_name} to TF-IDF features.")
    if column_name in left_df.columns:
        left_series = left_df[column_name].fillna("")
    else:
        raise AssertionError(
            f"{column_name} is not in left dataframe columns. Can't vectorize."
        )
    if right_df.empty:
        right_series = pd.Series()
    elif column_name in right_df.columns:
        right_series = right_df[column_name].fillna("")
    else:
        raise AssertionError(
            f"{column_name} is not in right dataframe columns. Can't vectorize."
        )
    vectorizer = TfidfVectorizer(**kwargs)
    vectorizer.fit(pd.concat([left_series, right_series]))
    return vectorizer.transform(left_series), vectorizer.transform(right_series)


def min_max_scale(
    left_df, right_df, column_name: str, feature_range: tuple = (0, 1)
) -> tuple[np.ndarray, np.ndarray]:
    """Column embedding function: Scale numeric column between a range.

    Arguments:
        left_df: The left dataframe to be embedded. If no ``right_df`` is given, then this
            is the only dataframe that will be embedded. The index should be the default,
            numeric index.
        right_df: The right dataframe to be embedded. If it's an empty dataframe, then
            only ``left_df`` will be embedded. The index should be the default, numeric
            index.
        column_name: the name of the column in the left and right dataframes
            to scale to the feature range.
        feature_range: The desired range of the transformed data.
    """
    # fill nulls with the median
    logger.info(f"Scaling {column_name} with MinMaxScaler.")
    if column_name in left_df.columns:
        left_series = left_df[column_name]
    else:
        raise AssertionError(
            f"{column_name} is not in left dataframe columns. Can't vectorize."
        )
    if right_df.empty:
        right_series = pd.Series()
    elif column_name in right_df.columns:
        right_series = right_df[column_name]
    else:
        raise AssertionError(
            f"{column_name} is not in right dataframe columns. Can't vectorize."
        )
    full_series = pd.concat([left_series, right_series])
    med = full_series.median()
    full_series = full_series.fillna(med)
    if not pd.to_numeric(full_series, errors="coerce").notnull().all():
        raise AssertionError(
            f"There are non-numeric values in the {column_name} column."
        )
    full_array = full_series.to_numpy().reshape(-1, 1)
    left_array = left_series.fillna(med).to_numpy().reshape(-1, 1)
    right_array = right_series.fillna(med).to_numpy().reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=feature_range)
    scaler.fit(full_array)
    return scaler.transform(left_array), scaler.transform(right_array)


def equal_weight_aggregate(embedded_attribute_dict: dict):
    """Equally weight and combine attribute embedding vectors into tuple embeddings.

    All matrices in ``embedded_attribute_dict`` must have the same number of rows.

    Arguments:
        embedded_attribute_dict: Dictionary where the keys are attribute names and the
            values are the matrix embeddings for the attributes that are to be concatenated
            together
    Returns:
        Matrix of the attribute matrices concatenated together horizontally.
    """
    tuple_embedding = hstack(list(embedded_attribute_dict.values()))
    tuple_embedding = tuple_embedding.tocsr()
    return tuple_embedding


def set_col_blocks(
    left_df: pd.DataFrame, right_df: pd.DataFrame, blocking_col: str
) -> tuple[dict, dict]:
    """Get the indices of blocks of records in a dataframe for a blocking column.

    Get blocks of records in a dataframe where the value of the ``blocking_col``
    column is equal within all records in a block. Update ``left_blocks_dict`` and
    ``right_blocks_dict`` to be a dictionary with keys as each unique value in
    the ``blocking_col`` column and values as the index of the rows with that value.
    This is the result of ``df.groupby(blocking_col).groups``. Update ``blocking_col``

    Arguments:
        left_df: The left dataframe to be embedded. If no ``right_df`` is given, then this
            is the only dataframe that will be embedded. The index should be the default,
            numeric index.
        right_df: The right dataframe to be embedded. If it's an empty dataframe, then
            only ``left_df`` will be embedded. The index should be the default, numeric
            index.
        blocking_col: The name of the column to block on.
    """
    if blocking_col in left_df.columns:
        left_blocks_dict = left_df.groupby(blocking_col).groups
    if blocking_col in right_df.columns:
        right_blocks_dict = right_df.groupby(blocking_col).groups

    return left_blocks_dict, right_blocks_dict


@multi_asset(
    outs={
        "ferc_embedded": AssetOut(),
        "eia_embedded": AssetOut(),
    }
)
def embed_dataframes(
    config: EmbeddingConfig,
    ferc_input: pd.DataFrame,
    eia_input: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Embed left (and right) dataframes and create blocks from a column.

    Set `left_embedding_matrix` and `right_embedding_matrix` with
    matrix embeddings for `left_df` and `right_df`. Embed attributes
    based on the functions for each column in `col_embedding_dict`.
    Concatenate the embeddings for each column together into one embedding
    matrix for each dataframe. Optionally set `left_blocks_dict` and
    ``right_blocks_dict`` to be the indices of blocks within ``blocking_col``.

    Arguments:
        config: Dagster configuration options.
        ferc_input: FERC derived input dataframe.
        eia_input: EIA derived input dataframe.
    """
    left_df = ferc_input[config.matching_cols].reset_index()
    right_df = eia_input[config.matching_cols].reset_index()

    left_embedding_attribute_dict = {}
    right_embedding_attribute_dict = {}
    for column_name, embedding_config in config.embedding_map.items():
        options = embedding_config.options
        if not options:
            options = {}
        match embedding_config.embedding_type:
            case "tfidf_vectorize":
                (
                    left_embedding_attribute_dict[column_name],
                    right_embedding_attribute_dict[column_name],
                ) = tfidf_vectorize(left_df, right_df, column_name, **options)
            case "min_max_scale":
                (
                    left_embedding_attribute_dict[column_name],
                    right_embedding_attribute_dict[column_name],
                ) = min_max_scale(left_df, right_df, column_name, **options)

    # TODO: allow for other types of aggregation
    left_embedding_matrix = equal_weight_aggregate(left_embedding_attribute_dict)
    right_embedding_matrix = equal_weight_aggregate(right_embedding_attribute_dict)

    logger.info(f"left_embedding_size: {left_embedding_matrix.shape}")
    logger.info(f"right_embedding_size: {right_embedding_matrix.shape}")

    return pd.DataFrame(left_embedding_matrix), pd.DataFrame(right_embedding_matrix)


class SimilaritySearcher:
    """Conduct a search to select a candidate set of likely matched tuples.

    Some methods perform an exact search over all vectors in the index while
    other searches perform an approximate nearest neighbors search, sacrificing
    some precision for increased speed.

    FAISS indexes: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """

    search_functions = ["l2_distance_search", "cosine_siilarity_search"]

    def __init__(
        self,
        query_embedding_matrix,
        menu_embedding_matrix,
        query_blocks_dict,
        menu_blocks_dict,
    ):
        """Initialize a similarity search object to create a candidate set of tuple pairs.

        If a sparce matrix is passed, it will be made dense.

        Arguments:
            query_embedding_matrix: The embedding matrix representing the query vectors, or
                vectors for which to search for the k most similar matches from the
                ``menu_embedding_matrix`` i.e. "for each query vector, find the top k most
                similar vectors from the index of potential matches".
            menu_embedding_matrix: The embedding matrix representing the potential match vectors
                for the query vectors. These vector embeddings which will be searched over to find
                vectors most similar or closest to the query vectors.
            query_blocks_dict: Dictionary where keys are the name of blocks of records and values
                are the index in the ``query_embedding_matrix`` for the records in that block.
                Search will be performed over records within the same block. If no blocking is done,
                then the only entry in the dictionary is the full index of the query matrix.
            menu_blocks_dict: Dictionary where keys are the name of blocks of records and values
                are the index in the ``menu_embedding_matrix`` for the records in that block.
                Search will be performed over records within the same block. If no blocking is done,
                then the only entry in the dictionary is the full index of the menu matrix.
        """
        if issparse(query_embedding_matrix):
            query_embedding_matrix = query_embedding_matrix.todense()
        if issparse(menu_embedding_matrix):
            menu_embedding_matrix = menu_embedding_matrix.todense()
        self.query_embedding_matrix = np.float32(query_embedding_matrix)
        self.menu_embedding_matrix = np.float32(menu_embedding_matrix)
        self.query_blocks_dict = query_blocks_dict
        self.menu_blocks_dict = menu_blocks_dict
        self.menu_dim = menu_embedding_matrix.shape[1]

    def l2_distance_search(self, queries_idx: np.ndarray, menu_idx: np.ndarray, k: int):
        """Conduct an exact search for smallest L2 distance between vectors.

        Get the k vectors from the menu matrix that minimizes the L2 distance to query
        vectors. The ``IndexIDMap`` maps the result of the search back to the record's
        original index in the ``menu_embedding_matrix``.

        Arguments:
            k (int): The number of potential record matches to find for each query vector.
            queries_idx: The index of the current block of query vectors in
                ``self.query_embedding_matrix``.
            menu_idx: The index of the current block of menu vectors in
                ``self.menu_embedding_matrix``.

        Returns:
            Numpy array of shape ``(len(query_embeddings), k)`` where each row represents
            the k indices of the index embeddings matrix that are closest to each query
            vector.
        """
        menu = self.menu_embedding_matrix[menu_idx]
        queries = self.query_embedding_matrix[queries_idx]
        index_l2 = faiss.IndexFlatL2(self.menu_dim)
        index = faiss.IndexIDMap(index_l2)
        index.add_with_ids(menu, menu_idx)
        distances, match_indices = index.search(queries, k)

        return match_indices

    def cosine_similarity_search(
        self, queries_idx: np.ndarray, menu_idx: np.ndarray, k: int
    ):
        """Conduct an exact search for highest cosine similarity between vectors.

        Normalize matrices and take the inner product, equivalent to cosine similarity.
        The ``IndexIDMap`` maps the result of the search back to the record's original
        index in the ``menu_embedding_matrix``.

        Arguments:
            k: The number of potential record matches to find for each query vector.
            queries_idx: The index of the current block of query vectors in
                ``self.query_embedding_matrix``.
            menu_idx: The index of the current block of menu vectors in
                ``self.menu_embedding_matrix``.

        Returns:
            Numpy array of shape ``(len(query_embeddings), k)`` where each row represents
            the k indices of the menu embeddings matrix that are closest to each query
            vector.
        """
        menu = self.menu_embedding_matrix[menu_idx]
        queries = self.query_embedding_matrix[queries_idx]
        faiss.normalize_L2(menu)
        faiss.normalize_L2(queries)
        # use the Inner Product Index, which is equivalent to cosine sim for normalized vectors
        index_ip = faiss.IndexFlatIP(self.menu_dim)
        index = faiss.IndexIDMap(index_ip)
        index.add_with_ids(menu, menu_idx)
        distances, match_indices = index.search(queries, k)

        return match_indices

    def get_search_function_names(self) -> list[str]:
        """Get the names of available search functions."""
        return self.search_functions

    def run_candidate_pair_search(
        self, k: int = 5, search_name: str = "l2_distance_search"
    ) -> np.ndarray:
        """Search for the k best candidate pairs from a left and right dataframe.

        Conducts a search on each block of records to get the k most similar/closest
        menu vectors for each query vector. Concatenates results from each block
        together to get a full candidate set.

        Arguments:
            k: The number of candidate pair records to select for each query vector.
                Default is 5.
            search_name: The name of the search function to use. Valid values are given
                by ``SimilaritySearcher.get_search_function_names``.

        Returns:
            A candidate set of length ``(self.query_embedding_matrix.shape[0], k)``
            where each row is a list of the indices of the k best menu vectors for the
            query vector at that index.
        """
        # a (currently empty) array of the k best right matches for each left record
        candidate_set = np.empty((self.query_embedding_matrix.shape[0], k))
        for block_key in self.query_blocks_dict:
            print(f"Conducting search for candidate pairs on the {block_key} block")
            left_idx = self.query_blocks_dict[block_key].to_numpy()
            right_idx = self.menu_blocks_dict[block_key].to_numpy()
            if hasattr(self, search_name):
                search_func = getattr(self, search_name)
                block_candidate_set = search_func(
                    queries_idx=left_idx, menu_idx=right_idx, k=k
                )
            else:
                raise ValueError(f"Search function {search_name} does not exist.")
            candidate_set[left_idx] = block_candidate_set
        return candidate_set
