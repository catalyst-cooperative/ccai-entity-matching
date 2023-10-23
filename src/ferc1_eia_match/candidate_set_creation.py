"""Blocking methods to create a candidate set of tuples matches."""
import logging

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from ferc1_eia_match.config import ColumnEmbedding

logger = logging.getLogger(__name__)


class DataframeEmbedder:
    """A class for creating an embedding matrix for one dataframe or a pair of dataframes."""

    # TODO: dynamically generate based on functions in superclass?
    column_embedding_functions = ["min_max_scale", "tfidf_vectorize", "one_hot_encode"]

    def __init__(
        self,
        left_df: pd.DataFrame,
        embedding_map: dict[str, ColumnEmbedding],
        # embedding_map: Type[EmbeddingConfig.embedding_map],
        right_df: pd.DataFrame = pd.DataFrame(),
    ):
        """Initialize a dataframe embedder object to vectorize one or two dataframe.

        If no ``right_df`` is given, then only ``left_df`` will be embedded. The left and right
        dataframe must share all the columns that are going to be vectorized. Column names of
        the embedded columns must be keys in ``embedding_map``.

        Arguments:
            left_df: The left dataframe to be embedded. If no ``right_df`` is given, then this
                is the only dataframe that will be embedded. The index should be the default,
                numeric index.
            embedding_map: A dictionary specifying the functions for embedding each column
                in the dataframe or dataframes and the keyword arguments for the function.
                All columns that are to be vectorized must be keys in the dictionary. Values are a
                dictionary where the key ``embedding_type`` specifies the embedding function name
                and ``options`` is a dictionary of keyword arguments for the function, if any.
                ``DataframeEmbedder.get_column_embedding_function_name`` gives valid function names.
            right_df: The right dataframe to be embedded. If it's an empty dataframe, then
                only ``left_df`` will be embedded. The index should be the default, numeric
                index.
        """
        self.left_df = left_df.copy()
        self.right_df = right_df.copy()
        self.embedding_map = embedding_map
        # self.col_embedding_dict = self._format_col_embedding_dict(col_embedding_dict)
        self.left_embedding_attribute_dict: dict[str, np.ndarray] = {}
        self.right_embedding_attribute_dict: dict[str, np.ndarray] = {}
        self.left_embedding_matrix: np.ndarray | None = None
        self.right_embedding_matrix: np.ndarray | None = None
        self.blocking_col: str = ""
        self.left_blocks_dict: dict[str, list] = {}
        self.right_blocks_dict: dict[str, list] = {}
        # TODO: incorporate dictionary for choosing how to handle nulls
        # self.null_handling_dict = null_handling_dict

    def _format_col_embedding_dict(self, col_embedding_dict) -> dict:
        """Format column embedding dictionary.

        Keys are the name of the column, values are a tuple with the name of the embedding
        function and a dictionary of keyword arguments for the embedding function. If there are
        no keyword arguments then the dictionary is empty.

        Arguments:
            col_embedding_dict: The dictionary to be formatted. Keys are function names and
                values are a list with the function name and optionally keyword arguments.
        """
        return {
            key: (func_name, {}) if len(kwargs) == 0 else (func_name, kwargs[0])
            for key, (func_name, *kwargs) in col_embedding_dict.items()
        }

    # TODO: make a superclass for these embedding functions
    def one_hot_encode(self, column_name: str, kwargs: dict = {}) -> None:
        """Column embedding function: One hot encode a column.

        Arguments:
            column_name: Name of the column in the dataframe to vectorize.
            kwargs: Keyword arguments for the sklearn TfidfVectorizer object.
        """
        logger.info(f"One hot encoding {column_name}.")
        # fill nulls with the empty string so they become 0 vectors
        if column_name in self.left_df.columns:
            left_series = self.left_df[column_name].fillna("")
        else:
            raise AssertionError(
                f"{column_name} is not in left dataframe columns. Can't vectorize."
            )
        if self.right_df.empty:
            right_series = pd.Series()
        elif column_name in self.right_df.columns:
            right_series = self.right_df[column_name].fillna("")
        else:
            raise AssertionError(
                f"{column_name} is not in right dataframe columns. Can't vectorize."
            )
        encoder = OneHotEncoder(**kwargs)
        encoder.fit(pd.concat([left_series, right_series]))
        self.left_embedding_attribute_dict[column_name] = encoder.transform(left_series)
        self.right_embedding_attribute_dict[column_name] = encoder.transform(
            right_series
        )

    def tfidf_vectorize(self, column_name: str, kwargs: dict = {}) -> None:
        """Column embedding function: Use TF-IDF to create vector embeddings for a column.

        Arguments:
            column_name: Name of the column in the dataframe to vectorize.
            kwargs: Keyword arguments for the sklearn TfidfVectorizer object.
        """
        # fill nulls with the empty string so they become 0 vectors
        logger.info(f"Converting {column_name} to TF-IDF features.")
        if column_name in self.left_df.columns:
            left_series = self.left_df[column_name].fillna("")
        else:
            raise AssertionError(
                f"{column_name} is not in left dataframe columns. Can't vectorize."
            )
        if self.right_df.empty:
            right_series = pd.Series()
        elif column_name in self.right_df.columns:
            right_series = self.right_df[column_name].fillna("")
        else:
            raise AssertionError(
                f"{column_name} is not in right dataframe columns. Can't vectorize."
            )
        vectorizer = TfidfVectorizer(**kwargs)
        vectorizer.fit(pd.concat([left_series, right_series]))
        self.left_embedding_attribute_dict[column_name] = vectorizer.transform(
            left_series
        )
        self.right_embedding_attribute_dict[column_name] = vectorizer.transform(
            right_series
        )

    def min_max_scale(self, column_name: str, feature_range: tuple = (0, 1)) -> None:
        """Column embedding function: Scale numeric column between a range.

        Arguments:
            column_name: the name of the column in the left and right dataframes
                to scale to the feature range.
            feature_range: The desired range of the transformed data.
        """
        # fill nulls with the median
        logger.info(f"Scaling {column_name} with MinMaxScaler.")
        if column_name in self.left_df.columns:
            left_series = self.left_df[column_name]
        else:
            raise AssertionError(
                f"{column_name} is not in left dataframe columns. Can't vectorize."
            )
        if self.right_df.empty:
            right_series = pd.Series()
        elif column_name in self.right_df.columns:
            right_series = self.right_df[column_name]
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
        self.left_embedding_attribute_dict[column_name] = scaler.transform(left_array)
        self.right_embedding_attribute_dict[column_name] = scaler.transform(right_array)

    def equal_weight_aggregate(self, embedded_attribute_dict: dict):
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

    def set_col_blocks(self, blocking_col: str) -> None:
        """Get the indices of blocks of records in a dataframe for a blocking column.

        Get blocks of records in a dataframe where the value of the ``blocking_col``
        column is equal within all records in a block. Update ``self.left_blocks_dict`` and
        ``self.right_blocks_dict`` to be a dictionary with keys as each unique value in
        the ``blocking_col`` column and values as the index of the rows with that value.
        This is the result of ``df.groupby(blocking_col).groups``. Update ``self.blocking_col``

        Arguments:
            blocking_col: The name of the column to block on.
        """
        self.blocking_col = blocking_col
        if blocking_col in self.left_df.columns:
            self.left_blocks_dict = self.left_df.groupby(blocking_col).groups
        if blocking_col in self.right_df.columns:
            self.right_blocks_dict = self.right_df.groupby(blocking_col).groups

    def get_column_embedding_function_names(self) -> list[str]:
        """Get the names of functions that can be used to vectorize a column."""
        return self.column_embedding_functions

    def embed_dataframes(self, blocking_col: str = "") -> None:
        """Embed left (and right) dataframes and create blocks from a column.

        Set `self.left_embedding_matrix` and `self.right_embedding_matrix` with
        matrix embeddings for `self.left_df` and `self.right_df`. Embed attributes
        based on the ``embedding_type`` functions for each column in `self.embedding_map`.
        Concatenate the embeddings for each column together into one embedding
        matrix for each dataframe. Optionally set `self.left_blocks_dict` and
        ``self.right_blocks_dict`` to be the indices of blocks within ``blocking_col``.

        Arguments:
            blocking_col: The name of the column to get blocks of indices of the
                dataframes. Default is the empty string if no manual blocking on
                the value of a column is to be performed. Passing a blocking column
                will not modify the dataframes themselves, just set the respective
                dictionary of blocks.
        """
        for column_name in self.embedding_map:
            vectorizer = self.embedding_map[column_name].embedding_type
            kwargs = self.embedding_map[column_name].options
            if hasattr(self, vectorizer):
                vectorizer_func = getattr(self, vectorizer)
                vectorizer_func(column_name=column_name, **kwargs)
            else:
                raise ValueError(
                    f"Vectorizing function {vectorizer} for {column_name} column does not exist."
                )
        # TODO: allow for other types of aggregation
        self.left_embedding_matrix = self.equal_weight_aggregate(
            self.left_embedding_attribute_dict
        )
        self.right_embedding_matrix = self.equal_weight_aggregate(
            self.right_embedding_attribute_dict
        )
        if blocking_col != "":
            self.set_col_blocks(blocking_col=blocking_col)
        else:
            # if there's no blocking col, then there's just one block with all records
            self.left_blocks_dict["all records"] = self.left_df.reset_index(
                drop=True
            ).index
            self.right_blocks_dict["all records"] = self.right_df.reset_index(
                drop=True
            ).index


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
            match indices: Numpy array of shape ``(len(query_embeddings), k)`` where each
                row represents the k indices of the menu embeddings matrix that are
                closest to each query vector.
            distances: Numpy array giving the distances from each query vector to
                corresponding menu vector in ``match_indices``.
        """
        menu = self.menu_embedding_matrix[menu_idx]
        queries = self.query_embedding_matrix[queries_idx]
        index_l2 = faiss.IndexFlatL2(self.menu_dim)
        index = faiss.IndexIDMap(index_l2)
        index.add_with_ids(menu, menu_idx)
        distances, match_indices = index.search(queries, k)

        return match_indices, distances

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
            match indices: Numpy array of shape ``(len(query_embeddings), k)`` where each
                row represents the k indices of the menu embeddings matrix that are
                closest to each query vector.
            distances: Numpy array giving the distances from each query vector to
                corresponding menu vector in ``match_indices``.
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

        return match_indices, distances

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
            candidate_set: A candidate set of length ``(self.query_embedding_matrix.shape[0], k)``
                where each row is a list of the indices of the k best menu vectors for the
                query vector at that index.
            distances: Distances from each query vector to menu vector in ``candidate_set``.
        """
        # a (currently empty) array of the k best right matches for each left record
        candidate_set = np.empty((self.query_embedding_matrix.shape[0], k))
        distances = np.empty((self.query_embedding_matrix.shape[0], k))
        for block_key in self.query_blocks_dict:
            print(f"Conducting search for candidate pairs on the {block_key} block")
            left_idx = self.query_blocks_dict[block_key].to_numpy()
            right_idx = self.menu_blocks_dict[block_key].to_numpy()
            if hasattr(self, search_name):
                search_func = getattr(self, search_name)
                block_candidate_set, block_distances = search_func(
                    queries_idx=left_idx, menu_idx=right_idx, k=k
                )
            else:
                raise ValueError(f"Search function {search_name} does not exist.")
            candidate_set[left_idx] = block_candidate_set
            distances[left_idx] = block_distances
        return candidate_set, distances
