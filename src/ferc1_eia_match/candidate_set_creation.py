"""Blocking methods to create a candidate set of tuples matches."""
import logging

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class DataframeEmbedder:
    """A class for creating an embedding matrix for one dataframe or a pair of dataframes."""

    def __init__(
        self,
        left_df: pd.DataFrame,
        col_embedding_dict: dict[str, str],
        right_df: pd.DataFrame = pd.DataFrame(),
    ):
        """Initialize a dataframe embedder object to vectorize one or two dataframe.

        If no ``right_df`` is given, then only ``left_df`` will be embedded. The left and right
        dataframe must share all the columns that are going to be vectorized. Column names of
        the embedded columns must be keys in ``col_embedding_dict``.

        Arguments:
            left_df: The left dataframe to be embedded. If no ``right_df`` is given, then this
                is the only dataframe that will be embedded. The index should be the default,
                numeric index.
            col_embedding_dict: A dictionary specifying the methods for embedding each column
                in the dataframe or dataframes. All columns that are to be vectorized must
                be keys in the dictionary.
            right_df: The right dataframe to be embedded. If it's an empty dataframe, then
                only ``left_df`` will be embedded. The index should be the default, numeric
                index.
        """
        self.left_df = left_df
        self.right_df = right_df
        self.col_embedding_dict = col_embedding_dict
        # TODO: what's actually the best data structure for holding these embedding vectors?
        # fillna until they're the same length columns and just turn this into a dataframe?
        self.left_embedding_attribute_dict: dict[str, np.ndarray] = {}
        self.right_embedding_attribute_dict: dict[str, np.ndarray] = {}
        self.left_embedding_matrix: np.ndarray | None = None
        self.right_embedding_matrix: np.ndarray | None = None
        self.blocking_col: str = ""
        self.left_blocks_dict: dict[str, list] = {}
        self.right_blocks_dict: dict[str, list] = {}
        # TODO: incorporate dictionary for choosing how to handle nulls
        # self.null_handling_dict = null_handling_dict

    def tfidf_vectorize(self, column_name: str) -> None:
        """Use TF-IDF to create vector embeddings for a column."""
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
        vectorizer = TfidfVectorizer()
        vectorizer.fit(pd.concat([left_series, right_series]))
        self.left_embedding_attribute_dict[column_name] = vectorizer.transform(
            left_series
        )
        self.right_embedding_attribute_dict[column_name] = vectorizer.transform(
            right_series
        )

    def min_max_scale(self, column_name: str):
        """Scale numeric column between 0 and 1."""
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
        scaler = MinMaxScaler()
        scaler.fit(full_array)
        self.left_embedding_attribute_dict[column_name] = scaler.transform(left_array)
        self.right_embedding_attribute_dict[column_name] = scaler.transform(right_array)

    def equal_weight_aggregate(self, embedded_attribute_dict: dict):
        """Equally weight and combine attribute embedding vectors into tuple embeddings."""
        # TODO: add an assert here to make sure the row count in all the matrices are the same
        tuple_embedding = hstack(list(embedded_attribute_dict.values()))
        tuple_embedding = tuple_embedding.tocsr()
        return tuple_embedding

    def set_col_blocks(self, blocking_col: str):
        """Get the indices of blocks of records in a dataframe for a blocking column.

        Get blocks of records in a dataframe where the value of the ``blocking_col``
        column is equal within all records in a block. Update ``self.left_blocks_dict`` and
        ``self.right_blocks_dict`` to be a dictionary with keys as each unique value in
        the ``blocking_col`` column and values as the index of the rows with that value.
        This is the result of ``df.groupby(blocking_col).groups``. Update ``self.blocking_col``

        Arguments:
            df: The dataframe from which to get blocks of records.
            blocking_col: The name of the column to block on.
        """
        self.blocking_col = blocking_col
        if blocking_col in self.left_df.columns:
            self.left_blocks_dict = self.left_df.groupby(blocking_col).groups
        if blocking_col in self.right_df.columns:
            self.right_blocks_dict = self.right_df.groupby(blocking_col).groups

    def embed_dataframes(self, blocking_col: str = "") -> None:
        """Embed left (and right) dataframes and create blocks from a column.

        Set `self.left_embedding_matrix` and `self.right_embedding_matrix` with
        matrix embeddings for `self.left_df` and `self.right_df`. Embed attributes
        based on the functions for each column in `self.col_embedding_dict`.
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
        # TODO: there's probably a better way to do this
        for column_name, vectorizer in self.col_embedding_dict.items():
            if vectorizer == "tfidf":
                self.tfidf_vectorize(column_name=column_name)
            elif vectorizer == "min_max_scale":
                self.min_max_scale(column_name=column_name)
        # TODO: allow for other types of aggregation
        self.left_embedding_matrix = self.equal_weight_aggregate(
            self.left_embedding_attribute_dict
        )
        self.right_embedding_matrix = self.equal_weight_aggregate(
            self.right_embedding_attribute_dict
        )
        if blocking_col != "":
            self.set_col_blocks(blocking_col=blocking_col)


class SimilaritySearcher:
    """Conduct a search to select a candidate set of likely matched tuples.

    Some methods perform an exact search over all vectors in the index while
    other searches perform an approximate nearest neighbors search, sacrificing
    some precision for increased speed.

    FAISS indexes: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """

    def __init__(self, query_embedding_matrix, menu_embedding_matrix, base_index=None):
        """Initialize a similarity search object to create a candidate set of tuple pairs.

        If a spare matrix is passed, it will be made dense.

        Arguments:
            query_embedding_matrix: The embedding matrix representing the query vectors, or
                vectors for which to search for the k most similar matches from the
                ``menu_embedding_matrix`` i.e. "for each query vector, find the top k most
                similar vectors from the index of potential matches".
            menu_embedding_matrix: The embedding matrix representing the potential match vectors
                for the query vectors. These vector embeddings which will be searched over to find
                vectors most similar or closest to the query vectors.
            base_index (list): The base index for ``menu_embedding_matrix`` that will be
                searched over for matches. Used to map back to the original vectors if the
                embeddings were transformed in some way. Default is None and the search
                will return the position of the vectors in the embedding matrix.
        """
        if issparse(query_embedding_matrix):
            query_embedding_matrix = query_embedding_matrix.todense()
        if issparse(menu_embedding_matrix):
            menu_embedding_matrix = menu_embedding_matrix.todense()
        self.query_embedding_matrix = np.float32(query_embedding_matrix)
        self.menu_embedding_matrix = np.float32(menu_embedding_matrix)
        self.base_index = base_index

    def l2_distance_search(self, k=5):
        """Conduct an exact search for smallest L2 distance between vectors.

        Arguments:
            k (int): The number of potential record matches to find for each query vector.

        Returns:
            Numpy array of shape ``(len(query_embeddings), k)`` where each row represents
            the k indices of the index embeddings matrix that are closest to each query
            vector. If ``base_index`` is not None, then this will be the index of the vector
            in the base index.
        """
        dim = self.menu_embedding_matrix.shape[1]
        if self.base_index is not None:
            index_l2 = faiss.IndexFlatL2(dim)
            index = faiss.IndexIDMap(index_l2)
            index.add_with_ids(self.menu_embedding_matrix, self.base_index)
        else:
            index = faiss.IndexFlatL2(dim)
            index.add(self.menu_embedding_matrix)
        distances, match_indices = index.search(self.query_embedding_matrix, k)

        return match_indices

    def cosine_similarity_search(self, k: int = 5):
        """Conduct an exact search for highest cosine similarity between vectors.

        Arguments:
            k: The number of potential record matches to find for each query vector.

        Returns:
            Numpy array of shape ``(len(query_embeddings), k)`` where each row represents
            the k indices of the index embeddings matrix that are closest to each query
            vector. If ``base_index`` is not None, then this will be the index of the vector
            in the base index.
        """
        dim = self.menu_embedding_matrix.shape[1]
        faiss.normalize_L2(self.menu_embedding_matrix)
        faiss.normalize_L2(self.query_embedding_matrix)
        # use the Inner Product Index, which is equivalent to cosine sim for normalized vectors
        if self.base_index is not None:
            index_ip = faiss.IndexFlatIP(dim)
            index = faiss.IndexIDMap(index_ip)
            index.add_with_ids(self.menu_embedding_matrix, self.base_index)
        else:
            index = faiss.IndexFlatIP(dim)
            index.add(self.menu_embedding_matrix)
        distances, match_indices = index.search(self.query_embedding_matrix, k)

        return match_indices
