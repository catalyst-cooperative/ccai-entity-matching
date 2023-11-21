"""Blocking methods to create a candidate set of tuples matches."""
import logging

import faiss
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class DataframeEmbedder:
    """A class for creating an embedding matrix for one dataframe or a pair of dataframes."""

    def __init__(
        self,
        left_df: pd.DataFrame,
        column_transformers: list[tuple],
        blocking_col: str = "",
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
            column_transformers: List of (name, transformer, columns) tuples specifying the
                transformer objects to be applied for embedding each column in the dataframe or
                dataframes. Transformers must have a fit and transform method. Used as the
                transformers list input to ``sklearn.compose.ColumnTransformer``.
            blocking_col: The name of the column to get blocks of indices of the
                dataframes. Default is the empty string if no manual blocking on
                the value of a column is to be performed. Passing a blocking column
                will not modify the dataframes themselves, just set the respective
                dictionary of blocks.
            right_df: The right dataframe to be embedded. If it's an empty dataframe, then
                only ``left_df`` will be embedded. The index should be the default, numeric
                index.
        """
        self.left_df = left_df.copy()
        self.right_df = right_df.copy()
        self.column_transformers = column_transformers
        self.blocking_col = blocking_col
        self.column_weights = None
        self.left_embedding_matrix: np.ndarray | None = None
        self.right_embedding_matrix: np.ndarray | None = None
        self.left_blocks_dict: dict[str, list] = {}
        self.right_blocks_dict: dict[str, list] = {}

    def set_col_blocks(self) -> None:
        """Get the indices of blocks of records in a dataframe for a blocking column.

        Get blocks of records in a dataframe where the value of the ``blocking_col``
        column is equal within all records in a block. Update ``self.left_blocks_dict`` and
        ``self.right_blocks_dict`` to be a dictionary with keys as each unique value in
        the ``blocking_col`` column and values as the index of the rows with that value.
        This is the result of ``df.groupby(blocking_col).groups``. Update ``self.blocking_col``

        Arguments:
            blocking_col: The name of the column to block on.
        """
        if self.blocking_col in self.left_df.columns:
            self.left_blocks_dict = self.left_df.groupby(self.blocking_col).groups
        if self.blocking_col in self.right_df.columns:
            self.right_blocks_dict = self.right_df.groupby(self.blocking_col).groups
            # TODO: do this better
            shared_keys = list(
                self.left_blocks_dict.keys() & self.right_blocks_dict.keys()
            )
            self.left_blocks_dict = {k: self.left_blocks_dict[k] for k in shared_keys}
            self.right_blocks_dict = {k: self.right_blocks_dict[k] for k in shared_keys}

    def embed_dataframes(self) -> None:
        """Embed left (and right) dataframes and create blocks from a column.

        Set `self.left_embedding_matrix` and `self.right_embedding_matrix` with
        matrix embeddings for `self.left_df` and `self.right_df`. Embed attributes
        based on the transformer pipeline in `self.embedding_transformers`.
        Optionally set `self.left_blocks_dict` and ``self.right_blocks_dict``
        to be the indices of blocks within ``blocking_col``.
        """
        embedder = Pipeline(
            [
                (
                    "column_embedding",
                    ColumnTransformer(
                        transformers=self.column_transformers,
                        transformer_weights=self.column_weights,
                    ),
                ),
            ]
        )
        if not self.right_df.empty:
            concat_df = pd.concat([self.left_df, self.right_df])
        else:
            concat_df = self.left_df
        embedder.fit(concat_df)
        self.left_embedding_matrix = embedder.transform(self.left_df)
        if not self.right_df.empty:
            self.right_embedding_matrix = embedder.transform(self.right_df)
        if self.blocking_col != "":
            self.set_col_blocks()
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
