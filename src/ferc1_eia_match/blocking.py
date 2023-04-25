"""Blocking methods to create a candidate set of tuples matches."""
import faiss
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler


class AttributeEmbedder:
    """A class for creating embedding vectors for the attributes of a dataframe."""

    def __init__(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        col_embedding_dict: dict[str, str],
    ):
        """Initialize an attribute embedder object to vectorize the left and right dataframe.

        The left and right dataframe must share all the columns that are going to be
        vectorized. Column names of the embedded columns must be keys in ``col_embedding_dict``.
        """
        # TODO: make this class work with just one dataframe
        self.left_df = left_df
        self.right_df = right_df
        self.col_embedding_dict = col_embedding_dict
        # TODO: what's actually the best data structure for holding these embedding vectors?
        # just keep them as one matrix? makes tuple aggregation harder
        self.left_embedded_attribute_dict = {}
        self.right_embedded_attribute_dict = {}
        # TODO: incorporate dictionary for choosing how to handle nulls
        # self.null_handling_dict = null_handling_dict

    def tfidf_vectorize(self, column_name: str):
        """Use TF-IDF to create vector embeddings for a column."""
        # fill nulls with the empty string so they become 0 vectors
        # TODO: add assertion to make sure column is a string column
        left_series = self.left_df[column_name].fillna("")
        right_series = self.right_df[column_name].fillna("")
        vectorizer = TfidfVectorizer()
        vectorizer.fit(pd.concat([left_series, right_series]))
        self.left_embedded_attribute_dict[column_name] = vectorizer.transform(
            left_series
        )
        self.right_embedded_attribute_dict[column_name] = vectorizer.transform(
            right_series
        )

    def min_max_scale(self, column_name: str):
        """Scale numeric column between 0 and 1."""
        # fill nulls with the median
        # TODO: add assertion to make sure column is numeric
        left_series = self.left_df[column_name]
        right_series = self.right_df[column_name]
        full_series = pd.concat([left_series, right_series])
        med = full_series.median()
        full_array = full_series.fillna(med).to_numpy().reshape(-1, 1)
        left_array = left_series.fillna(med).to_numpy().reshape(-1, 1)
        right_array = right_series.fillna(med).to_numpy().reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(full_array)
        self.left_embedded_attribute_dict[column_name] = scaler.transform(left_array)
        self.right_embedded_attribute_dict[column_name] = scaler.transform(right_array)

    def embed_attributes(self):
        # TODO: there's probably a better way to do this
        for column_name, vectorizer in self.col_embedding_dict.items():
            if vectorizer == "tfidf":
                self.tfidf_vectorize(column_name=column_name)
            elif vectorizer == "min_max_scale":
                self.min_max_scale(column_name=column_name)


class TupleEmbedder:
    """Combine attribute embedding vectors into tuple embeddings for each row of a dataframe."""

    # TODO fill this class out

    def equal_weight_aggregation(embedded_attribute_dict: dict):
        # TODO: add an assert here to make sure the row count in all the matrices are the same
        tuple_embedding = hstack(list(embedded_attribute_dict.values()))
        tuple_embedding = tuple_embedding.tocsr()
        return tuple_embedding


class SimilaritySearch:
    """Conduct a similarity search to select a candidate set of likely matched tuples.

    FAISS indexes: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    """

    # TODO implement with cosine similarity as well

    def l2_search(
        self,
        left_embeddings,
        right_embeddings,
        left_query: bool = True,
        right_query: bool = False,
        k: int = 5,
    ):
        """Conduct an exact search for L2 distance between vectors.

        Right and left embedding matrices must be sparse.
        """
        candidate_set = np.empty((0, 2))
        if left_query:
            right_d = right_embeddings.shape[1]
            index = faiss.IndexFlatL2(right_d)
            index.add(right_embeddings.todense())
            # TODO: do without looping?, can just feed index.search the left embedding matrix?
            for i in range(left_embeddings.shape[0]):
                xq = left_embeddings[i].todense()
                distances, match_indices = index.search(xq, k)
                new_matches = np.concatenate(
                    (np.tile(i, (k, 1)), match_indices.T), axis=1
                )
                candidate_set = np.append(candidate_set, new_matches, axis=0)
        if right_query:
            left_d = left_embeddings.shape[1]
            index = faiss.IndexFlatL2(left_d)
            index.add(left_embeddings.todense())
            # TODO: do without looping?, can just feed index.search the left embedding matrix?
            for i in range(right_embeddings.shape[0]):
                xq = right_embeddings[i].todense()
                distances, match_indices = index.search(xq, k)
                new_matches = np.concatenate(
                    (match_indices.T, np.tile(i, (k, 1))), axis=1
                )
                candidate_set = np.append(candidate_set, new_matches, axis=0)
        # drop duplicate match pairs
        candidate_set = np.unique(candidate_set, axis=0)
        return candidate_set
