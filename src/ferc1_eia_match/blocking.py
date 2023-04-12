"""Blocking methods to create a candidate set of tuples matches."""
# from sklearn.feature_extraction.text import TfidfVectorizer


class AttributeEmbedder:
    """A class for creating embedding vectors for the attributes of a dataframe."""

    def __init__(self, df):
        """Initialize an attribute embedder object to vectorize a dataframe."""
        self.input_df = df

    def tfidf_vectorize(self, column_name: str):
        """Use TF-IDF to create vector embeddings for a column."""
        # vectorizer = TfidfVectorizer()
        # handle nulls


class TupleEmbedder:
    """Combine attribute embedding vectors into tuple embeddings for each row of a dataframe."""


class SimlaritySearch:
    """Conduct a similarity search to select a candidate set of likely matched tuples."""
