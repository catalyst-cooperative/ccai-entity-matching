"""Helper functions to compute and output various metrics."""

from pydantic import BaseModel


class Inputs(BaseModel):
    """Configuration for input resources."""

    start_year: int
    end_year: int


class ColumnEmbedding(BaseModel):
    """Specify embedding vectorizer to use for column and any options."""

    embedding_type: str
    options: dict = {}


class EmbeddingConfig(BaseModel):
    """Configuration options for embed_dataframes."""

    embedding_map: dict[str, ColumnEmbedding]
    matching_cols: list[str]
    blocking_col: str


class SimilaritySearch(BaseModel):
    """Configuration for similarity search step."""

    distance_metric: str


class Model(BaseModel):
    """Configuration for all steps of model."""

    inputs: Inputs
    embedding: EmbeddingConfig
    similarity_search: SimilaritySearch
