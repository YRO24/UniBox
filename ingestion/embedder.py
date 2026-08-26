"""
Embedder module for UniBox using sentence-transformers.
"""

from typing import List

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model once when this module is imported.
_model = SentenceTransformer(MODEL_NAME)


def get_model_info() -> dict:
    """
    Return information about the embedding model.
    """

    return {
        "model_name": MODEL_NAME,
        "max_seq_length": _model.max_seq_length,
        "embedding_dimension": _model.get_embedding_dimension()
    }


def generate_embeddings(
    texts: List[str],
    batch_size: int = 32
) -> List[List[float]]:
    """
    Generate embeddings for a list of text chunks.

    Args:
        texts: List of text chunks.
        batch_size: Number of chunks processed at once.

    Returns:
        List of embedding vectors.
    """

    if not texts:
        return []

    # Remove empty chunks.
    texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not texts:
        return []

    embeddings = _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.tolist()