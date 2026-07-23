"""
embeddings.py
--------------
Wraps a Sentence Transformers model (via LangChain's HuggingFace
embeddings wrapper) so it can be reused consistently across
ingestion (vector_store.py) and querying (rag_pipeline.py).

Using a local Sentence Transformers model keeps embedding generation
free and fast (no external API calls needed for this step — only
the final answer generation uses the Gemini API).
"""

from langchain_huggingface import HuggingFaceEmbeddings

# Default model: small, fast, strong general-purpose quality.
# Runs locally on CPU without issues.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embedding_instance = None  # module-level cache so we don't reload the model repeatedly


def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    Load (or return a cached) Sentence Transformers embedding model
    wrapped for LangChain compatibility.

    Args:
        model_name: HuggingFace model id / Sentence-Transformers model name.

    Returns:
        A HuggingFaceEmbeddings instance ready to use with Chroma.
    """
    global _embedding_instance

    if _embedding_instance is None:
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    return _embedding_instance
