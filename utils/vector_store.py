"""
vector_store.py
-----------------
Manages a persistent ChromaDB collection: adding new chunks
(with duplicate-embedding prevention via content hash), and
running top-k similarity search at query time.
"""

import os
from typing import List, Dict
import chromadb
from langchain_chroma import Chroma
from utils.embeddings import get_embedding_model

# Persistent on-disk location for the Chroma database
CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")
COLLECTION_NAME = "document_qa_collection"


def get_vector_store() -> Chroma:
    """
    Create (or connect to) the persistent Chroma vector store.

    Returns:
        A LangChain Chroma vector store instance backed by an
        on-disk PersistentClient, using the shared embedding model.
    """
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

    embedding_model = get_embedding_model()
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
    )
    return vector_store


def get_existing_hashes(vector_store: Chroma) -> set:
    """
    Fetch the set of content_hash values already stored in the
    collection, so we can skip re-embedding identical chunks
    (duplicate prevention when the same PDF is uploaded twice).

    Args:
        vector_store: The Chroma vector store instance.

    Returns:
        A set of content_hash strings currently stored.
    """
    try:
        existing = vector_store.get(include=["metadatas"])
        metadatas = existing.get("metadatas", []) or []
        return {m.get("content_hash") for m in metadatas if m and m.get("content_hash")}
    except Exception:
        # Empty/new collection
        return set()


def add_chunks_to_store(vector_store: Chroma, chunks: List[Dict]) -> int:
    """
    Embed and add new chunks to the vector store, skipping any chunk
    whose content_hash already exists (duplicate prevention).

    Args:
        vector_store: The Chroma vector store instance.
        chunks: List of chunk dicts from text_splitter.split_documents().

    Returns:
        Number of chunks actually added (post-deduplication).
    """
    if not chunks:
        return 0

    existing_hashes = get_existing_hashes(vector_store)

    texts, metadatas, ids = [], [], []

    for chunk in chunks:
        if chunk["content_hash"] in existing_hashes:
            continue  # Skip duplicate content

        texts.append(chunk["text"])
        metadatas.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "content_hash": chunk["content_hash"],
        })
        ids.append(chunk["chunk_id"])
        existing_hashes.add(chunk["content_hash"])  # avoid dupes within same batch too

    if not texts:
        return 0

    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    return len(texts)


def similarity_search(vector_store: Chroma, query: str, top_k: int = 4) -> List[Dict]:
    """
    Run a top-k similarity search against the vector store.

    Args:
        vector_store: The Chroma vector store instance.
        query: The user's question.
        top_k: Number of most relevant chunks to retrieve.

    Returns:
        List of dicts with "text", "source", "page", "score" (lower = more similar,
        since Chroma returns a distance).
    """
    results = vector_store.similarity_search_with_score(query, k=top_k)

    formatted = []
    for doc, score in results:
        formatted.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": score
        })
    return formatted


def get_collection_count(vector_store: Chroma) -> int:
    """Return how many chunks currently exist in the collection."""
    try:
        data = vector_store.get()
        return len(data.get("ids", []) or [])
    except Exception:
        return 0


def clear_collection() -> None:
    """
    Delete the entire persistent collection (used by a 'Reset Database'
    button in the UI, in case the user wants to start fresh).
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection may not exist yet
