"""
text_splitter.py
------------------
Splits extracted page-level text into smaller overlapping chunks
suitable for embedding + retrieval. Uses LangChain's
RecursiveCharacterTextSplitter, which tries to split on natural
boundaries (paragraphs -> sentences -> words) before falling back
to a hard character cut.
"""

from typing import List, Dict
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    pages_data: List[Dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150
) -> List[Dict]:
    """
    Split page-level text into overlapping chunks and attach metadata
    (source file, page number, chunk id, content hash) to each chunk.

    Args:
        pages_data: Output of pdf_loader.extract_text_from_pdf(s) —
            a list of dicts with "text", "source", "page".
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap in characters between consecutive chunks
            (helps preserve context across chunk boundaries).

    Returns:
        A list of dicts, one per chunk:
            - "text": chunk text
            - "source": originating filename
            - "page": originating page number
            - "chunk_id": unique id (source_page_chunkindex)
            - "content_hash": md5 hash of chunk text (used later for
              duplicate-embedding prevention)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for page in pages_data:
        page_text = page["text"]
        source = page["source"]
        page_num = page["page"]

        splits = splitter.split_text(page_text)

        for idx, chunk_text in enumerate(splits):
            content_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
            chunk_id = f"{source}_p{page_num}_c{idx}_{content_hash[:8]}"

            all_chunks.append({
                "text": chunk_text,
                "source": source,
                "page": page_num,
                "chunk_id": chunk_id,
                "content_hash": content_hash
            })

    return all_chunks
