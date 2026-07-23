"""
pdf_loader.py
--------------
Handles extraction of text from one or multiple PDF files.

Uses PyPDF (via pypdf) to read PDF pages and returns structured
documents with metadata (source filename + page number) so that
later stages (chunking, retrieval) can trace answers back to the
exact page/file they came from.
"""

from typing import List, Dict
from pypdf import PdfReader
import os


def extract_text_from_pdf(file_path: str) -> List[Dict]:
    """
    Extract text from a single PDF file, page by page.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        A list of dicts, one per page, each containing:
            - "text": extracted text of that page
            - "source": the original filename
            - "page": 1-indexed page number

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the PDF cannot be read/parsed.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    filename = os.path.basename(file_path)
    pages_data = []

    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise ValueError(f"Could not read PDF '{filename}': {e}")

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            # Skip pages that fail to parse (e.g. corrupted/image-only)
            text = ""

        text = text.strip()
        if text:  # Skip empty pages
            pages_data.append({
                "text": text,
                "source": filename,
                "page": page_num
            })

    if not pages_data:
        raise ValueError(
            f"No extractable text found in '{filename}'. "
            "It may be a scanned/image-only PDF."
        )

    return pages_data


def extract_text_from_multiple_pdfs(file_paths: List[str]) -> List[Dict]:
    """
    Extract text from multiple PDF files.

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        Combined list of page-level dicts (text, source, page) across
        all successfully processed PDFs. Files that fail are skipped
        with an error message printed (caller can also catch/log this
        differently if needed).
    """
    all_pages = []
    errors = []

    for path in file_paths:
        try:
            pages = extract_text_from_pdf(path)
            all_pages.extend(pages)
        except (FileNotFoundError, ValueError) as e:
            errors.append(str(e))

    if errors:
        # Non-fatal: surface errors but continue with whatever succeeded
        print(f"[pdf_loader] Warnings while processing files: {errors}")

    if not all_pages:
        raise ValueError("No text could be extracted from any of the provided PDFs.")

    return all_pages
