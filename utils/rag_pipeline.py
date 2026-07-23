"""
rag_pipeline.py
------------------
Ties retrieval (vector_store.similarity_search) together with
generation to answer user questions strictly based on retrieved
document chunks.

Primary LLM: Google Gemini (via google-genai SDK).
Fallback LLM: Groq (via groq SDK) — automatically used if the Gemini
call fails for any reason (deprecated model, quota limit, downtime,
missing key, etc.), so the app keeps working instead of just erroring out.
"""

import os
from typing import List, Dict
from google import genai
from google.genai import types
from groq import Groq

from utils.vector_store import similarity_search

# "gemini-flash-latest" is an alias Google keeps pointed at their current
# generally-available Flash model, so this doesn't need manual updates
# every time a specific dated model (e.g. gemini-2.5-flash) is retired.
GEMINI_MODEL = "gemini-flash-latest"

# Groq's Llama 3.x models were deprecated in June 2026; gpt-oss-120b is
# their recommended high-quality replacement, with a smaller/faster
# fallback in case that ever changes too.
GROQ_MODEL_PRIMARY = "openai/gpt-oss-120b"
GROQ_MODEL_FALLBACK = "openai/gpt-oss-20b"

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided document excerpts (context). Follow these rules strictly:\n"
    "1. Only use information found in the context below to answer.\n"
    "2. If the answer is not present in the context, say clearly: "
    "\"I could not find this information in the uploaded documents.\"\n"
    "3. Do not use outside knowledge or make assumptions beyond the context.\n"
    "4. Be concise and accurate. Cite the source file and page number(s) "
    "you used when relevant.\n"
)


def _build_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a single labeled context string."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Excerpt {i} | Source: {chunk['source']} | Page: {chunk['page']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _generate_with_gemini(prompt: str) -> str:
    """
    Attempt to generate an answer using Gemini.

    Raises:
        Exception: any failure (missing key, deprecated model, quota, etc.)
        is propagated so the caller can fall back to Groq.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    answer_text = (response.text or "").strip()
    if not answer_text:
        raise RuntimeError("Gemini returned an empty response.")
    return answer_text


def _generate_with_groq(prompt: str) -> str:
    """
    Attempt to generate an answer using Groq, trying the primary model
    first and a lighter fallback model if that also fails.

    Raises:
        Exception: if both Groq models fail (e.g. missing key).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    client = Groq(api_key=api_key)

    last_error = None
    for model_name in (GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            answer_text = (completion.choices[0].message.content or "").strip()
            if answer_text:
                return answer_text
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Groq API call failed: {last_error}")


def answer_question(vector_store, question: str, top_k: int = 4) -> Dict:
    """
    Full RAG flow: retrieve top-k relevant chunks for the question,
    then generate an answer using only that context. Tries Gemini
    first; automatically falls back to Groq if Gemini fails.

    Args:
        vector_store: The Chroma vector store instance.
        question: The user's natural-language question.
        top_k: Number of chunks to retrieve for context.

    Returns:
        A dict with:
            - "answer": generated answer text
            - "sources": list of retrieved chunk dicts (text, source, page, score)
            - "provider": which LLM actually generated the answer ("Gemini" or "Groq")

    Raises:
        ValueError: if no relevant chunks are found (empty database).
        RuntimeError: if BOTH Gemini and Groq fail.
    """
    retrieved_chunks = similarity_search(vector_store, question, top_k=top_k)

    if not retrieved_chunks:
        raise ValueError(
            "No documents found in the database. Please upload and process "
            "at least one PDF before asking questions."
        )

    context = _build_context(retrieved_chunks)
    prompt = (
        f"Context from uploaded documents:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the context above."
    )

    gemini_error = None
    try:
        answer_text = _generate_with_gemini(prompt)
        provider = "Gemini"
    except Exception as e:
        gemini_error = e
        try:
            answer_text = _generate_with_groq(prompt)
            provider = "Groq (Gemini fallback)"
        except Exception as groq_error:
            raise RuntimeError(
                f"Both Gemini and Groq failed.\n"
                f"Gemini error: {gemini_error}\n"
                f"Groq error: {groq_error}"
            )

    return {
        "answer": answer_text,
        "sources": retrieved_chunks,
        "provider": provider,
    }

