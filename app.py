"""
app.py
--------
Streamlit frontend for the AI Document Question Answering System (RAG).

Flow:
1. User uploads one or more PDFs.
2. On "Process Documents": extract text -> split into chunks ->
   embed -> store in persistent ChromaDB (duplicates skipped).
3. User types a question.
4. Top-k relevant chunks are retrieved and passed to Gemini, which
   answers strictly from that context.
5. Answer + source chunks (file/page) are displayed.
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Loads GEMINI_API_KEY / GROQ_API_KEY from the .env file sitting next to
# this script. We build an explicit path (instead of relying on the
# current working directory) because streamlit run can be launched from
# a different folder, which otherwise silently fails to find .env.
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

try:
    load_dotenv(dotenv_path=ENV_PATH, encoding="utf-8")
except UnicodeDecodeError:
    st.warning(
        "⚠️ Could not read .env file (bad encoding). "
        "Re-save it as UTF-8, or set environment variables manually. "
        "See the troubleshooting note in README.md."
    )

from utils.pdf_loader import extract_text_from_multiple_pdfs
from utils.text_splitter import split_documents
from utils.vector_store import (
    get_vector_store,
    add_chunks_to_store,
    get_collection_count,
    clear_collection,
)
from utils.rag_pipeline import answer_question


# ----------------------------- Page Setup -----------------------------
st.set_page_config(
    page_title="AI Document Q&A (RAG)",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Question Answering System")
st.caption("Upload PDFs and ask questions — answers are generated strictly from your documents using RAG + Gemini.")


# ----------------------------- Session State -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vector_store()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"question":..., "answer":..., "sources":...}


# ----------------------------- Sidebar: Upload & Settings -----------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    gemini_key_set = bool(os.environ.get("GEMINI_API_KEY"))
    groq_key_set = bool(os.environ.get("GROQ_API_KEY"))

    if gemini_key_set:
        st.success("Gemini API key detected ✅ (primary)")
    else:
        st.warning("GEMINI_API_KEY not set — will try Groq fallback if configured.")

    if groq_key_set:
        st.success("Groq API key detected ✅ (fallback)")
    else:
        st.warning("GROQ_API_KEY not set — no fallback if Gemini fails.")

    if not gemini_key_set and not groq_key_set:
        st.error("No API keys set. Set at least one before asking questions.")
        with st.expander("🔍 Troubleshooting"):
            st.markdown(f"""
            - Looking for `.env` at: `{ENV_PATH}`
            - File exists: **{os.path.exists(ENV_PATH)}**
            - Make sure the file is named exactly `.env` (not `.env.txt` —
              Windows often hides the real extension; check "File name extensions"
              in File Explorer's View tab to confirm).
            - Make sure it's saved in the **same folder as `app.py`**.
            - Make sure the lines inside read exactly:
              `GEMINI_API_KEY=your_actual_key` and/or `GROQ_API_KEY=your_actual_key`
              (no quotes, no spaces around `=`).
            - After fixing it, fully stop and restart `streamlit run app.py`
              (editing `.env` while the app is running won't reload it).
            """)

    st.divider()
    st.header("📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, step=50)
    top_k = st.slider("Chunks to retrieve per question (top-k)", 1, 10, 4)

    process_clicked = st.button("🚀 Process Documents", type="primary", use_container_width=True)

    st.divider()
    doc_count = get_collection_count(st.session_state.vector_store)
    st.metric("Chunks stored in database", doc_count)

    if st.button("🗑️ Reset Database", use_container_width=True):
        clear_collection()
        st.session_state.vector_store = get_vector_store()
        st.session_state.chat_history = []
        st.success("Database cleared.")
        st.rerun()


# ----------------------------- Document Processing -----------------------------
if process_clicked:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF file first.")
    else:
        with st.spinner("Extracting text, chunking, and generating embeddings..."):
            try:
                # Save uploaded files to a temp directory so pypdf can read them by path
                temp_dir = tempfile.mkdtemp()
                saved_paths = []
                for uploaded_file in uploaded_files:
                    path = os.path.join(temp_dir, uploaded_file.name)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_paths.append(path)

                # 1. Extract text
                pages_data = extract_text_from_multiple_pdfs(saved_paths)

                # 2. Split into chunks
                chunks = split_documents(
                    pages_data,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                # 3 & 4. Embed + store (duplicates automatically skipped)
                added_count = add_chunks_to_store(st.session_state.vector_store, chunks)

                skipped_count = len(chunks) - added_count

                st.sidebar.success(
                    f"Processed {len(uploaded_files)} file(s): "
                    f"{added_count} new chunks added, {skipped_count} duplicate chunks skipped."
                )

            except (FileNotFoundError, ValueError) as e:
                st.sidebar.error(f"Document processing error: {e}")
            except Exception as e:
                st.sidebar.error(f"Unexpected error while processing documents: {e}")


# ----------------------------- Question & Answer -----------------------------
st.subheader("💬 Ask a question about your documents")

question = st.text_input("Type your question here", placeholder="e.g. What is the main conclusion of the report?")
ask_clicked = st.button("Ask", type="primary")

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    elif get_collection_count(st.session_state.vector_store) == 0:
        st.warning("No documents in the database yet. Upload and process at least one PDF first.")
    else:
        with st.spinner("Retrieving relevant context and generating answer..."):
            try:
                result = answer_question(
                    st.session_state.vector_store,
                    question,
                    top_k=top_k
                )
                if result["provider"] != "Gemini":
                    st.info(f"ℹ️ Gemini was unavailable — answered using {result['provider']} instead.")
                st.session_state.chat_history.insert(0, {
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "provider": result["provider"]
                })
            except EnvironmentError as e:
                st.error(str(e))
            except ValueError as e:
                st.warning(str(e))
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ----------------------------- Display Chat History -----------------------------
if st.session_state.chat_history:
    st.divider()
    st.subheader("📝 Answers")

    for i, entry in enumerate(st.session_state.chat_history):
        with st.container(border=True):
            st.markdown(f"**Q: {entry['question']}**")
            st.caption(f"Answered by: {entry.get('provider', 'Gemini')}")
            st.markdown(entry["answer"])

            with st.expander(f"📚 View {len(entry['sources'])} source chunk(s) used"):
                for j, src in enumerate(entry["sources"], start=1):
                    st.markdown(f"**Excerpt {j} — `{src['source']}`, page {src['page']}** (distance: {src['score']:.4f})")
                    st.text(src["text"][:500] + ("..." if len(src["text"]) > 500 else ""))
                    st.markdown("---")
else:
    st.info("Upload PDFs from the sidebar, click 'Process Documents', then ask a question here.")
