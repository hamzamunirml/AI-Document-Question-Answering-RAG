# 📄 AI Document Question Answering System (RAG)

An end-to-end Retrieval-Augmented Generation (RAG) application that lets you upload one or multiple PDF documents and ask natural-language questions about their content. Answers are generated **strictly from the uploaded documents** — no hallucinated outside knowledge.

Built with **Python, Streamlit, LangChain, ChromaDB, Sentence Transformers, and Google Gemini**.

---

## 🎯 How It Works

1. **Upload** — Upload one or more PDF files via the Streamlit UI.
2. **Extract** — Text is extracted page-by-page using `pypdf`.
3. **Chunk** — Text is split into overlapping chunks with LangChain's `RecursiveCharacterTextSplitter` for better retrieval granularity.
4. **Embed** — Each chunk is embedded locally using a Sentence Transformers model (`all-MiniLM-L6-v2`) — free, fast, runs on CPU.
5. **Store** — Embeddings are stored in a **persistent ChromaDB** collection on disk, with duplicate-content detection so re-uploading the same file doesn't create redundant entries.
6. **Ask** — When you ask a question, the top-k most similar chunks are retrieved via vector similarity search.
7. **Generate** — Retrieved chunks are sent as context to **Google Gemini** (`gemini-2.5-flash`), which is instructed to answer *only* from that context.
8. **Display** — The answer is shown alongside the exact source chunks (file name + page number) used to generate it.

---

## 🗂️ Project Structure

```
AI-Document-QA/
│
├── app.py                     # Streamlit UI — main entry point
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .env.example                # Template for your Gemini API key
├── .gitignore
├── data/                       # (optional) local scratch folder for PDFs
├── chroma_db/                  # Persistent vector database (auto-created)
└── utils/
    ├── __init__.py
    ├── pdf_loader.py           # PDF text extraction
    ├── text_splitter.py        # Chunking logic
    ├── embeddings.py           # Sentence Transformers embedding model
    ├── vector_store.py         # ChromaDB storage, dedup, similarity search
    └── rag_pipeline.py         # Retrieval + Gemini answer generation
```

---

## ⚙️ Tech Stack

| Component        | Technology                              |
|-------------------|------------------------------------------|
| Frontend UI       | Streamlit                               |
| Orchestration     | LangChain                               |
| Vector Database   | ChromaDB (persistent, local)            |
| Embeddings        | Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM               | Google Gemini API (`gemini-2.5-flash`)  |
| PDF Parsing       | pypdf                                   |

---

## 🚀 Setup & Run

### 1. Clone / download the project
```bash
cd AI-Document-QA
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Gemini API key
Get a **free** API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

Copy the example env file and fill in your key:
```bash
cp .env.example .env
```
Then edit `.env`:
```
GEMINI_API_KEY=your_actual_key_here
```

### 5. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### 6. Use it
1. Upload one or more PDFs from the sidebar.
2. Click **"🚀 Process Documents"** — wait for chunking & embedding to complete.
3. Type a question in the main panel and click **"Ask"**.
4. Read the answer, and expand **"View source chunks"** to see exactly which text was used.

---

## ✅ Features

- Multi-PDF upload and processing
- Automatic text chunking with configurable size/overlap (sidebar sliders)
- Local, free embedding generation (no API cost for embeddings)
- Persistent vector storage — documents remain searchable across app restarts
- **Duplicate-prevention**: re-uploading the same PDF won't create duplicate embeddings
- Adjustable top-k retrieval count
- Source attribution: every answer shows which file/page it came from
- Clean error handling for missing API keys, empty databases, unreadable PDFs
- "Reset Database" button to clear all stored documents

---

## 🧪 Notes & Limitations

- Scanned/image-only PDFs (no embedded text layer) will not extract text — OCR is not included in this version.
- The system answers only from uploaded documents; if the answer isn't in the context, Gemini will say so explicitly rather than guessing.
- Embeddings run locally on CPU by default; for large document sets, GPU acceleration can be enabled by changing `device` in `utils/embeddings.py`.

---

## 📷 Suggested Screenshots for Documentation

1. Sidebar showing PDF upload + "Process Documents" button with success message
2. Main panel with a question typed in and the generated answer displayed
3. Expanded "source chunks" section showing file name, page number, and excerpt text
4. Sidebar metric showing number of chunks stored in the database
5. Empty-state screen (before any documents are uploaded)

---

## 📄 License

This project is provided as-is for educational and portfolio purposes.
