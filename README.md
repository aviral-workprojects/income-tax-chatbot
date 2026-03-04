# Indian Income Tax RAG Assistant

An AI-powered question-answering system for Indian Income Tax law, built with Retrieval-Augmented Generation (RAG). The system uses the **Income Tax Act 1961 (amended by Finance Act 2024)** as its sole authoritative knowledge source and answers natural language questions with section-level citations.

---

## Features

- Ask natural language questions about Indian income tax law
- Answers grounded exclusively in the Income Tax Act — no hallucinations
- Hybrid retrieval combining vector similarity search and BM25 keyword matching
- Cross-encoder reranking for high-precision context selection
- Source excerpts with page numbers and section references shown with every answer
- Built-in Income Tax Calculator for Old and New regimes (AY 2025-26)
- Side-by-side regime comparison to help choose the better option
- Vector database persisted to disk — embeddings built only once

---

## Tech Stack

| Component | Library / Service |
|---|---|
| UI | Streamlit |
| PDF Extraction | pdfplumber |
| Embeddings | sentence-transformers/all-mpnet-base-v2 |
| Vector Database | ChromaDB (persistent) |
| Keyword Search | BM25 (rank-bm25) |
| Reranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | llama-3.3-70b-versatile via Groq API |
| Environment | python-dotenv |

---

## Project Structure

```
income_tax_rag/
├── app.py              # Streamlit UI — three pages: Chat, Calculator, Comparison
├── rag_pipeline.py     # End-to-end RAG orchestration + Groq LLM call
├── retriever.py        # Hybrid retrieval (Vector + BM25) and cross-encoder reranking
├── vector_store.py     # ChromaDB wrapper — build once, load every run
├── pdf_processor.py    # PDF loading, text cleaning, section-aware chunking
├── tax_calculator.py   # Slab-wise tax computation for Old and New regimes
├── config.py           # All constants and tunable parameters in one place
├── requirements.txt    # Python dependencies
├── .env                # Your API key (never commit this)
├── data/
│   └── income_tax_act.pdf   ← place the PDF here before running
└── chroma_db/               ← auto-created on first run, do not delete
```

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- A free Groq API key — get one at https://console.groq.com
- The Income Tax Act 1961 PDF (filename: `income_tax_act.pdf`)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/income-tax-rag.git
cd income-tax-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the PDF

Place the Income Tax Act PDF at:

```
data/income_tax_act.pdf
```

### 5. Configure your API key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### 6. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## First Run Note

On the first run, the system processes the full PDF (~800-900 pages) and builds the vector index. This takes **15-30 minutes** depending on your hardware. On every subsequent run, the index loads from `chroma_db/` in a few seconds.

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| CHUNK_SIZE_WORDS | 500 | Words per text chunk |
| CHUNK_OVERLAP_WORDS | 120 | Overlap between consecutive chunks |
| TOP_K_RETRIEVAL | 10 | Candidates from hybrid search |
| TOP_K_RERANK | 5 | Final chunks passed to the LLM |
| VECTOR_WEIGHT | 0.6 | Weight for vector score in hybrid merge |
| BM25_WEIGHT | 0.4 | Weight for BM25 score in hybrid merge |
| LLM_MAX_TOKENS | 1024 | Maximum tokens in LLM response |
| LLM_TEMPERATURE | 0.1 | Lower = more factual, less creative |

---

## RAG Pipeline

```
User Question
     |
     v
Hybrid Retrieval
     |-- Vector Search  (ChromaDB)  -- semantic similarity
     |-- BM25 Search                -- exact keyword matching
              |
              v  top 10 candidates (weighted score merge)
     Cross-Encoder Reranking
              |
              v  top 5 most relevant chunks
     Context Construction
              |
              v
     Groq LLM (llama-3.3-70b-versatile)
              |
              v
     Answer + Source References
```

---

## Deploying to Streamlit Cloud

> **Important:** Streamlit Cloud free tier has ~1 GB RAM. Loading the sentence-transformer model plus a ChromaDB index for an 800-page PDF will likely exceed this limit. For a working cloud deployment, use Hugging Face Spaces (with a persistent storage addon) or a small cloud VM (AWS EC2 t3.medium or equivalent).

If you want to try Streamlit Cloud regardless:

1. Push this repository to GitHub. Use the `.gitignore` below — do not commit the PDF or `chroma_db/`.
2. Add `GROQ_API_KEY` as a secret in the Streamlit Cloud dashboard under **Settings > Secrets**.
3. For the PDF: host it at a public URL and fetch it programmatically at startup instead of committing the binary to git.
4. Be aware the cold-start index build may time out on free tier.

---

## Recommended .gitignore

```
# Python
__pycache__/
*.pyc
venv/
.env

# Large generated files -- do not commit
data/
chroma_db/

# OS
.DS_Store
Thumbs.db
```

---

## Example Questions

- What deductions are available under Section 80C?
- How is HRA calculated for salaried employees?
- What is the tax treatment of long-term capital gains?
- Explain presumptive taxation under Section 44AD.
- What are the TDS rates under Section 194C?
- What is the basic exemption limit under the new tax regime?

---

## Disclaimer

This tool is for **educational and research purposes only**. Tax laws are complex and subject to change. Always consult a qualified Chartered Accountant for personal tax advice.

---

## License

MIT License. See `LICENSE` for details.
