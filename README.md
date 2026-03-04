# Indian Income Tax RAG Assistant

An AI-powered question-answering system for Indian Income Tax law, built with
Retrieval-Augmented Generation (RAG).  It uses the Income Tax Act 1961
(amended by Finance Act 2024) as its single authoritative knowledge source.

---

## Project Structure

```
income_tax_rag/
├── app.py              # Streamlit application (UI + page routing)
├── rag_pipeline.py     # Orchestrates the full RAG workflow
├── retriever.py        # Hybrid retrieval (Vector + BM25) + reranking
├── vector_store.py     # ChromaDB wrapper for persistent vector storage
├── pdf_processor.py    # PDF loading, cleaning, and chunking
├── tax_calculator.py   # Tax slab computation (Old & New regimes)
├── config.py           # All constants and configuration in one place
├── requirements.txt    # Python dependencies
├── data/
│   └── income_tax_act.pdf   ← place the PDF here
└── chroma_db/               ← auto-created on first run
```

---

## Prerequisites

- Python 3.10 or later
- A Groq API key (free tier available at https://console.groq.com)
- The Income Tax Act 1961 PDF placed at `data/income_tax_act.pdf`

---

## Setup Instructions

### 1. Clone or download the project

```bash
cd income_tax_rag
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# On macOS / Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** The sentence-transformers library requires PyTorch.
> On CPU-only machines the installation above works fine.
> If you have a CUDA GPU, install the matching torch version from
> https://pytorch.org/get-started/locally/ before running the command above.

### 4. Place the PDF

Copy the Income Tax Act PDF into the `data/` folder:

```
income_tax_rag/
└── data/
    └── income_tax_act.pdf
```

### 5. Set your Groq API key

```bash
# macOS / Linux
export GROQ_API_KEY="your_groq_api_key_here"

# Windows Command Prompt
set GROQ_API_KEY=your_groq_api_key_here

# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"
```

### 6. Run the application

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501

---

## First-Run Behaviour

On the **first run**, the system will:

1. Extract text from all ~800–900 pages of the PDF using pdfplumber.
2. Clean and split the text into ~500-word overlapping chunks (~6,000–8,000 chunks).
3. Generate dense embeddings for every chunk using `all-mpnet-base-v2`.
4. Persist the vectors to the `chroma_db/` folder.

**This takes approximately 15–30 minutes** depending on your hardware.

On every **subsequent run**, the system loads the pre-built vector database from
disk in a few seconds — no re-processing is needed.

---

## How It Works

```
User Question
    │
    ▼
Hybrid Retrieval
    ├── Vector Search  (ChromaDB, cosine similarity)  — finds semantically similar chunks
    └── BM25 Search    (keyword matching)              — finds exact term matches
            │
            ▼ top 10 candidates (weighted combination of both scores)
    Cross-Encoder Reranking
            │
            ▼ top 5 most relevant chunks
    Context Construction
            │
            ▼
    Groq LLM (llama-3.1-70b-versatile)
            │
            ▼
    Answer + Source References
```

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter            | Default  | Description                              |
|----------------------|----------|------------------------------------------|
| CHUNK_SIZE_WORDS     | 500      | Words per chunk                          |
| CHUNK_OVERLAP_WORDS  | 120      | Overlap between consecutive chunks      |
| TOP_K_RETRIEVAL      | 10       | Candidates fetched from hybrid search   |
| TOP_K_RERANK         | 5        | Final chunks passed to the LLM          |
| VECTOR_WEIGHT        | 0.6      | Weight for vector score in hybrid merge |
| BM25_WEIGHT          | 0.4      | Weight for BM25 score in hybrid merge   |
| LLM_MAX_TOKENS       | 1024     | Maximum tokens in the LLM response      |
| LLM_TEMPERATURE      | 0.1      | Generation temperature (lower = safer)  |

---

## Pages

### 1. Chat Interface
Ask natural language questions about Indian income tax.
Example questions:
- "What deductions are available under Section 80C?"
- "What is the tax treatment of HRA for salaried employees?"
- "Explain the provisions of Section 44AD for presumptive taxation."
- "What are the capital gains tax rates for FY 2024-25?"

### 2. Tax Calculator
Enter your income and select a regime to see a slab-wise tax breakdown.

### 3. Regime Comparison
Enter income and your old-regime deductions to see which regime saves more tax.

---

## Disclaimer

This tool is for **educational purposes only**.  
Tax laws are complex and subject to change.  
Always consult a qualified Chartered Accountant for tax advice.
