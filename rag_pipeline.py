"""
rag_pipeline.py
---------------
Orchestrates the complete RAG pipeline:

1. On first run  – load the PDF, build embeddings, persist the vector DB.
2. On every run  – load the persisted vector DB and build the BM25 index.
3. For each query:
     a) Retrieve top candidates via hybrid search + reranking.
     b) Construct a context block from the retrieved chunks.
     c) Call the Groq LLM API with a carefully crafted prompt.
     d) Return the answer together with source references.

This module is imported by app.py and used as the single entry point for
all question-answering logic.
"""

import logging
from typing import Dict, Any, List

from groq import Groq

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)
from pdf_processor import load_and_process_pdf
from vector_store import VectorStore
from retriever import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt – instructs the LLM on how to behave
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert assistant specialising in Indian Income Tax law.

Your task is to answer questions accurately using only the context provided, which comes from the Income Tax Act, 1961 (amended by the Finance Act 2024).

You must follow these rules strictly.

GENERAL RULES :

1. Use only the provided context to answer the question.

2. Do not invent rules, numbers, deductions, or interpretations that are not explicitly mentioned in the context.

3. If the context does not contain enough information, clearly state:

   "The provided context does not contain sufficient information to answer this question."

4. Prefer relevant sections of the law when answering.

---

EXPLANATION STYLE :

The Income Tax Act is written in legal language. Your job is to interpret it and explain it clearly.

When answering:

• First summarise the rule in simple language
• Then provide key points or conditions from the law
• Avoid copying long legal paragraphs directly
• Convert complex clauses into clear explanations

The goal is to make the answer understandable for a normal taxpayer with no legal background.

---

STRUCTURE OF ANSWERS :

Structure answers using the following format whenever possible:

1. Short Explanation
A clear explanation of the rule or concept in simple language.

2. Key Points / Conditions
List the main provisions or requirements using bullet points.

3. Limits or Important Numbers (if mentioned in context)

4. Source Reference
Mention the relevant Section number and/or page number.

Example:

Source: Section 80C (Page 279)

---

SOURCE USAGE RULES : 

• Prefer chunks that clearly mention the relevant section number.
• Ignore context that appears unrelated to the user's question.
• Do not combine unrelated sections into one explanation.

---

CITATION RULES :

When relevant, cite the source using:

Source: Section X (Page Y)

Example:

Source: Section 80C (Page 279)

---

CALCULATION QUESTIONS : 

If the user asks a tax calculation question:

1. Identify the income amount and regime.
2. Apply the relevant tax slabs.
3. Show the calculation step by step.
4. Present the final tax amount clearly.

---

IMPORTANT :

Your goal is to interpret the law and explain it clearly, not to copy the legal text.

Always prioritise clarity, accuracy, and usefulness for the user.
"""


class RAGPipeline:
    """
    High-level class that wires together all components of the RAG system.

    Typical usage in app.py:
        pipeline = RAGPipeline()
        pipeline.initialise()    # call once at startup
        result = pipeline.ask("What are the deductions under Section 80C?")
    """

    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever: HybridRetriever = None
        self.all_chunks: List[Dict[str, Any]] = []
        self.groq_client = Groq(api_key=GROQ_API_KEY)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """
        Set up the vector store and retriever.

        - If the ChromaDB collection already exists on disk, load it.
        - Otherwise, process the PDF, generate embeddings, and build it.

        This method is called once when the Streamlit app starts.
        """
        if self.vector_store.exists():
            logger.info("Existing vector store found. Loading from disk …")
            self.vector_store.load()
            # We still need chunk texts for BM25, so reload from the PDF.
            # pdfplumber is fast enough that this is acceptable at startup.
            logger.info("Reloading chunks for BM25 index …")
            self.all_chunks = load_and_process_pdf()
        else:
            logger.info("No vector store found. Processing PDF and building index …")
            self.all_chunks = load_and_process_pdf()
            self.vector_store.build(self.all_chunks)

        # Build the HybridRetriever (includes BM25 and cross-encoder)
        self.retriever = HybridRetriever(self.vector_store, self.all_chunks)
        logger.info("RAG pipeline initialised and ready.")

    # ------------------------------------------------------------------
    # Question answering
    # ------------------------------------------------------------------

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Answer a natural language question about Indian income tax.

        Parameters
        ----------
        question : str
            The user's question.

        Returns
        -------
        dict with keys:
            answer  : str              – the generated answer
            sources : list of dicts    – retrieved chunks used as context
                Each source dict has: text, page, section, doc_name, score
        """
        if self.retriever is None:
            raise RuntimeError("Pipeline not initialised. Call initialise() first.")

        # Step 1 – retrieve relevant chunks
        retrieved = self.retriever.retrieve(question)

        if not retrieved:
            return {
                "answer": "No relevant information could be found in the knowledge base.",
                "sources": [],
            }

        # Step 2 – build the context string for the LLM
        context = self._build_context(retrieved)

        # Step 3 – call the LLM
        answer = self._generate_answer(question, context)

        # Step 4 – format source references for display
        sources = self._format_sources(retrieved)

        return {"answer": answer, "sources": sources}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Concatenate retrieved chunks into a numbered context block.

        Each chunk is clearly delimited so the LLM knows where each
        piece of source text begins and ends.
        """
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk["metadata"]
            page = meta.get("page", "?")
            section = meta.get("section", "")
            section_note = f"  |  Section {section}" if section else ""
            header = f"[Source {i} – Page {page}{section_note}]"
            parts.append(f"{header}\n{chunk['text']}")

        return "\n\n---\n\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """
        Send the question + context to the Groq LLM and return the response.
        """
        user_message = (
            f"Context from the Income Tax Act:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Please answer the question based only on the context above."
        )

        try:
            response = self.groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"An error occurred while generating the answer: {str(e)}"

    def _format_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert retrieved chunk dicts into a cleaner format for display
        in the Streamlit UI.
        """
        sources = []
        for chunk in chunks:
            meta = chunk["metadata"]
            sources.append({
                "text": chunk["text"],
                "page": meta.get("page", "N/A"),
                "section": meta.get("section", ""),
                "doc_name": meta.get("doc_name", ""),
                "rerank_score": round(chunk.get("rerank_score", 0.0), 4),
            })
        return sources
