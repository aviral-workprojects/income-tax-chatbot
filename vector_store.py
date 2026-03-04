"""
vector_store.py
---------------
Manages the ChromaDB vector database:
  - Create a new collection and insert chunks with embeddings.
  - Load an existing persisted collection from disk.
  - Perform similarity search (vector retrieval).

ChromaDB stores vectors on disk so embeddings are only computed once.
On every subsequent run the database is loaded from the chroma_db/ folder.
"""

import logging
import os
from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_DB_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thin wrapper around a ChromaDB collection.

    Usage
    -----
    vs = VectorStore()

    # First run – build the database from chunks
    if not vs.exists():
        vs.build(chunks)

    # Every run – load for querying
    vs.load()
    results = vs.search("What is Section 80C?", top_k=10)
    """

    def __init__(self):
        # Persistent ChromaDB client: data is saved to CHROMA_DB_DIR
        self.client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        # Load the embedding model (downloaded automatically on first use)
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.collection = None   # set by build() or load()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def exists(self) -> bool:
        """Return True if the collection already exists in the database."""
        existing = [c.name for c in self.client.list_collections()]
        return CHROMA_COLLECTION_NAME in existing

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embed all chunks and insert them into a new ChromaDB collection.

        Parameters
        ----------
        chunks : list of dicts produced by pdf_processor.load_and_process_pdf()
        """
        logger.info(f"Building vector store with {len(chunks)} chunks …")

        # Delete any stale collection with the same name
        try:
            self.client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass  # collection did not exist – that is fine

        self.collection = self.client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            # cosine distance is appropriate for sentence embeddings
            metadata={"hnsw:space": "cosine"},
        )

        # Process in batches to avoid memory spikes and show progress
        batch_size = 64
        total = len(chunks)

        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start : batch_start + batch_size]

            texts = [c["text"] for c in batch]
            ids = [f"chunk_{c['chunk_index']}" for c in batch]
            metadatas = [
                {
                    "page": str(c["page"]),
                    "chunk_index": str(c["chunk_index"]),
                    "doc_name": c["doc_name"],
                    "section": c["section"],
                }
                for c in batch
            ]

            # Generate dense embeddings for the batch
            embeddings = self.embed_model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()

            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            processed = min(batch_start + batch_size, total)
            logger.info(f"  Indexed {processed}/{total} chunks …")

        logger.info("Vector store built and persisted to disk.")

    def load(self) -> None:
        """Load an existing collection from disk."""
        self.collection = self.client.get_collection(CHROMA_COLLECTION_NAME)
        count = self.collection.count()
        logger.info(f"Loaded vector store: {count} chunks available.")

    def search(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k most similar chunks for a query.

        Returns a list of result dicts:
            text       : str   – chunk content
            metadata   : dict  – page, section, doc_name, chunk_index
            score      : float – cosine similarity (higher = more similar)
        """
        if self.collection is None:
            raise RuntimeError("Call load() or build() before searching.")

        query_embedding = self.embed_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns distances (lower = closer); convert to similarity
        formatted = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist   # cosine distance → similarity
            formatted.append({
                "text": doc,
                "metadata": meta,
                "score": similarity,
            })

        return formatted
