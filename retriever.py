"""
retriever.py
------------
Implements the two-stage retrieval pipeline:

Stage 1 – Hybrid retrieval
    a) Vector similarity search  (semantic understanding)
    b) BM25 keyword search       (exact term matching)
    Scores from both methods are normalised and merged with a weighted sum.

Stage 2 – Cross-encoder reranking
    The merged candidates are re-scored by a cross-encoder model which reads
    the query and each document together, giving much more accurate relevance
    judgements than the bi-encoder used for initial retrieval.

The top TOP_K_RERANK results after reranking are returned to the LLM.
"""

import logging
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import (
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    RERANKER_MODEL,
)
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines ChromaDB vector search with BM25 and applies cross-encoder reranking.

    Parameters
    ----------
    vector_store : VectorStore
        A VectorStore instance that has been loaded/built.
    all_chunks : list of dicts
        The complete list of chunk dicts (needed to build the BM25 index).
    """

    def __init__(self, vector_store: VectorStore, all_chunks: List[Dict[str, Any]]):
        self.vs = vector_store
        self.all_chunks = all_chunks

        # Build the BM25 index over all chunk texts
        logger.info("Building BM25 index …")
        tokenised_corpus = [chunk["text"].lower().split() for chunk in all_chunks]
        self.bm25 = BM25Okapi(tokenised_corpus)

        # Load the cross-encoder for reranking
        logger.info(f"Loading reranker model: {RERANKER_MODEL}")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        logger.info("HybridRetriever ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Run the full retrieval pipeline for a user query.

        Returns the top TOP_K_RERANK chunks after reranking, each with:
            text           : str
            metadata       : dict  (page, section, doc_name, chunk_index)
            hybrid_score   : float
            rerank_score   : float
        """
        # Stage 1a – vector search
        vector_results = self.vs.search(query, top_k=TOP_K_RETRIEVAL)

        # Stage 1b – BM25 search (returns indices into self.all_chunks)
        bm25_results = self._bm25_search(query, top_k=TOP_K_RETRIEVAL)

        # Merge both result sets into a single candidate pool
        candidates = self._merge_results(vector_results, bm25_results)

        # Stage 2 – cross-encoder reranking
        reranked = self._rerank(query, candidates)

        return reranked[:TOP_K_RERANK]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Score all chunks with BM25 and return the top_k as result dicts.

        BM25 scores are normalised to [0, 1] for fair combination with
        cosine similarity scores from the vector store.
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)   # shape: (num_chunks,)

        # Get indices of the top_k highest scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Normalise scores to [0, 1]
        max_score = scores[top_indices[0]] if scores[top_indices[0]] > 0 else 1.0

        results = []
        for idx in top_indices:
            chunk = self.all_chunks[idx]
            normalised_score = float(scores[idx]) / max_score
            results.append({
                "text": chunk["text"],
                "metadata": {
                    "page": str(chunk["page"]),
                    "chunk_index": str(chunk["chunk_index"]),
                    "doc_name": chunk["doc_name"],
                    "section": chunk["section"],
                },
                "score": normalised_score,
            })

        return results

    def _merge_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and BM25 results using a weighted sum.

        Deduplication is done by chunk_index so the same chunk cannot appear
        twice in the candidate pool.

        Returns a list sorted by descending hybrid_score.
        """
        # Build a dict keyed by chunk_index → score accumulator
        combined: Dict[str, Dict[str, Any]] = {}

        for result in vector_results:
            key = result["metadata"]["chunk_index"]
            combined[key] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "vector_score": result["score"],
                "bm25_score": 0.0,
            }

        for result in bm25_results:
            key = result["metadata"]["chunk_index"]
            if key in combined:
                combined[key]["bm25_score"] = result["score"]
            else:
                combined[key] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "vector_score": 0.0,
                    "bm25_score": result["score"],
                }

        # Compute the weighted hybrid score for every candidate
        candidates = []
        for key, item in combined.items():
            hybrid = (
                VECTOR_WEIGHT * item["vector_score"]
                + BM25_WEIGHT * item["bm25_score"]
            )
            candidates.append({
                "text": item["text"],
                "metadata": item["metadata"],
                "hybrid_score": hybrid,
                "rerank_score": 0.0,
            })

        # Sort by hybrid score descending
        candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # Return at most TOP_K_RETRIEVAL candidates for reranking
        return candidates[:TOP_K_RETRIEVAL]

    def _rerank(
        self, query: str, candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Use the cross-encoder to assign a relevance score to each candidate.

        The cross-encoder sees (query, document) pairs and returns a single
        relevance score.  This is slower than bi-encoder search but much more
        accurate, making it ideal as a second-stage filter.
        """
        if not candidates:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.reranker.predict(pairs)   # returns a numpy array

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates
