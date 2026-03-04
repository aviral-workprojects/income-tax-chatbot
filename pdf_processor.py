"""
pdf_processor.py
----------------
Handles everything related to the PDF knowledge base:
  1. Extract raw text from each page using pdfplumber.
  2. Clean the extracted text (remove noise, normalise whitespace).
  3. Split the text into overlapping word-based chunks while preserving
     legal section structure as much as possible.
  4. Return chunks as a list of dicts with rich metadata.
"""

import re
import logging
from typing import List, Dict, Any

import pdfplumber

from config import (
    PDF_PATH,
    CHUNK_SIZE_WORDS,
    CHUNK_OVERLAP_WORDS,
    DOCUMENT_NAME,
)

# Set up a simple logger so students can see what is happening
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for detecting section headings in the Income Tax Act
# Examples: "Section 80C", "Sec. 44AD", "S. 10(10D)", "SECTION 194"
# ---------------------------------------------------------------------------
SECTION_PATTERN = re.compile(
    r"(?:section|sec\.?|s\.)\s*(\d+[A-Z]{0,3}(?:\(\w+\))*)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_process_pdf() -> List[Dict[str, Any]]:
    """
    Main entry point.  Reads the PDF, extracts text page by page, cleans it,
    and returns a flat list of chunk dictionaries.

    Each chunk dict has the following keys:
        text        : str   – the chunk content
        page        : int   – 1-based page number where the chunk starts
        chunk_index : int   – global sequential index of this chunk
        doc_name    : str   – human-readable document name
        section     : str   – detected section number, or "" if none found
    """
    logger.info(f"Loading PDF from: {PDF_PATH}")

    pages = _extract_pages(PDF_PATH)
    logger.info(f"Extracted text from {len(pages)} pages.")

    chunks = _build_chunks(pages)
    logger.info(f"Created {len(chunks)} chunks from the document.")

    return chunks


# ---------------------------------------------------------------------------
# Step 1 – PDF text extraction
# ---------------------------------------------------------------------------

def _extract_pages(path: str) -> List[Dict[str, Any]]:
    """
    Open the PDF with pdfplumber and extract text page by page.

    Returns a list of dicts:
        { "page": int, "text": str }
    """
    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""
            cleaned = _clean_text(raw_text)
            if cleaned.strip():          # skip blank / header-only pages
                pages.append({"page": i, "text": cleaned})
            if i % 100 == 0:
                logger.info(f"  Processed {i}/{total} pages …")
    return pages


# ---------------------------------------------------------------------------
# Step 2 – Text cleaning
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """
    Remove common PDF extraction artefacts and normalise whitespace.

    Operations performed:
      - Remove form-feed / carriage-return characters
      - Collapse runs of spaces into a single space
      - Remove lines that are purely numeric (page footers like "123")
      - Normalise unicode dashes and quotes to ASCII equivalents
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unicode control characters (but keep newlines and tabs)
    text = re.sub(r"[^\S\n\t ]+", " ", text)

    # Remove standalone page-number lines (a line containing only digits)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Collapse multiple consecutive blank lines into one
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Normalise unicode quotation marks and dashes to ASCII
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Collapse multiple spaces within a line
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


# ---------------------------------------------------------------------------
# Step 3 – Section-aware, overlapping word chunking
# ---------------------------------------------------------------------------

def _build_chunks(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Combine all page texts into a single word stream, then create
    overlapping fixed-size chunks.

    We also record:
      - which page each word came from (for metadata)
      - the first section heading found in each chunk

    Returns a list of chunk dicts (see load_and_process_pdf docstring).
    """
    # Build a flat list of (word, page_number) tuples
    word_page_pairs: List[tuple] = []
    for page_dict in pages:
        page_num = page_dict["page"]
        words = page_dict["text"].split()
        for word in words:
            word_page_pairs.append((word, page_num))

    total_words = len(word_page_pairs)
    logger.info(f"Total words in document: {total_words:,}")

    chunks = []
    chunk_index = 0
    start = 0

    while start < total_words:
        end = min(start + CHUNK_SIZE_WORDS, total_words)

        # Extract words and determine the starting page of this chunk
        chunk_pairs = word_page_pairs[start:end]
        chunk_words = [pair[0] for pair in chunk_pairs]
        chunk_text = " ".join(chunk_words)
        start_page = chunk_pairs[0][1] if chunk_pairs else 0

        # Try to detect a section heading inside this chunk
        section = _detect_section(chunk_text)

        chunks.append({
            "text": chunk_text,
            "page": start_page,
            "chunk_index": chunk_index,
            "doc_name": DOCUMENT_NAME,
            "section": section,
        })

        chunk_index += 1

        # Advance by (chunk_size - overlap) words for the next chunk
        step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS
        start += step

    return chunks


def _detect_section(text: str) -> str:
    """
    Search the chunk text for the first Income Tax Act section reference.

    Returns the section number as a string (e.g. "80C", "10", "44AD"),
    or an empty string if no section is found.
    """
    match = SECTION_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return ""
