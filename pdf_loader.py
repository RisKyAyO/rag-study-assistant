"""
PDF text extraction using PyMuPDF.
"""
from typing import List, Dict
import fitz  # PyMuPDF


def extract_text_from_pdf(path: str) -> List[Dict]:
    """Extract text page by page, returning list of {page, text} dicts."""
    doc = fitz.open(path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num, "text": text})
    doc.close()
    return pages
