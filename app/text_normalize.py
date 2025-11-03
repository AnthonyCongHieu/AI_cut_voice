from __future__ import annotations

import re
from typing import Any

from docx import Document


def extract_text_from_docx(path: str) -> str:
    """Extract plain text from a .docx file.

    Args:
        path: Path to DOCX file.

    Returns:
        Combined paragraphs as a single string separated by newlines.
    """
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join([p for p in paragraphs if p is not None]).strip()


_multi_space_re = re.compile(r"[\t\f\r ]+")
_multi_newline_re = re.compile(r"\n{3,}")


def normalize_text(raw: str, lang: str = "vi") -> str:
    """Normalize input text minimally for alignment.

    MVP behavior: trim ends, collapse excessive whitespace, keep Vietnamese diacritics.

    Args:
        raw: Raw text input.
        lang: Language code (unused for MVP), default 'vi'.

    Returns:
        Normalized text.
    """
    if not raw:
        return ""
    text = raw.strip()
    # Collapse multiple spaces and tabs
    text = _multi_space_re.sub(" ", text)
    # Collapse 3+ newlines down to 2 newlines to preserve paragraph breaks
    text = _multi_newline_re.sub("\n\n", text)
    return text

