from __future__ import annotations

import re
from typing import Dict, List


PUNCT_PATTERN = re.compile(r"([\.,\?\!\:\;])")


def _tokenize_with_punct(text: str) -> List[str]:
    # Split words and keep punctuation as separate tokens
    # Normalize whitespace to single spaces, but keep original punctuation
    text = re.sub(r"\s+", " ", text.strip())
    # Insert spaces around punctuation to split safely
    text = PUNCT_PATTERN.sub(r" \1 ", text)
    tokens = [t for t in text.split(" ") if t]
    return tokens


def build_punctuation_indices(text: str, words: List[dict]) -> Dict[int, str]:
    """Map the last word index before punctuation to that punctuation char.

    Priority: period '.' as hard group boundary; also record , ? ! : ;
    Strategy: iterate transcript tokens and align sequentially to aligned words
    by order. This assumes roughly consistent flow; if length mismatches, we
    clamp to available words.

    Args:
        text: The reference transcript text.
        words: Aligned words from the audio (list of dicts with 'word').

    Returns:
        Dict mapping last_word_index -> punctuation character string.
    """
    tokens = _tokenize_with_punct(text)
    punct_map: Dict[int, str] = {}
    w_idx = 0
    n_words = len(words)
    last_word_idx_seen = -1

    for tok in tokens:
        if len(tok) == 1 and tok in ",?!:;.":
            if last_word_idx_seen >= 0:
                punct_map[last_word_idx_seen] = tok
            continue
        # Token is a word; advance word index if possible
        if w_idx < n_words:
            last_word_idx_seen = w_idx
            w_idx += 1
        else:
            # No more words; still allow punctuation to map to final word
            last_word_idx_seen = n_words - 1

    return punct_map

