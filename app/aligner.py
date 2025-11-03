from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Dict

import torch
import stable_whisper


FALLBACK_MODELS = [
    "large-v3",
    "large",
    "medium",
    "small",
    "base",
]


def _select_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model_smart(name: str, device: str) -> Any:
    """Load a stable-ts model with device auto-selection and OOM fallback.

    Attempts to load the requested model, then falls back through a list on failure
    (OOM or other RuntimeError).
    """
    dev = _select_device(device)
    tried: list[str] = []
    for m in FALLBACK_MODELS:
        if name and m != name and FALLBACK_MODELS.index(m) < FALLBACK_MODELS.index(name):
            # Skip fallbacks before requested model
            continue
        try:
            print(f"[aligner] Loading model '{m}' on device '{dev}'...")
            model = stable_whisper.load_model(m, device=dev)
            print(f"[aligner] Loaded model: {m}")
            return model
        except RuntimeError as e:
            tried.append(m)
            msg = str(e).lower()
            print(f"[aligner] Failed to load '{m}': {e}")
            if "out of memory" in msg or "cuda" in msg:
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                continue
        except Exception as e:  # generic fallback
            tried.append(m)
            print(f"[aligner] Error loading '{m}': {e}")
            continue
    raise RuntimeError(f"Could not load any Whisper model. Tried: {', '.join(tried)}")


def align_words(
    audio_path: str,
    transcript: str,
    language: str = "vi",
    model_name: str = "large-v3",
    device: str = "auto",
) -> List[Dict[str, float]]:
    """Align word-level timestamps using stable-ts.

    Uses stable-ts (stable_whisper) transcription with word_timestamps enabled.
    The provided transcript is used as an initial_prompt to bias recognition.

    Returns list of dict: {"word": str, "start": float, "end": float} in seconds.
    """
    model = load_model_smart(model_name, device)
    # Bias with initial_prompt if available. condition_on_previous_text=False for determinism.
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        initial_prompt=(transcript or None),
        condition_on_previous_text=False,
        vad=True,
    )

    words: List[Dict[str, float]] = []
    for seg in result.segments:
        seg_words = getattr(seg, "words", None)
        if not seg_words:
            # fallback: use segment text as a single word
            words.append({
                "word": seg.text.strip(),
                "start": float(seg.start),
                "end": float(seg.end),
            })
            continue
        for w in seg_words:
            # Some words may not have both start/end; skip incomplete ones
            ws = getattr(w, "start", None)
            we = getattr(w, "end", None)
            wt = getattr(w, "word", None) or getattr(w, "text", None)
            if ws is None or we is None or wt is None:
                continue
            words.append({
                "word": str(wt).strip(),
                "start": float(ws),
                "end": float(we),
            })
    return words


def save_aligned_json(words: List[Dict[str, float]], out_path: str) -> None:
    """Save aligned words to JSON file.

    Args:
        words: List of word dicts with start/end times in seconds.
        out_path: Output JSON file path.
    """
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(words, f, ensure_ascii=False, indent=2)

