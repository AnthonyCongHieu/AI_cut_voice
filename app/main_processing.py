from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `app.*` imports work when running from app/ directory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.utils import read_yaml, ensure_dir
from app.text_normalize import extract_text_from_docx, normalize_text
from app.aligner import align_words, save_aligned_json
from app.prosody_infer import build_punctuation_indices
from app.audio_edit import synthesize_from_words


def _read_text(docx_path: str | None) -> str:
    if docx_path:
        return extract_text_from_docx(docx_path)
    # Read from stdin if available
    if not sys.stdin.isatty():
        return sys.stdin.read()
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Voice Aligner CLI")
    ap.add_argument("--audio", required=True, help="Path to input audio (.wav/.mp3)")
    ap.add_argument("--docx", required=False, help="Path to transcript .docx (optional)")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    ap.add_argument("--out_json", required=True, help="Path to write aligned words JSON")
    ap.add_argument("--out_audio", required=True, help="Path to write synthesized audio")
    ap.add_argument("--model", required=False, default=None, help="Override model name (default from config)")
    ap.add_argument("--device", required=False, default=None, help="Override device (auto/cuda/cpu)")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    language = cfg.get("language", "vi")
    model_name = args.model or cfg.get("whisper_model", "large-v3")
    device = args.device or cfg.get("device", "auto")

    print("[cli] Loading transcript...")
    raw_text = _read_text(args.docx)
    text = normalize_text(raw_text, lang=language)
    if not text:
        print("[cli] Warning: Transcript is empty. Alignment quality may be degraded.")

    print("[cli] Aligning words (stable-ts)...")
    words = []
    try:
        words = align_words(args.audio, text, language=language, model_name=model_name, device=device)
    except Exception as e:
        print(f"[cli] Error during alignment: {e}")

    if not words:
        print("[cli] Warning: No words aligned. Skipping synthesis.")
        save_aligned_json(words, args.out_json)
        return

    print("[cli] Building punctuation indices...")
    punct_map = build_punctuation_indices(text, words)

    print("[cli] Synthesizing audio with safe cuts and pauses...")
    synthesize_from_words(args.audio, words, punct_map, cfg, args.out_audio)

    print("[cli] Saving aligned words JSON...")
    save_aligned_json(words, args.out_json)
    print("[cli] Done.")


if __name__ == "__main__":
    main()
