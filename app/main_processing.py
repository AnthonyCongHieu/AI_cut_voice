from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is on sys.path so `app.*` imports work when running from app/ directory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.utils import read_yaml, ensure_dir
from app.text_normalize import extract_text_from_docx, normalize_text
from app.aligner import align_words, save_aligned_json


def seconds_to_hhmmss(seconds: float, max_seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format based on max_seconds."""
    total_seconds = round(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if max_seconds < 3600:
        return f"{minutes:02d}:{secs:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def detect_targets(words: list[dict], targets: list[str]) -> dict:
    """Detect timestamps for target words or phrases in aligned words list.

    Args:
        words: List of dicts with 'word', 'start', 'end' in seconds.
        targets: List of target strings, each can be single or multi-word.

    Returns:
        Dict with target as key, list of merged {"start": float, "end": float} or "Not found".
    """
    results = {}
    for target in targets:
        target_words = target.split()  # split by space for multi-word
        matches = []
        n = len(target_words)
        for i in range(len(words) - n + 1):
            # Case-sensitive match
            aligned_words = [w['word'] for w in words[i:i+n]]
            if aligned_words == target_words:
                start = words[i]['start']
                end = words[i+n-1]['end']
                matches.append({"start": start, "end": end})

        if not matches:
            results[target] = "Not found"
            continue

        # Sort matches by start time
        matches.sort(key=lambda x: x['start'])

        # Merge consecutive occurrences within 60 seconds
        merged = []
        current = matches[0]
        for match in matches[1:]:
            if current['end'] + 60 >= match['start']:
                # Merge: extend the end
                current['end'] = max(current['end'], match['end'])
            else:
                # No merge, add current to merged, start new current
                merged.append(current)
                current = match
        # Add the last current
        merged.append(current)

        results[target] = merged
    return results


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
    transcript_ok = bool(text)
    if not text:
        print("[cli] Error: Transcript is missing. Cannot generate timeline.")

    print("[cli] Aligning words (stable-ts)...")
    words = []
    try:
        words = align_words(args.audio, text, language=language, model_name=model_name, device=device)
    except Exception as e:
        print(f"[cli] Error during alignment: {e}")

    alignment_ok = bool(words)

    if not words:
        print("[cli] Warning: No words aligned. Skipping synthesis.")
        save_aligned_json(words, args.out_json)
        return

    max_end = max((w['end'] for w in words), default=0)

    # Detect targets if specified and conditions met
    targets = cfg.get("targets", [])
    if targets and transcript_ok and alignment_ok:
        detection_results = detect_targets(words, targets)
        print("Timeline for Audio:")
        for target, res in detection_results.items():
            if res == "Not found":
                print(f'- "{target}" not found')
            else:
                for match in res:
                    start_hms = seconds_to_hhmmss(match['start'], max_end)
                    end_hms = seconds_to_hhmmss(match['end'], max_end)
                    print(f'- {start_hms} - {end_hms}: "{target}"')
        total_found = sum(1 for res in detection_results.values() if res != "Not found")
        total_not_found = len(detection_results) - total_found
        print(f"Total found: {total_found}")
        print(f"Total not found: {total_not_found}")
        save_aligned_json(words, args.out_json)
        return
    elif targets:
        if not transcript_ok:
            print("[cli] Skipping timeline due to missing transcript.")
        elif not alignment_ok:
            print("[cli] Skipping timeline due to alignment failure.")

    print("[cli] Saving aligned words JSON...")
    save_aligned_json(words, args.out_json)
    print("[cli] Done.")


if __name__ == "__main__":
    main()
