from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, List, Dict
from concurrent.futures import ThreadPoolExecutor

import torch
import yaml
from pydub import AudioSegment
from pydub.utils import which
import faster_whisper
from app.audio_chunking import split_audio_into_chunks


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
            model = faster_whisper.WhisperModel(m, device=dev, compute_type="float16" if dev == "cuda" else "int8")
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


def align_chunk(
    audio_chunk_path: str,
    offset: float,
    transcript: str,
    model,
    language: str = "vi",
) -> List[Dict[str, float]]:
    """Align word-level timestamps for a single audio chunk with offset adjustment.

    Args:
        audio_chunk_path: Path to the audio chunk file.
        offset: Time offset in seconds to add to all timestamps.
        transcript: Transcript text for initial prompt.
        model: Pre-loaded Whisper model.
        language: Language code.

    Returns:
        List of word dicts with adjusted start/end times.
    """
    # Reuse the alignment logic from align_words, but without preflight checks
    try:
        segments, _ = model.transcribe(
            audio_chunk_path,
            language=language,
            word_timestamps=True,
            initial_prompt=transcript,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        segments = list(segments)
        if not segments:
            return []
    except Exception as e:
        print(f"[align_chunk] Error processing chunk: {e}")
        return []

    words: List[Dict[str, float]] = []
    for seg in segments:
        seg_words = getattr(seg, "words", None)
        if not seg_words:
            words.append({
                "word": seg.text.strip(),
                "start": float(seg.start) + offset,
                "end": float(seg.end) + offset,
            })
            continue
        for w in seg_words:
            ws = getattr(w, "start", None)
            we = getattr(w, "end", None)
            wt = getattr(w, "word", None) or getattr(w, "text", None)
            if ws is None or we is None or wt is None:
                continue
            words.append({
                "word": str(wt).strip(),
                "start": float(ws) + offset,
                "end": float(we) + offset,
            })
    return words


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
    # Preflight checks: file existence and ffmpeg availability (ffprobe/ffmpeg used by Whisper I/O)
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if which("ffmpeg") is None:
        raise RuntimeError(
            "FFmpeg not found in PATH. Install a static build (Windows Essentials) from https://www.gyan.dev/ffmpeg/builds/ and add its 'bin' folder to PATH, or run scripts/activate_with_ffmpeg.ps1 before launching."
        )

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get audio duration
    audio = AudioSegment.from_file(audio_path)
    audio_duration = len(audio) / 1000  # in seconds

    if (
        config.get("enable_parallel_processing", False)
        and audio_duration > config.get("parallel_threshold_minutes", 10) * 60
    ):
        # Parallel processing
        chunk_duration_ms = config.get("chunk_duration_minutes", 5) * 60 * 1000
        overlap_ms = config.get("chunk_overlap_seconds", 1) * 1000
        max_workers = config.get("max_parallel_workers", 3)

        # Load model once
        model = load_model_smart(model_name, device)

        # Split audio into chunks
        chunks = split_audio_into_chunks(audio_path, chunk_duration_ms, overlap_ms)

        # Create temp files and offsets
        temp_files = []
        offsets = []
        current_offset_ms = 0
        for chunk in chunks:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_fd)
            chunk.export(temp_path, format="wav")
            temp_files.append(temp_path)
            offsets.append(current_offset_ms / 1000)  # offset in seconds
            current_offset_ms += len(chunk) - overlap_ms  # adjust for overlap

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    align_chunk,
                    temp_path,
                    offset,
                    transcript,
                    model,
                    language,
                )
                for temp_path, offset in zip(temp_files, offsets)
            ]
            word_lists = [future.result() for future in futures]

        # Merge word lists
        all_words = []
        for word_list in word_lists:
            all_words.extend(word_list)
        all_words.sort(key=lambda w: w["start"])

        # Clean up temp files
        for temp_path in temp_files:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_words

    else:
        # Original single-threaded processing
        used_device = _select_device(device)
        print(f"[aligner] Using device: {used_device}")
        model = load_model_smart(model_name, device)
        # Bias with initial_prompt if available. condition_on_previous_text=False for determinism.
        try:
            segments, info = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                initial_prompt=transcript,
                condition_on_previous_text=False,
                vad_filter=True,
            )
            segments = list(segments)
            if not segments:
                raise ValueError("No segments found")
        except FileNotFoundError as e:
            # Common on Windows when ffmpeg is missing; surface clear guidance
            raise RuntimeError(
                f"Audio decode failed (possibly missing FFmpeg). Original error: {e}.\n"
                "Install FFmpeg and ensure ffmpeg.exe is in PATH, or run scripts/activate_with_ffmpeg.ps1."
            )
        except OSError as e:
            if getattr(e, "winerror", None) == 2:
                raise RuntimeError(
                    "[WinError 2] External tool not found. FFmpeg is likely missing. Install from https://www.gyan.dev/ffmpeg/builds/ and add to PATH, or run scripts/activate_with_ffmpeg.ps1."
                )
            raise

        words: List[Dict[str, float]] = []
        for seg in segments:
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

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
