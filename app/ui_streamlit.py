from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
import sys

import librosa
import streamlit as st
import torch

# Ensure project root is on sys.path so `app.*` imports work when running from app/ directory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.utils import read_yaml
from app.text_normalize import extract_text_from_docx, normalize_text
from app.aligner import align_words
from app.main_processing import detect_targets, seconds_to_hhmmss


st.set_page_config(page_title="Voice Aligner", layout="centered")
st.title("Voice Aligner — stable-ts + Streamlit")
if 'targets' not in st.session_state:
    st.session_state.targets = [""]
if 'to_delete' not in st.session_state:
    st.session_state['to_delete'] = None
if 'cancel_requested' not in st.session_state:
    st.session_state.cancel_requested = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0.0
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'estimated_total' not in st.session_state:
    st.session_state.estimated_total = 0.0
if 'aligned_words' not in st.session_state:
    st.session_state.aligned_words = []
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None

def get_estimate_multiplier(model: str) -> float:
    """Get processing time multiplier based on model (real time * multiplier)."""
    estimates = {
        "base": 1.5,
        "small": 2.0,
        "medium": 3.0,
        "large": 4.0,
        "large-v3": 4.5
    }
    return estimates.get(model, 4.0)

def process_alignment(audio_path: str, transcript: str, cfg: dict, result_queue: queue.Queue):
    """Run alignment in a separate thread."""
    try:
        words = align_words(
            audio_path,
            transcript,
            language=cfg.get("language", "vi"),
            model_name=cfg.get("whisper_model", "large-v3"),
            device=cfg.get("device", "auto"),
        )
        result_queue.put({'words': words, 'error': None})
    except Exception as e:
        result_queue.put({'words': None, 'error': str(e)})

# Handle deletion
if st.session_state['to_delete'] is not None:
    idx = st.session_state['to_delete']
    if 0 <= idx < len(st.session_state.targets):
        st.session_state.targets.pop(idx)
    st.session_state['to_delete'] = None


def _load_default_config() -> dict:
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        return read_yaml(str(cfg_path))
    return {
        "frame_rate": 30,
        "silence_after_period_frames": 24,
        "pre_pad_ms": 40,
        "post_pad_ms": 80,
        "snap_window_ms": 25,
        "snap_step_ms": 5,
        "overlap_crossfade_ms": 25,
        "comma_soft_break_min_gap_ms": 250,
        "export_bitrate": "192k",
        "export_format": "mp3",
        "language": "vi",
        "whisper_model": "large-v3",
        "device": "auto",
    }


with st.sidebar:
    st.header("Inputs")
    audio_file = st.file_uploader("Upload audio (.wav/.mp3)", type=["wav", "mp3"])
    docx_file = st.file_uploader("Upload .docx (optional)", type=["docx"])
    text_area = st.text_area("Or paste text", height=150)

    st.header("Model & Device")
    model_name = st.selectbox("Model", ["base", "small", "medium", "large", "large-v3"], index=4)
    device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)

    st.header("Performance Tips")
    if model_name in ["large", "large-v3"]:
        st.info("💡 Use smaller models (medium/small/base) for faster processing.")
    if device == "cpu":
        if torch.cuda.is_available():
            st.info("💡 Switch to CUDA for GPU acceleration.")
        else:
            st.warning("⚠️ CUDA not available. Consider installing CUDA-compatible PyTorch.")

    st.header("Overrides")
    cfg = _load_default_config()
    # frame_rate = st.number_input("frame_rate", min_value=1, max_value=120, value=int(cfg.get("frame_rate", 30)))
    # silence_after_period_frames = st.number_input(
    #     "silence_after_period_frames", min_value=0, max_value=120, value=int(cfg.get("silence_after_period_frames", 24))
    # )
    # pre_pad_ms = st.number_input("pre_pad_ms", min_value=0, max_value=1000, value=int(cfg.get("pre_pad_ms", 40)))
    # post_pad_ms = st.number_input("post_pad_ms", min_value=0, max_value=2000, value=int(cfg.get("post_pad_ms", 80)))
    # snap_window_ms = st.number_input("snap_window_ms", min_value=0, max_value=200, value=int(cfg.get("snap_window_ms", 25)))
    # snap_step_ms = st.number_input("snap_step_ms", min_value=1, max_value=50, value=int(cfg.get("snap_step_ms", 5)))
    # overlap_crossfade_ms = st.number_input(
    #     "overlap_crossfade_ms", min_value=0, max_value=200, value=int(cfg.get("overlap_crossfade_ms", 25))
    # )
    # comma_soft_break_min_gap_ms = st.number_input(
    #     "comma_soft_break_min_gap_ms", min_value=0, max_value=1000, value=int(cfg.get("comma_soft_break_min_gap_ms", 250))
    # )

    # Apply overrides
    cfg.update(
        {
            "whisper_model": model_name,
            "device": device,
        }
    )

st.divider()

st.header("Targets")

targets_text = st.text_area("Paste targets (one per line)", height=100)

if st.button("Load Targets"):
    lines = targets_text.split('\n')
    filtered = [line.strip() for line in lines if line.strip()]
    st.session_state.targets = filtered

# Grid layout for targets
num_cols = 3
cols = st.columns(num_cols)
for i in range(len(st.session_state.targets)):
    col_idx = i % num_cols
    with cols[col_idx]:
        col_input, col_button = st.columns([4, 1])
        with col_input:
            st.session_state.targets[i] = st.text_input(f"Target {i+1}", value=st.session_state.targets[i], key=f"target_{i}", label_visibility="collapsed")
        with col_button:
            if st.button("🗑️", key=f"delete_{i}", help="Delete this target"):
                st.session_state['to_delete'] = i

if st.button("Add Target"):
    st.session_state.targets.append("")

col1, col2 = st.columns(2)
with col1:
    run = st.button("Bắt đầu xử lý", type="primary")
with col2:
    cancel = st.button("Cancel")

if cancel:
    st.session_state.cancel_requested = True
    st.info("Cancellation requested. The process may take a moment to stop.")

if run:
    if not audio_file:
        st.error("Vui lòng upload audio.")
    else:
        # Persist uploaded audio to a temp file for processing
        tmp_audio_path = Path(st.session_state.get("tmp_audio_path", "_tmp_input_audio"))
        tmp_audio_path.write_bytes(audio_file.read())

        # Get audio duration
        try:
            duration = librosa.get_duration(path=str(tmp_audio_path))
        except Exception as e:
            st.error(f"Could not get audio duration: {e}")
            st.stop()

        # Load transcript
        transcript = ""
        if docx_file is not None:
            # Save to temp and read via extractor
            tmp_docx_path = Path(st.session_state.get("tmp_docx_path", "_tmp_input.docx"))
            tmp_docx_path.write_bytes(docx_file.read())
            transcript = extract_text_from_docx(str(tmp_docx_path))
        elif text_area.strip():
            transcript = text_area

        transcript = normalize_text(transcript, lang=cfg.get("language", "vi"))

        # Estimate time
        multiplier = get_estimate_multiplier(cfg.get("whisper_model", "large-v3"))
        estimated_total = duration * multiplier
        st.session_state.estimated_total = estimated_total
        st.session_state.start_time = time.time()
        st.session_state.processing = True
        st.session_state.progress = 0.0

        # Start processing thread
        result_queue = queue.Queue()
        thread = threading.Thread(target=process_alignment, args=(str(tmp_audio_path), transcript, cfg, result_queue))
        thread.start()

        # Progress display
        progress_placeholder = st.empty()
        elapsed_placeholder = st.empty()
        remaining_placeholder = st.empty()

        while st.session_state.processing:
            if st.session_state.cancel_requested:
                st.session_state.processing = False
                st.warning("Processing cancelled.")
                break
            # Check for results from thread
            if not result_queue.empty():
                result = result_queue.get()
                st.session_state.aligned_words = result['words']
                st.session_state.processing_error = result['error']
                st.session_state.processing = False
                break
            elapsed = time.time() - st.session_state.start_time
            progress = min(elapsed / st.session_state.estimated_total, 1.0)
            st.session_state.progress = progress
            remaining = max(st.session_state.estimated_total - elapsed, 0)
            elapsed_hms = seconds_to_hhmmss(elapsed, st.session_state.estimated_total)
            remaining_hms = seconds_to_hhmmss(remaining, st.session_state.estimated_total)
            progress_placeholder.progress(progress)
            elapsed_placeholder.text(f"Elapsed: {elapsed_hms}")
            remaining_placeholder.text(f"Estimated remaining: {remaining_hms}")
            time.sleep(1)

        # After processing
        if st.session_state.processing_error:
            st.error(f"Processing error: {st.session_state.processing_error}")
        else:
            words = st.session_state.aligned_words
            if not words:
                st.warning("Không align được từ nào.")
            else:
                max_end = max((w['end'] for w in words), default=0)
                targets = [t.strip() for t in st.session_state.targets if t.strip()]
                if targets:
                    results = detect_targets(words, targets)
                    timeline_lines = []
                    for target, res in results.items():
                        if res == "Not found":
                            timeline_lines.append(f'- "{target}" not found')
                        else:
                            for match in res:
                                start_hms = seconds_to_hhmmss(match['start'], max_end)
                                end_hms = seconds_to_hhmmss(match['end'], max_end)
                                timeline_lines.append(f'- {start_hms} - {end_hms}: "{target}"')
                    total_found = sum(1 for res in results.values() if res != "Not found")
                    total_not_found = len(results) - total_found
                    timeline_lines.append(f"Total found: {total_found}")
                    timeline_lines.append(f"Total not found: {total_not_found}")
                    timeline_str = "\n".join(timeline_lines)
                    st.code(timeline_str)
                    st.download_button(
                        label="Download Timeline",
                        data=timeline_str,
                        file_name="timeline.txt",
                        mime="text/plain",
                    )
                else:
                    st.info("Alignment completed. No targets provided for detection.")
                # Preview words
                st.caption(f"Tổng số từ align: {len(words)}")
                preview = words[:30]
                st.code(json.dumps(preview, ensure_ascii=False, indent=2))

