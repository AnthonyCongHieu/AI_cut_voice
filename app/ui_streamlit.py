from __future__ import annotations

import io
import json
from pathlib import Path
import sys

import streamlit as st

# Ensure project root is on sys.path so `app.*` imports work when running from app/ directory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.utils import read_yaml
from app.text_normalize import extract_text_from_docx, normalize_text
from app.aligner import align_words
from app.prosody_infer import build_punctuation_indices
from app.audio_edit import synthesize_from_words


st.set_page_config(page_title="Voice Aligner", layout="centered")
st.title("Voice Aligner — stable-ts + Streamlit")


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
    model_name = st.selectbox("Model", ["large-v3", "large", "medium", "small", "base"], index=0)
    device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)

    st.header("Overrides")
    cfg = _load_default_config()
    frame_rate = st.number_input("frame_rate", min_value=1, max_value=120, value=int(cfg.get("frame_rate", 30)))
    silence_after_period_frames = st.number_input(
        "silence_after_period_frames", min_value=0, max_value=120, value=int(cfg.get("silence_after_period_frames", 24))
    )
    pre_pad_ms = st.number_input("pre_pad_ms", min_value=0, max_value=1000, value=int(cfg.get("pre_pad_ms", 40)))
    post_pad_ms = st.number_input("post_pad_ms", min_value=0, max_value=2000, value=int(cfg.get("post_pad_ms", 80)))
    snap_window_ms = st.number_input("snap_window_ms", min_value=0, max_value=200, value=int(cfg.get("snap_window_ms", 25)))
    snap_step_ms = st.number_input("snap_step_ms", min_value=1, max_value=50, value=int(cfg.get("snap_step_ms", 5)))
    overlap_crossfade_ms = st.number_input(
        "overlap_crossfade_ms", min_value=0, max_value=200, value=int(cfg.get("overlap_crossfade_ms", 25))
    )
    comma_soft_break_min_gap_ms = st.number_input(
        "comma_soft_break_min_gap_ms", min_value=0, max_value=1000, value=int(cfg.get("comma_soft_break_min_gap_ms", 250))
    )

    # Apply overrides
    cfg.update(
        {
            "frame_rate": int(frame_rate),
            "silence_after_period_frames": int(silence_after_period_frames),
            "pre_pad_ms": int(pre_pad_ms),
            "post_pad_ms": int(post_pad_ms),
            "snap_window_ms": int(snap_window_ms),
            "snap_step_ms": int(snap_step_ms),
            "overlap_crossfade_ms": int(overlap_crossfade_ms),
            "comma_soft_break_min_gap_ms": int(comma_soft_break_min_gap_ms),
            "whisper_model": model_name,
            "device": device,
        }
    )

st.divider()

run = st.button("Bắt đầu xử lý", type="primary")

if run:
    if not audio_file:
        st.error("Vui lòng upload audio.")
    else:
        with st.spinner("Đang xử lý... (align → tái tạo ngắt nghỉ)"):
            # Persist uploaded audio to a temp file for processing
            tmp_audio_path = Path(st.session_state.get("tmp_audio_path", "_tmp_input_audio"))
            tmp_audio_path.write_bytes(audio_file.read())

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

            # Align
            try:
                words = align_words(
                    str(tmp_audio_path),
                    transcript,
                    language=cfg.get("language", "vi"),
                    model_name=cfg.get("whisper_model", "large-v3"),
                    device=cfg.get("device", "auto"),
                )
            except Exception as e:
                st.error(f"Lỗi khi align: {e}")
                words = []

            if not words:
                st.warning("Không align được từ nào.")
            else:
                punct_map = build_punctuation_indices(transcript, words)
                out_audio_path = Path("data/outputs/output.mp3")
                out_audio_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    synthesize_from_words(str(tmp_audio_path), words, punct_map, cfg, str(out_audio_path))
                except Exception as e:
                    st.error(f"Lỗi khi tổng hợp audio: {e}")
                else:
                    st.success("Hoàn tất!")
                    audio_bytes = out_audio_path.read_bytes()
                    st.audio(io.BytesIO(audio_bytes), format="audio/mpeg")
                    st.download_button(
                        label="Download output.mp3",
                        data=audio_bytes,
                        file_name="output.mp3",
                        mime="audio/mpeg",
                    )
                    # Preview words
                    st.caption(f"Tổng số từ align: {len(words)}")
                    preview = words[:30]
                    st.code(json.dumps(preview, ensure_ascii=False, indent=2))
