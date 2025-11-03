from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydub import AudioSegment
from pydub.utils import which


def frames_to_ms(frames: int, fps: int) -> int:
    """Convert a number of frames to milliseconds given FPS.

    Args:
        frames: Number of frames.
        fps: Frames per second.

    Returns:
        Duration in milliseconds (rounded to nearest integer).
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return int(round((frames / float(fps)) * 1000.0))


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists.

    Args:
        path: Directory path to create if missing.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def read_yaml(path: str) -> dict[str, Any]:
    """Read a YAML file into a dict.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed YAML dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def safe_export(seg: AudioSegment, out_path: str, cfg: dict) -> None:
    """Export an AudioSegment with safety checks for ffmpeg.

    Supports exporting to mp3/wav. If exporting to MP3 and ffmpeg is missing,
    raises a clear error with install guidance for static ffmpeg builds and PATH.

    Args:
        seg: Audio segment to export.
        out_path: Output file path.
        cfg: Config dict, expects keys: export_bitrate, export_format.

    Raises:
        RuntimeError: If ffmpeg is required but not found.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    export_format = (cfg.get("export_format") or out.suffix.replace(".", "")).lower()
    bitrate = cfg.get("export_bitrate", "192k")

    if export_format == "mp3":
        if which("ffmpeg") is None:
            raise RuntimeError(
                "FFmpeg is required to export MP3 but was not found in PATH.\n"
                "Install a static build from: https://www.gyan.dev/ffmpeg/builds/ \n"
                "1) Download an Essentials or Full static build (Windows).\n"
                "2) Extract and add the 'bin' folder (containing ffmpeg.exe) to your PATH.\n"
                "3) Restart your terminal and try again."
            )
        seg.export(str(out), format="mp3", bitrate=str(bitrate))
        return

    if export_format == "wav":
        seg.export(str(out), format="wav")
        return

    # Fallback to extension-based export with ffmpeg check
    if which("ffmpeg") is None:
        raise RuntimeError(
            f"FFmpeg is required to export format '{export_format}' but was not found in PATH.\n"
            "Install a static build from: https://www.gyan.dev/ffmpeg/builds/ and add to PATH."
        )
    seg.export(str(out), format=export_format)

