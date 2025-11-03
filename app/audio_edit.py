from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Dict, Optional

from pydub import AudioSegment

from .utils import frames_to_ms


def _rms(seg: AudioSegment) -> float:
    """Compute RMS of a segment.

    Returns:
        RMS amplitude (linear). 0 for empty segments.
    """
    if len(seg) <= 0:
        return 0.0
    try:
        return float(seg.rms)
    except Exception:
        return 0.0


def _window_slice(src: AudioSegment, center_ms: int, half_win_ms: int) -> AudioSegment:
    start = max(0, int(center_ms - half_win_ms))
    end = min(len(src), int(center_ms + half_win_ms))
    if end <= start:
        end = min(len(src), start + max(1, half_win_ms * 2))
    return src[start:end]


def _snap_boundary(src: AudioSegment, target_ms: int, win_ms: int, step_ms: int) -> int:
    """Snap a boundary near target_ms to the lowest-RMS valley within ±win_ms.

    Args:
        src: Source audio.
        target_ms: Target boundary in milliseconds.
        win_ms: Half-window search radius.
        step_ms: Step size in milliseconds.

    Returns:
        Snapped boundary position in milliseconds.
    """
    best_ms = max(0, min(int(target_ms), len(src)))
    best_rms = float("inf")
    half_eval_win = max(5, step_ms * 2)
    start = max(0, target_ms - win_ms)
    end = min(len(src), target_ms + win_ms)
    for t in range(int(start), int(end) + 1, max(1, int(step_ms))):
        wseg = _window_slice(src, t, half_eval_win)
        r = _rms(wseg)
        if r < best_rms:
            best_rms = r
            best_ms = t
    return best_ms


def _find_true_silence_end(
    src: AudioSegment,
    raw_end_ms: int,
    max_seek_ms: int = 200,
    step_ms: int = 5,
) -> int:
    """Seek forward from raw_end_ms to the earliest stable-silence point.

    We search up to max_seek_ms ahead, choosing the earliest point where a
    small window (≈30ms) remains low-RMS and stable (plateau). If none, pick
    the minimal-RMS position.
    """
    eval_half_win = 15  # ~30ms eval window
    search_start = int(raw_end_ms)
    search_end = min(len(src), search_start + int(max_seek_ms))

    # Baseline RMS near boundary (before end)
    pre_seg = src[max(0, search_start - 100):search_start]
    baseline = max(1.0, _rms(pre_seg))
    # dynamic threshold: 15% of baseline (clamped)
    thr = max(5.0, baseline * 0.15)

    best_ms = search_start
    best_r = float("inf")

    for t in range(search_start, search_end + 1, max(1, int(step_ms))):
        w1 = _window_slice(src, t, eval_half_win)
        r1 = _rms(w1)
        if r1 < best_r:
            best_r = r1
            best_ms = t
        # Check stability: next window also quiet
        t2 = min(len(src), t + eval_half_win)
        w2 = _window_slice(src, t2, eval_half_win)
        r2 = _rms(w2)
        if r1 <= thr and r2 <= thr:
            return t
    return best_ms


@dataclass
class Group:
    start_ms: int
    end_ms: int
    last_word_index: int


def _build_groups_by_period(words: List[Dict], punct_map: Dict[int, str], gap_merge_ms: Optional[int] = None) -> List[Group]:
    """Build groups split strictly by '.' punctuation.

    If there is no period in punct_map, fallback to gap-based grouping.
    """
    period_indices = sorted([i for i, p in punct_map.items() if p == "."])
    if not period_indices:
        return _group_words_by_gap(words, gap_merge_ms or 180)

    groups: List[Group] = []
    cur_start = 0
    prev_end_idx = -1
    for idx in period_indices:
        if idx < 0 or idx >= len(words):
            continue
        # group from prev_end_idx+1 .. idx
        first_w = words[prev_end_idx + 1]
        last_w = words[idx]
        start_ms = int(round(first_w["start"] * 1000))
        end_ms = int(round(last_w["end"] * 1000))
        groups.append(Group(start_ms=start_ms, end_ms=end_ms, last_word_index=idx))
        prev_end_idx = idx

    # Tail after last period, if any remaining words
    if prev_end_idx < len(words) - 1:
        first_w = words[prev_end_idx + 1]
        last_w = words[-1]
        groups.append(
            Group(
                start_ms=int(round(first_w["start"] * 1000)),
                end_ms=int(round(last_w["end"] * 1000)),
                last_word_index=len(words) - 1,
            )
        )
    return groups


def _group_words_by_gap(words: List[Dict], gap_merge_ms: int) -> List[Group]:
    groups: List[Group] = []
    if not words:
        return groups

    cur_start_idx = 0
    for i in range(len(words) - 1):
        cur_end_ms = int(round(words[i]["end"] * 1000))
        next_start_ms = int(round(words[i + 1]["start"] * 1000))
        gap = next_start_ms - cur_end_ms
        if gap > gap_merge_ms:
            # end current group at i
            start_ms = int(round(words[cur_start_idx]["start"] * 1000))
            end_ms = int(round(words[i]["end"] * 1000))
            groups.append(Group(start_ms=start_ms, end_ms=end_ms, last_word_index=i))
            cur_start_idx = i + 1

    # close last group
    start_ms = int(round(words[cur_start_idx]["start"] * 1000))
    end_ms = int(round(words[-1]["end"] * 1000))
    groups.append(Group(start_ms=start_ms, end_ms=end_ms, last_word_index=len(words) - 1))
    return groups


def _silence_after_group(
    idx: int,
    groups: List[Group],
    punct_map: Dict[int, str],
    cfg: Dict,
) -> int:
    """Determine silence to add after a group based on punctuation rules."""
    last_idx = groups[idx].last_word_index
    punct = punct_map.get(last_idx, None)
    if punct == ".":
        frames = int(cfg.get("silence_after_period_frames", 24))
        fps = int(cfg.get("frame_rate", 30))
        return frames_to_ms(frames, fps)

    # Other punctuation: , ? ! : ;
    if punct in {",", "?", "!", ":", ";"}:
        # natural gap between this group end and next group start
        if idx < len(groups) - 1:
            cur_end = groups[idx].end_ms
            nxt_start = groups[idx + 1].start_ms
            natural_gap = max(0, nxt_start - cur_end)
        else:
            natural_gap = 0

        min_gap = int(cfg.get("comma_soft_break_min_gap_ms", 250))
        if natural_gap < min_gap:
            min_f = int(cfg.get("silence_other_punct_frames_min", 5))
            max_f = int(cfg.get("silence_other_punct_frames_max", 7))
            frames = random.randint(min_f, max_f)
            fps = int(cfg.get("frame_rate", 30))
            ms = frames_to_ms(frames, fps)
            if ms < int(cfg.get("min_gap_between_groups_ms", 60)):
                return 0
            return ms
        return 0

    return 0


def synthesize_from_words(
    audio_path: str,
    words: List[Dict],
    punct_map: Dict[int, str],
    cfg: Dict,
    out_path: str,
) -> None:
    """Synthesize edited audio from aligned words and punctuation rules.

    Steps per group:
    - Build groups split by period; fallback to gap-based grouping.
    - Apply pre/post padding, then snap boundaries to local RMS valleys.
    - For groups ending with '.', seek forward to true silence end before inserting 24 frames.
    - Concatenate segments; if boundaries touch/overlap after snapping, append with crossfade.
    - Insert silence between groups according to rules.
    - Export via safe_export handled by caller.
    """
    from .utils import safe_export  # local import to avoid cycles

    src = AudioSegment.from_file(audio_path)
    gap_merge_ms = int(cfg.get("gap_merge_ms", 180))
    groups = _build_groups_by_period(words, punct_map, gap_merge_ms)

    pre_pad = int(cfg.get("pre_pad_ms", 40))
    post_pad = int(cfg.get("post_pad_ms", 80))
    snap_win = int(cfg.get("snap_window_ms", 25))
    snap_step = int(cfg.get("snap_step_ms", 5))
    crossfade_ms = int(cfg.get("overlap_crossfade_ms", 25))

    output = AudioSegment.silent(duration=0)
    last_adj_end_src: Optional[int] = None

    for gi, g in enumerate(groups):
        # Apply padding in source timeline
        raw_start = max(0, g.start_ms - pre_pad)
        raw_end = min(len(src), g.end_ms + post_pad)

        # Snap to local valleys
        adj_start = _snap_boundary(src, raw_start, win_ms=snap_win, step_ms=snap_step)
        adj_end = _snap_boundary(src, raw_end, win_ms=snap_win, step_ms=snap_step)
        if adj_end <= adj_start:
            adj_end = min(len(src), adj_start + 10)

        # If period, seek to true silence end for end boundary
        if punct_map.get(g.last_word_index) == ".":
            adj_end = _find_true_silence_end(src, adj_end, max_seek_ms=200, step_ms=5)

        seg = src[adj_start:adj_end]

        # Append with crossfade if overlap with previous adjusted end in source timeline
        if last_adj_end_src is not None and adj_start <= last_adj_end_src:
            output = output.append(seg, crossfade=crossfade_ms)
        else:
            output += seg
        last_adj_end_src = adj_end

        # Add silence after group except last
        if gi < len(groups) - 1:
            add_ms = _silence_after_group(gi, groups, punct_map, cfg)
            if add_ms > 0:
                output += AudioSegment.silent(duration=add_ms)

    # Export via utility
    safe_export(output, out_path, cfg)

