from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

from pydub import AudioSegment  # type: ignore

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
    sliced = src[start:end]  # type: ignore
    return sliced  # type: ignore


def find_silence_boundary(src: AudioSegment, target_ms: int, is_start: bool, cfg: Dict[str, Any]) -> int:
    """Robust boundary snapping using windowed RMS analysis to find silence valleys.

    Scans ±75ms around target for lowest RMS valley using window_ms=75, step_ms=5, ±15ms sub-window for RMS.
    Ensures boundaries never clip vocal audio by shifting only to valleys with RMS <=20% baseline.
    For end boundaries, if residual audio remains after snapping, shift forward until silence.
    """
    window_ms = 75  # Scan ±75ms around target
    step_ms = 5     # Step size for scanning
    sub_win_ms = 15  # ±15ms sub-window for RMS calculation
    threshold_percent = cfg.get("silence_threshold_rms_percent", 20)

    # Compute baseline RMS from a longer segment around target
    baseline_win_ms = 200  # Use 200ms around target for baseline
    baseline_start = max(0, target_ms - baseline_win_ms // 2)
    baseline_end = min(len(src), target_ms + baseline_win_ms // 2)
    baseline_seg = src[baseline_start:baseline_end]  # type: ignore
    baseline_rms = _rms(baseline_seg)  # type: ignore
    threshold_rms = baseline_rms * (threshold_percent / 100.0)

    # Scan range: ±75ms around target
    scan_start = max(0, target_ms - window_ms)
    scan_end = min(len(src), target_ms + window_ms)

    best_ms = target_ms
    best_rms = float("inf")

    for t in range(scan_start, scan_end + 1, step_ms):
        sub_seg = _window_slice(src, t, sub_win_ms)
        r = _rms(sub_seg)
        if r <= threshold_rms and r < best_rms:
            best_rms = r
            best_ms = t

    # For end boundaries, ensure no residual audio; shift forward if needed
    if not is_start:
        # Check if there's residual audio after best_ms
        check_win_ms = 50  # Check next 50ms for silence
        check_start = best_ms
        check_end = min(len(src), best_ms + check_win_ms)
        if check_end > check_start:
            check_seg = src[check_start:check_end]  # type: ignore
            check_rms = _rms(check_seg)  # type: ignore
            if check_rms > threshold_rms:
                # Shift forward until silence
                for t in range(best_ms, check_end, step_ms):
                    sub_seg = _window_slice(src, t, sub_win_ms)
                    r = _rms(sub_seg)
                    if r <= threshold_rms:
                        best_ms = t
                        break

    return best_ms


def _find_true_silence_end(
    src: AudioSegment,
    raw_end_ms: int,
    max_seek_ms: int = 200,
    step_ms: int = 5,
) -> int:
    """Seek forward from raw_end_ms to the earliest stable-silence start.

    Looks ahead up to max_seek_ms for the first point where two consecutive
    short windows (≈30ms each) are both below a dynamic threshold computed
    from the last 60ms before raw_end_ms. If not found, returns the lowest RMS
    point in the search range.
    """
    eval_half_win = 15  # ~30ms eval window
    search_start = int(raw_end_ms)
    search_end = min(len(src), search_start + int(max_seek_ms))

    # Baseline RMS from 60ms just before raw_end_ms (avoid using snapped end noise)
    pre1 = max(0, search_start - 60)
    pre_seg = src[pre1:search_start]  # type: ignore
    baseline = max(1.0, _rms(pre_seg))  # type: ignore
    # Dynamic threshold: 20% of baseline (clamped)
    thr = max(5.0, baseline * 0.20)

    best_ms = search_start
    best_r = float("inf")

    for t in range(search_start, search_end + 1, max(1, int(step_ms))):
        w1 = _window_slice(src, t, eval_half_win)
        r1 = _rms(w1)  # type: ignore
        if r1 < best_r:
            best_r = r1
            best_ms = t
        # Stability check: also quiet shortly after t
        t2 = min(len(src), t + eval_half_win)
        w2 = _window_slice(src, t2, eval_half_win)
        r2 = _rms(w2)  # type: ignore
        if r1 <= thr and r2 <= thr:
            return t
    return best_ms


def _find_speech_onset(
    src: AudioSegment,
    raw_start_ms: int,
    max_seek_ms: int = 250,
    step_ms: int = 5,
) -> int:
    """Find the earliest stable speech onset at or after raw_start_ms.

    We compute a noise baseline from 80ms before raw_start_ms (if available),
    then scan forward for the earliest time where two consecutive short
    windows (~30ms) exceed an onset threshold (e.g., 35% of baseline + guard).
    Returns the detected onset slightly earlier (5ms) to avoid clipping.
    """
    eval_half_win = 15
    search_start = max(0, int(raw_start_ms))
    search_end = min(len(src), search_start + int(max_seek_ms))

    pre0 = max(0, search_start - 80)
    noise_seg = src[pre0:search_start]  # type: ignore
    noise = max(1.0, _rms(noise_seg))  # type: ignore
    # Onset threshold: 35% of (noise + local energy guard)
    thr = max(15.0, noise * 0.35)

    for t in range(search_start, search_end + 1, max(1, int(step_ms))):
        w1 = _window_slice(src, t, eval_half_win)
        w2 = _window_slice(src, min(len(src), t + eval_half_win), eval_half_win)
        r1, r2 = _rms(w1), _rms(w2)  # type: ignore
        if r1 >= thr and r2 >= thr:
            return max(0, t - 5)
    # Fallback: use find_silence_boundary for start
    return find_silence_boundary(src, search_start, is_start=True, cfg={"silence_threshold_rms_percent": 20})


@dataclass
class Group:
    start_ms: int
    end_ms: int
    last_word_index: int


def _build_groups_by_period(words: List[Dict[str, Any]], punct_map: Dict[int, str], gap_merge_ms: int = 180) -> List[Group]:
    """Build groups by splitting on punctuation '.' or large silences > gap_merge_ms.

    Groups words into chunks where each chunk ends with '.' or when there's a large silence (>180ms) to the next word.
    This preserves micro-pauses within chunks and cuts audio per chunk without internal editing.
    """
    groups: List[Group] = []
    if not words:
        return groups  # type: ignore

    cur_start_idx = 0
    for i in range(len(words)):
        punct = punct_map.get(i, '')
        split = False
        if punct == '.':
            split = True
        elif i < len(words) - 1:
            gap = int(round(words[i + 1]["start"] * 1000)) - int(round(words[i]["end"] * 1000))
            if gap > gap_merge_ms:
                split = True
        if split:
            start_ms = int(round(words[cur_start_idx]["start"] * 1000))
            end_ms = int(round(words[i]["end"] * 1000))
            groups.append(Group(start_ms=start_ms, end_ms=end_ms, last_word_index=i))  # type: ignore
            cur_start_idx = i + 1

    # Last group
    if cur_start_idx < len(words):
        start_ms = int(round(words[cur_start_idx]["start"] * 1000))
        end_ms = int(round(words[-1]["end"] * 1000))
        groups.append(Group(start_ms=start_ms, end_ms=end_ms, last_word_index=len(words) - 1))  # type: ignore
    return groups  # type: ignore


def _group_words_by_gap(words: List[Dict[str, Any]], gap_merge_ms: int) -> List[Group]:  # type: ignore
    """Group words by large silences (>gap_merge_ms).

    This is used when no periods are present for grouping.
    """
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
    cfg: Dict[str, Any],
) -> int:
    """Determine silence to add after a group based on punctuation rules.

    For periods: Always insert exactly 24 frames of silence, calculated as duration_ms = 24 * (1000 / fps).
    Do not add silence if natural ending gap > silence duration before cut.
    Integrate silence into boundary adjustments, not as separate appends.
    """
    last_idx = groups[idx].last_word_index
    punct = punct_map.get(last_idx, None)
    fps = int(cfg.get("frame_rate", 30))
    silence_duration_ms = frames_to_ms(24, fps)  # Always 24 frames for periods

    # Natural gap between this group end and next group start (raw timeline)
    if idx < len(groups) - 1:
        cur_end = groups[idx].end_ms
        nxt_start = groups[idx + 1].start_ms
        natural_gap = max(0, nxt_start - cur_end)
    else:
        natural_gap = 0

    if punct == ".":
        # Do not add silence if natural gap > silence duration
        if natural_gap > silence_duration_ms:
            print(f"Natural gap {natural_gap}ms > silence duration {silence_duration_ms}ms, skipping silence insertion for period.")
            return 0
        print(f"Inserting {silence_duration_ms}ms silence after period.")
        return silence_duration_ms

    # Other punctuation: , ? ! : ;
    if punct in {",", "?", "!", ":", ";"}:
        # If natural gap is already very small (<= 3 frames), keep as-is
        three_frames_ms = frames_to_ms(3, fps)
        if natural_gap <= three_frames_ms:
            return 0
        frames = 3
        silence_ms = frames_to_ms(frames, fps)
        print(f"Inserting {silence_ms}ms silence after punctuation '{punct}'.")
        return silence_ms

    return 0

def synthesize_from_words(
    audio_path: str,
    words: List[Dict[str, Any]],
    punct_map: Dict[int, str],
    cfg: Dict[str, Any],
    out_path: str,
) -> None:
    """Synthesize edited audio from aligned words and punctuation rules.

    All changes are abstracted here for shared use by Streamlit and CLI.
    Steps per group:
    - Group words into chunks based on punctuation (.) or large silences (>180ms).
    - Cut audio per chunk, preserving micro-pauses within chunks. No editing inside chunks.
    - Use robust boundary snapping with find_silence_boundary.
    - Integrate silence insertion into boundary adjustments for periods.
    - Only crossfade between vocal segments when necessary. Use hard joins for silent segments.
    - Add safety guards and logs.
    """
    from .utils import safe_export  # type: ignore  # local import to avoid cycles

    src: AudioSegment = AudioSegment.from_file(audio_path)  # type: ignore
    gap_merge_ms = int(cfg.get("gap_merge_ms", 180))
    groups = _build_groups_by_period(words, punct_map, gap_merge_ms)

    pre_pad = int(cfg.get("pre_pad_ms", 40))
    post_pad = int(cfg.get("post_pad_ms", 80))
    crossfade_ms = int(cfg.get("overlap_crossfade_ms", 25))
    onset_seek_ms = int(cfg.get("onset_seek_ms", 250))

    output: AudioSegment = AudioSegment.silent(duration=0)
    last_adj_end_src: Optional[int] = None

    for gi, g in enumerate(groups):
        # Determine previous punctuation
        prev_punct = punct_map.get(groups[gi - 1].last_word_index) if gi > 0 else None

        # Apply padding in source timeline (pre-pad may be disabled after period)
        effective_pre_pad = pre_pad if prev_punct != "." else 0
        raw_start = max(0, g.start_ms - effective_pre_pad)
        raw_end = min(len(src), g.end_ms + post_pad)  # type: ignore

        # Start boundary: if previous group ended with '.', lock start to true speech onset
        if prev_punct == ".":
            adj_start = _find_speech_onset(src, raw_start, max_seek_ms=onset_seek_ms, step_ms=5)  # type: ignore
        else:
            adj_start = find_silence_boundary(src, raw_start, is_start=True, cfg=cfg)  # type: ignore

        adj_end = find_silence_boundary(src, raw_end, is_start=False, cfg=cfg)  # type: ignore
        if adj_end <= adj_start:
            adj_end = min(len(src), adj_start + 10)  # type: ignore

        # For periods, integrate silence into boundary adjustments
        punct = punct_map.get(g.last_word_index)
        if punct == ".":
            fps = int(cfg.get("frame_rate", 30))
            silence_duration_ms = frames_to_ms(24, fps)
            # Extend end boundary to include silence if natural gap is small
            if gi < len(groups) - 1:
                nxt_start = groups[gi + 1].start_ms
                natural_gap = max(0, nxt_start - g.end_ms)
                if natural_gap <= silence_duration_ms:
                    adj_end = min(len(src), adj_end + silence_duration_ms)  # type: ignore
                    print(f"Integrated {silence_duration_ms}ms silence into boundary for period.")
            # Seek to true silence end if needed
            adj_end = _find_true_silence_end(src, adj_end, max_seek_ms=350, step_ms=5)  # type: ignore

        seg = src[adj_start:adj_end]  # type: ignore

        # Safety guard: warn if chunk end still has RMS >25% after snapping
        end_rms = _rms(_window_slice(src, adj_end, 15))  # type: ignore
        baseline_seg_end = src[max(0, adj_end - 200):adj_end]  # type: ignore
        baseline_rms = _rms(baseline_seg_end)  # type: ignore
        if end_rms > baseline_rms * 0.25:
            print(f"WARNING: Chunk end at {adj_end}ms has RMS {end_rms:.2f} > 25% baseline {baseline_rms:.2f}, potential vocal clipping.")

        # Detect and flag repeated waveform energy after crossfade
        if last_adj_end_src is not None and adj_start <= last_adj_end_src:
            # Check for repeated energy in overlap region
            overlap_start = max(last_adj_end_src, adj_start)
            overlap_end = min(last_adj_end_src + crossfade_ms, adj_end)
            if overlap_end > overlap_start:
                overlap_seg = src[overlap_start:overlap_end]  # type: ignore
                overlap_rms = _rms(overlap_seg)  # type: ignore
                if overlap_rms > baseline_rms * 0.1:  # Arbitrary threshold for repeated energy
                    print(f"FLAG: Repeated waveform energy detected in overlap at {overlap_start}-{overlap_end}ms, RMS {overlap_rms:.2f}.")

        # Append with crossfade only if both segments are vocal (not silent)
        # Hard joins for silent segments
        is_vocal = _rms(seg) > baseline_rms * 0.05  # Threshold for vocal segment  # type: ignore
        prev_is_vocal = False
        if last_adj_end_src is not None:
            prev_seg_end = src[max(0, last_adj_end_src - 50):last_adj_end_src]  # type: ignore
            prev_is_vocal = _rms(prev_seg_end) > baseline_rms * 0.05  # type: ignore

        if last_adj_end_src is not None and adj_start <= last_adj_end_src and is_vocal and prev_is_vocal:
            output = output.append(seg, crossfade=crossfade_ms)  # type: ignore
        else:
            output += seg  # type: ignore
        last_adj_end_src = adj_end

        # Add silence after group except last (only if not integrated)
        if gi < len(groups) - 1 and punct != ".":
            add_ms = _silence_after_group(gi, groups, punct_map, cfg)
            if add_ms > 0:
                output += AudioSegment.silent(duration=add_ms)  # type: ignore

    # Export via utility
    safe_export(output, out_path, cfg)  # type: ignore
