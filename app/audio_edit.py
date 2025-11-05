from __future__ import annotations

from dataclasses import dataclass
import random
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
        # Require continuous low-RMS span of ~90ms (3 windows)
        w0 = _window_slice(src, t, eval_half_win)
        w1 = _window_slice(src, min(len(src), t + eval_half_win), eval_half_win)
        w2 = _window_slice(src, min(len(src), t + eval_half_win * 2), eval_half_win)
        r0, r1, r2 = _rms(w0), _rms(w1), _rms(w2)  # type: ignore
        if r0 < best_r:
            best_r = r0
            best_ms = t
        if r0 <= thr and r1 <= thr and r2 <= thr:
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


def _nearest_zero_crossing_ms(src: AudioSegment, target_ms: int, window_ms: int = 8) -> int:
    """Snap a boundary to the nearest zero crossing within a small window.

    Converts a tiny slice around target to mono samples and looks for a sign
    change or minimal absolute amplitude.
    """
    try:
        if window_ms <= 0:
            return int(target_ms)
        left = max(0, int(target_ms) - window_ms)
        right = min(len(src), int(target_ms) + window_ms)
        tiny = src[left:right].set_channels(1)  # type: ignore
        sr = tiny.frame_rate
        arr = tiny.get_array_of_samples()
        if not arr:
            return int(target_ms)
        best_idx = 0
        best_abs = abs(arr[0])
        prev = arr[0]
        for i in range(1, len(arr)):
            s = arr[i]
            if s == 0 or (s > 0 and prev < 0) or (s < 0 and prev > 0):
                ms = int(round(left + (i * 1000.0 / sr)))
                return ms
            aval = abs(s)
            if aval < best_abs:
                best_abs = aval
                best_idx = i
            prev = s
        ms = int(round(left + (best_idx * 1000.0 / sr)))
        return ms
    except Exception:
        return int(target_ms)


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
    """Synthesize edited audio from aligned words with safe, exact pauses.

    Strategy:
    - Build groups (period priority).
    - Adjust boundaries with directional snapping + onset/silence seek.
    - Snap to zero-crossings.
    - Normalize period pauses to exactly 24 frames @ 30fps by adding or trimming
      within silence (with a small guard before the next onset).
    - For other punctuation, add 3–5 frames only if natural gap > 3 frames.
    - Ensure no overlap (butt splice) to avoid duplicated audio.
    """
    from .utils import safe_export  # type: ignore

    src: AudioSegment = AudioSegment.from_file(audio_path)  # type: ignore
    gap_merge_ms = int(cfg.get("gap_merge_ms", 180))
    groups = _build_groups_by_period(words, punct_map, gap_merge_ms)

    pre_pad = int(cfg.get("pre_pad_ms", 40))
    post_pad = int(cfg.get("post_pad_ms", 80))
    onset_seek_ms = int(cfg.get("onset_seek_ms", 250))
    period_seek_ms = int(cfg.get("period_silence_seek_ms", 350))
    min_stable_silence_ms = int(cfg.get("min_stable_silence_ms", 90))

    # Pass 1: adjusted boundaries per group
    adj_starts: List[int] = []
    adj_ends: List[int] = []
    puncts: List[Optional[str]] = []

    for gi, g in enumerate(groups):
        prev_punct = punct_map.get(groups[gi - 1].last_word_index) if gi > 0 else None
        puncts.append(punct_map.get(g.last_word_index))

        effective_pre_pad = pre_pad if prev_punct != "." else 0
        raw_start = max(0, g.start_ms - effective_pre_pad)
        raw_end = min(len(src), g.end_ms + post_pad)

        if prev_punct == ".":
            s = _find_speech_onset(src, raw_start, max_seek_ms=onset_seek_ms, step_ms=5)
        else:
            s = find_silence_boundary(src, raw_start, is_start=True, cfg=cfg)
        s = _nearest_zero_crossing_ms(src, s, window_ms=8)

        e = find_silence_boundary(src, raw_end, is_start=False, cfg=cfg)
        if puncts[-1] == ".":
            # Seek until we have at least min_stable_silence_ms of stable silence
            e0 = _find_true_silence_end(src, e, max_seek_ms=period_seek_ms, step_ms=5)
            # expand until stable silence length satisfied
            seek_end = min(len(src), e0 + period_seek_ms)
            cur = e0
            while cur < seek_end:
                span = src[max(0, cur - min_stable_silence_ms):cur]  # type: ignore
                baseline = max(1.0, _rms(src[max(0, cur - 200):cur]))  # type: ignore
                thr = max(5.0, baseline * float(cfg.get("silence_threshold_ratio", 0.18)))
                if _rms(span) <= thr:
                    break
                cur += 5
            e = _nearest_zero_crossing_ms(src, cur, window_ms=8)
        e = _nearest_zero_crossing_ms(src, e, window_ms=8)

        if e <= s:
            e = min(len(src), s + 10)

        adj_starts.append(s)
        adj_ends.append(e)

    # Pass 2: normalize pauses
    fps = int(cfg.get("frame_rate", 30))
    target_ms = frames_to_ms(int(cfg.get("silence_after_period_frames", 24)), fps)
    guard_ms = 12
    extra_silences: List[int] = [0 for _ in range(max(0, len(groups) - 1))]

    for i in range(len(groups) - 1):
        punct = puncts[i]
        gap = adj_starts[i + 1] - adj_ends[i]
        if punct == ".":
            if gap < target_ms:
                extra_silences[i] = target_ms - gap
            elif gap > target_ms:
                need_trim = gap - target_ms
                allowed = max(0, adj_starts[i + 1] - guard_ms - adj_ends[i])
                trim = min(allowed, need_trim)
                if trim > 0:
                    adj_ends[i] = _nearest_zero_crossing_ms(src, adj_ends[i] + trim, window_ms=6)
                # recompute missing (if any)
                gap2 = adj_starts[i + 1] - adj_ends[i]
                if gap2 < target_ms:
                    extra_silences[i] = target_ms - gap2
        else:
            three = frames_to_ms(3, fps)
            if gap <= three:
                extra_silences[i] = 0
            else:
                min_f = int(cfg.get("silence_other_punct_frames_min", 3))
                max_f = int(cfg.get("silence_other_punct_frames_max", 5))
                extra_silences[i] = frames_to_ms(random.randint(min_f, max_f), fps)

    # Pass 3: de-overlap junctions
    for i in range(len(groups) - 1):
        if adj_starts[i + 1] < adj_ends[i]:
            mid = int((adj_ends[i] + adj_starts[i + 1]) / 2)
            j = _nearest_zero_crossing_ms(src, mid, window_ms=8)
            adj_ends[i] = j
            adj_starts[i + 1] = j

    # Pass 4: render
    output: AudioSegment = AudioSegment.silent(duration=0)  # type: ignore
    for gi in range(len(groups)):
        seg = src[adj_starts[gi]:adj_ends[gi]]  # type: ignore
        output += seg  # type: ignore
        if gi < len(groups) - 1:
            add_ms = extra_silences[gi]
            if add_ms > 0:
                output += AudioSegment.silent(duration=add_ms)  # type: ignore

    safe_export(output, out_path, cfg)  # type: ignore
