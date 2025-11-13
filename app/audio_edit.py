from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Sequence, cast

from pydub import AudioSegment  # type: ignore

from .utils import frames_to_ms


# =============================
# Frame grid helpers (30 fps)
# =============================

def ms_per_frame(fps: float) -> float:
    """Return milliseconds per frame for a given frame rate.

    Args:
        fps: Frames per second (e.g., 30 for CapCut timeline).

    Returns:
        Milliseconds per frame as float (1000.0 / fps).
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    return 1000.0 / float(fps)


def quantize_ms(t_ms: float, fps: float) -> float:
    """Round t_ms to nearest frame (grid = 1000/fps)."""
    grid = ms_per_frame(fps)
    return round(float(t_ms) / grid) * grid


# =============================
# Low-level audio analysis
# =============================

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


def local_noise_rms(audio: AudioSegment, anchor_ms: int, pre_from_ms: int = 300, pre_to_ms: int = 100) -> float:
    """Estimate local noise floor before anchor_ms.

    Takes a slice in [anchor_ms - pre_from_ms, anchor_ms - pre_to_ms] and measures RMS.
    Falls back to a shorter available slice if near start.
    """
    a = int(anchor_ms)
    start = max(0, a - int(pre_from_ms))
    end = max(0, a - int(pre_to_ms))
    if end <= start:
        start = max(0, a - int(pre_to_ms))
        end = a
    seg = audio[start:end]  # type: ignore
    return max(0.0, _rms(seg))  # type: ignore


def speech_peak_rms(audio: AudioSegment, anchor_ms: int, lookback_ms: int = 250) -> float:
    """Estimate recent speech peak RMS to build adaptive thresholds."""
    a = int(anchor_ms)
    start = max(0, a - int(lookback_ms))
    end = a
    seg = audio[start:end]  # type: ignore
    return max(0.0, _rms(seg))  # type: ignore


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
        tiny_slice: AudioSegment = cast(AudioSegment, src[left:right])  # type: ignore
        tiny: AudioSegment = cast(AudioSegment, tiny_slice.set_channels(1))  # type: ignore
        sr_obj = getattr(tiny, "frame_rate")
        sr: int = int(sr_obj)
        raw_samples = getattr(tiny, "get_array_of_samples")()
        samples_seq: Sequence[int] = cast(Sequence[int], raw_samples)
        samples: list[int] = [int(x) for x in samples_seq]
        if not samples:
            return int(target_ms)
        best_idx: int = 0
        best_abs: int = abs(int(samples[0]))
        prev: int = int(samples[0])
        for i in range(1, len(samples)):
            s: int = int(samples[i])
            if s == 0 or (s > 0 and prev < 0) or (s < 0 and prev > 0):
                ms = int(round(left + (i * 1000.0 / sr)))
                return ms
            aval: int = abs(int(s))
            if aval < best_abs:
                best_abs = aval
                best_idx = i
            prev = s
        ms = int(round(left + (best_idx * 1000.0 / sr)))
        return ms
    except Exception:
        return int(target_ms)


def snap_zero_cross(audio: AudioSegment, center_ms: int, radius_ms: int = 8) -> int:
    """Return ms near center at a zero crossing within ±radius."""
    return int(_nearest_zero_crossing_ms(audio, int(center_ms), window_ms=int(radius_ms)))


# =============================
# Boundary finders (EoS / Onset)
# =============================

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


# (Legacy experimental seekers removed; using spec-compliant helpers below.)


# New EoS/Onset that satisfy spec

def find_true_eos(
    audio: AudioSegment,
    end_ms: int,
    seek_max: int = 700,
    window_ms: int = 40,
    stable_ms: int = 150,
    step_ms: int = 5,
    thr_ratio: float = 0.15,
) -> int:
    """
    EoS seek FORWARD: from end_ms, scan ≤ seek_max to find sustained low-RMS (RMS < thr) for ≥ stable_ms.
    thr = max(local_noise*1.2, speech_peak*thr_ratio). Return snapped zero-crossing near the best valley.
    """
    start_probe = int(end_ms)
    limit = min(len(audio), start_probe + int(seek_max))

    # Adaptive threshold from local context
    noise = local_noise_rms(audio, start_probe)
    peak = speech_peak_rms(audio, start_probe)
    thr = max(noise * 1.2, peak * float(thr_ratio), 5.0)

    half = max(1, int(window_ms // 2))
    need_consec = max(1, int(round(stable_ms / max(1, step_ms))))
    best_t = start_probe
    best_r = float("inf")
    consec = 0

    t = start_probe
    while t <= limit:
        seg = _window_slice(audio, t, half)
        r = _rms(seg)
        if r < best_r:
            best_r = r
            best_t = t
        if r <= thr:
            consec += 1
            if consec >= need_consec:
                return snap_zero_cross(audio, t, radius_ms=8)
        else:
            consec = 0
        t += max(1, int(step_ms))
    return snap_zero_cross(audio, best_t, radius_ms=8)


def find_safe_onset(
    audio: AudioSegment,
    start_ms: int,
    seek_back_max: int = 200,
    window_ms: int = 30,
    consec: int = 3,
    ratio: float = 0.5,
) -> int:
    """
    Onset seeker BACKWARD: pull start into silence; confirm 3 consecutive windows exceeding
    ratio*noise_baseline when moving forward again. Snap to zero crossing to avoid clipping leading consonants.
    """
    s = int(start_ms)
    limit = max(0, s - int(seek_back_max))
    # Estimate noise baseline using a small window preceding the candidate
    pre_noise = local_noise_rms(audio, s, pre_from_ms=250, pre_to_ms=80)
    thr = max(5.0, pre_noise * float(ratio))

    step = max(1, int(window_ms // 2))
    probe = s
    best = s
    best_r = float("inf")
    while probe >= limit:
        seg = _window_slice(audio, probe, max(1, int(window_ms // 2)))
        r = _rms(seg)
        if r < best_r:
            best_r = r
            best = probe
        if r <= thr:
            # Now scan forward to certify onset with consecutive above-threshold windows
            t = probe
            cnt = 0
            while t < min(len(audio), probe + 200):
                w = _window_slice(audio, t, max(1, int(window_ms // 2)))
                if _rms(w) >= thr:
                    cnt += 1
                    if cnt >= max(1, int(consec)):
                        return snap_zero_cross(audio, probe, radius_ms=8)
                else:
                    break
                t += step
        probe -= step
    return snap_zero_cross(audio, best, radius_ms=8)


# =============================
# Grouping utilities
# =============================

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


# =============================
# Pause normalization helpers
# =============================

def trim_tail_silence(
    audio: AudioSegment,
    end_ms: int,
    max_cut_ms: int,
    thr_rms: float,
    window_ms: int = 30,
    step_ms: int = 5,
) -> int:
    """Trim only low-RMS tail (≤ thr_rms) up to max_cut_ms; return new end_ms (never into voiced region)."""
    start = int(end_ms)
    limit = min(len(audio), start + int(max_cut_ms))
    t = start
    last_ok = start
    while t <= limit:
        seg = _window_slice(audio, t, max(1, int(window_ms // 2)))
        if _rms(seg) <= thr_rms:
            last_ok = t
            t += max(1, int(step_ms))
        else:
            break
    return snap_zero_cross(audio, last_ok, radius_ms=8)


def trim_head_silence(
    audio: AudioSegment,
    start_ms: int,
    max_cut_ms: int,
    thr_rms: float,
    window_ms: int = 30,
    step_ms: int = 5,
) -> int:
    """Trim only low-RMS head (≤ thr_rms) up to max_cut_ms; return new start_ms (never into voiced region)."""
    s = int(start_ms)
    limit = max(0, s - int(max_cut_ms))
    t = s
    last_ok = s
    while t >= limit:
        seg = _window_slice(audio, t, max(1, int(window_ms // 2)))
        if _rms(seg) <= thr_rms:
            last_ok = t
            t -= max(1, int(step_ms))
        else:
            break
    return snap_zero_cross(audio, last_ok, radius_ms=8)


def enforce_exact_pause(
    audio: AudioSegment,
    end_ms: int,
    start_ms: int,
    fps: float,
    frames: int,
    thr_rms: float,
) -> tuple[int, int]:
    """
    Target = frames*(1000/fps). If gap > target: trim silent tail/head (≤ thr_rms) from both sides.
    If gap < target: extend end_ms forward by (target-gap) (insert silence later).
    Quantize both ends to frame grid. Return (new_end_ms, new_start_ms).
    """
    target = frames_to_ms(frames, int(round(fps)))
    gap = int(start_ms) - int(end_ms)
    new_end = int(end_ms)
    new_start = int(start_ms)

    if gap > target:
        need = gap - target
        # Trim tail first
        te = trim_tail_silence(audio, new_end, max_cut_ms=need, thr_rms=thr_rms)
        moved = te - new_end
        if moved > 0:
            new_end = te
            need = max(0, need - moved)
        # Then trim head if still need
        if need > 0:
            ts = trim_head_silence(audio, new_start, max_cut_ms=need, thr_rms=thr_rms)
            moved2 = new_start - ts
            if moved2 > 0:
                new_start = ts
        if new_start < new_end:
            new_start = new_end
    elif gap < target:
        # Keep boundaries; caller will add hard silence to fill the missing part
        pass

    q_end = int(quantize_ms(new_end, fps))
    q_start = int(quantize_ms(new_start, fps))
    return q_end, q_start


def _ensure_wav48(seg: Any) -> AudioSegment:
    """Return a 48kHz, 16-bit PCM width AudioSegment.

    Pydub's fluent setters are untyped; keep a narrow helper to encapsulate
    the operations and return a concrete AudioSegment for static checkers.
    """
    out = seg
    out = out.set_frame_rate(48000)  # type: ignore
    out = out.set_sample_width(2)  # type: ignore
    return out  # type: ignore[return-value]


# =============================
# Legacy helpers retained
# =============================

def safe_push_end_forward(end_ms: int, full_audio: AudioSegment, max_seek: int = 350, rms_thresh: float = 0.15) -> int:
    """Seek forward from end_ms until RMS < threshold for a >=100ms span.

    Threshold is relative to a baseline from 200ms preceding the probe point.
    Returns the new end_ms (>= original) aligned to a nearby zero crossing.
    """
    step = 5
    probe = int(end_ms)
    limit = min(len(full_audio), probe + int(max_seek))
    best = probe
    best_r = float("inf")
    while probe <= limit:
        base_start = max(0, probe - 200)
        baseline = max(1.0, _rms(cast(AudioSegment, full_audio[base_start:probe])))  # type: ignore
        thr = max(5.0, baseline * rms_thresh)
        span_end = min(len(full_audio), probe + 100)
        r = _rms_over_span(full_audio, probe, span_end)
        if r < best_r:
            best_r = r
            best = probe
        if r <= thr:
            return _nearest_zero_crossing_ms(full_audio, probe, window_ms=8)
        probe += step
    return _nearest_zero_crossing_ms(full_audio, best, window_ms=8)


def safe_pull_start_backward(start_ms: int, full_audio: AudioSegment, max_seek: int = 200, rms_thresh: float = 0.15) -> int:
    """Seek backward from start_ms into a preceding silence region.

    Looks for a >=100ms window whose RMS is below threshold relative to a
    baseline from the next 200ms after the candidate (avoid cutting attacks).
    Returns a new start_ms (<= original) aligned to a nearby zero crossing.
    """
    step = 5
    probe = int(start_ms)
    limit = max(0, probe - int(max_seek))
    best = probe
    best_r = float("inf")
    while probe >= limit:
        base_end = min(len(full_audio), probe + 200)
        baseline = max(1.0, _rms(cast(AudioSegment, full_audio[probe:base_end])))  # type: ignore
        thr = max(5.0, baseline * rms_thresh)
        span_end = min(len(full_audio), probe + 100)
        r = _rms_over_span(full_audio, probe, span_end)
        if r < best_r:
            best_r = r
            best = probe
        if r <= thr:
            return _nearest_zero_crossing_ms(full_audio, probe, window_ms=8)
        probe -= step
    return _nearest_zero_crossing_ms(full_audio, best, window_ms=8)


def _rms_over_span(src: AudioSegment, start_ms: int, end_ms: int) -> float:
    start = max(0, int(start_ms))
    end = min(len(src), int(end_ms))
    if end <= start:
        return 0.0
    return _rms(cast(AudioSegment, src[start:end]))  # type: ignore


# =============================
# Core synthesis
# =============================

def synthesize_from_words(
    audio_path: str,
    words: List[Dict[str, Any]],
    punct_map: Dict[int, str],
    cfg: Dict[str, Any],
    out_path: str,
) -> None:
    """Synthesize edited audio with exact frame pauses and safe boundaries.

    - Groups by period '.' (fallback to long gaps).
    - Seeks EoS forward to stable low-RMS valley; snaps to zero-cross.
    - Seeks onset backward to safe silence; snaps to zero-cross.
    - Enforces exact 24 frames @ fps for '.' using two-way normalization.
    - Avoids double-counting silence (measure natural gap).
    - Quantizes all timecodes to fps grid.
    - Exports deterministic WAV 48kHz PCM for frame measurement; MP3 only for preview.
    """
    from .utils import safe_export  # type: ignore

    src: AudioSegment = cast(AudioSegment, AudioSegment.from_file(audio_path))  # type: ignore

    fps: float = float(cfg.get("frame_rate", 30))
    target_period_frames = int(cfg.get("silence_after_period_frames", 24))
    target_period_ms = frames_to_ms(target_period_frames, int(round(fps)))

    gap_merge_ms = int(cfg.get("gap_merge_ms", 180))
    groups = _build_groups_by_period(words, punct_map, gap_merge_ms)
    if not groups:
        # Nothing to do; export silence
        safe_export(AudioSegment.silent(duration=0), out_path, cfg)  # type: ignore
        return

    pre_pad = int(cfg.get("pre_pad_ms", 40))
    post_pad = int(cfg.get("post_pad_ms", 80))

    # Pass 1: initial raw boundaries with pads
    raw_starts: List[int] = []
    raw_ends: List[int] = []
    for gi, g in enumerate(groups):
        prev_punct = punct_map.get(groups[gi - 1].last_word_index) if gi > 0 else None
        effective_pre = pre_pad if prev_punct != "." else 0
        raw_start = max(0, g.start_ms - effective_pre)
        raw_end = min(len(src), g.end_ms + post_pad)
        raw_starts.append(int(raw_start))
        raw_ends.append(int(raw_end))

    # Pass 2: EoS and onset seek
    starts: List[int] = []
    ends: List[int] = []
    puncts: List[Optional[str]] = []

    for gi, g in enumerate(groups):
        puncts.append(punct_map.get(g.last_word_index))

        # Start: prefer safe onset when previous boundary is a period or we have overlap risk
        if gi == 0:
            s = find_silence_boundary(src, raw_starts[gi], is_start=True, cfg=cfg)
        else:
            prev_p = punct_map.get(groups[gi - 1].last_word_index)
            if prev_p == ".":
                s = find_safe_onset(src, raw_starts[gi], seek_back_max=200, window_ms=30, consec=3, ratio=0.5)
            else:
                s = find_silence_boundary(src, raw_starts[gi], is_start=True, cfg=cfg)
        s = snap_zero_cross(src, s, radius_ms=8)
        s = int(quantize_ms(s, fps))

        # End: true EoS forward search, never earlier than raw end
        e0 = max(raw_ends[gi], g.end_ms)
        true_e = find_true_eos(src, e0, seek_max=700, window_ms=40, stable_ms=150, step_ms=5, thr_ratio=0.15)
        true_e = max(true_e, raw_ends[gi])
        true_e = snap_zero_cross(src, true_e, radius_ms=8)
        true_e = int(quantize_ms(true_e, fps))

        if true_e <= s:
            true_e = min(len(src), s + 10)
            true_e = int(quantize_ms(true_e, fps))

        starts.append(s)
        ends.append(true_e)

    # Pass 3: normalize period pauses with two-way trimming (no double counting)
    extra_silences: List[int] = [0 for _ in range(max(0, len(groups) - 1))]

    for i in range(len(groups) - 1):
        punct = puncts[i]
        # Compute tentative safe onset for next chunk (stricter to protect next word onset)
        next_onset = find_safe_onset(src, starts[i + 1], seek_back_max=200, window_ms=30, consec=3, ratio=0.5)
        next_onset = int(quantize_ms(snap_zero_cross(src, next_onset, radius_ms=8), fps))

        # Current true end already computed in ends[i]
        true_end = ends[i]

        natural_gap = next_onset - true_end
        target_ms = target_period_ms if punct == "." else 0

        if punct == ".":
            # Adaptive threshold for silence-only trimming
            noise = local_noise_rms(src, true_end)
            peak = speech_peak_rms(src, true_end)
            thr = max(noise * 1.2, peak * 0.15, 5.0)

            # Two-way normalization only if natural gap > target
            trimmed_tail = False
            trimmed_head = False
            if natural_gap > target_ms:
                new_end, new_start = enforce_exact_pause(src, true_end, next_onset, fps, target_period_frames, thr)
                trimmed_tail = new_end != true_end
                trimmed_head = new_start != next_onset
                ends[i], starts[i + 1] = new_end, new_start
                true_end, next_onset = new_end, new_start
                natural_gap = next_onset - true_end

            insert_ms = 0
            if natural_gap < target_ms - 5:
                insert_ms = target_ms - natural_gap

            # Overlap handling (no midpoint): push end forward, then pull start backward
            if starts[i + 1] < ends[i]:
                ends[i] = find_true_eos(src, ends[i], seek_max=400, window_ms=40, stable_ms=120, step_ms=5, thr_ratio=0.15)
                ends[i] = int(quantize_ms(snap_zero_cross(src, ends[i], radius_ms=8), fps))
                if starts[i + 1] < ends[i]:
                    starts[i + 1] = find_safe_onset(src, starts[i + 1], seek_back_max=200, window_ms=30, consec=3, ratio=0.5)
                    starts[i + 1] = int(quantize_ms(snap_zero_cross(src, starts[i + 1], radius_ms=8), fps))

            # Log debug for period boundary
            final_gap = (starts[i + 1] - ends[i]) + insert_ms
            log = (
                f"[PERIOD] true_end_ms={true_end} next_onset_ms={next_onset} "
                f"natural_gap_ms={natural_gap} insert_ms={insert_ms} final_gap_ms={final_gap} target_ms={target_ms} "
                f"trim_tail={'Y' if trimmed_tail else 'N'} trim_head={'Y' if trimmed_head else 'N'}"
            )
            print(log)
            if abs(final_gap - target_ms) > 5:
                print(f"[WARN] pause off: expected={target_ms}ms, got={final_gap}ms at junction {i}")

            extra_silences[i] = max(0, int(round(insert_ms)))
        else:
            # Non-period: deterministic, do not insert extra silence by default
            extra_silences[i] = 0

    # Pass 4: de-overlap for non-period boundaries as safety
    for i in range(len(groups) - 1):
        if starts[i + 1] < ends[i]:
            # push end forward first
            ends[i] = find_true_eos(src, ends[i], seek_max=300, window_ms=40, stable_ms=100, step_ms=5, thr_ratio=0.15)
            ends[i] = int(quantize_ms(snap_zero_cross(src, ends[i], radius_ms=8), fps))
            if starts[i + 1] < ends[i]:
                starts[i + 1] = find_safe_onset(src, starts[i + 1], seek_back_max=200, window_ms=30, consec=3, ratio=0.5)
                starts[i + 1] = int(quantize_ms(snap_zero_cross(src, starts[i + 1], radius_ms=8), fps))

    # Pass 5: render (hard join unless both sides voiced and no insert)
    crossfade_ms = int(cfg.get("overlap_crossfade_ms", 20))
    output: AudioSegment = AudioSegment.silent(duration=0)  # type: ignore

    for gi in range(len(groups)):
        seg = cast(AudioSegment, src[starts[gi]:ends[gi]])  # type: ignore

        if gi > 0:
            # Decide crossfade eligibility for non-period joins without inserted silence
            is_period = puncts[gi - 1] == "."
            add_ms = extra_silences[gi - 1] if gi - 1 < len(extra_silences) else 0
            if not is_period and add_ms == 0 and crossfade_ms > 0:
                prev_tail = cast(AudioSegment, src[max(0, ends[gi - 1] - 60):ends[gi - 1]])  # type: ignore
                next_head = cast(AudioSegment, src[starts[gi]:min(len(src), starts[gi] + 60)])  # type: ignore
                baseline_prev = _rms(prev_tail)
                baseline_next_noise = _rms(cast(AudioSegment, src[max(0, starts[gi] - 80):starts[gi]]))  # type: ignore
                end_is_voiced = _rms(prev_tail) > max(5.0, baseline_prev * 0.05)
                start_is_voiced = _rms(next_head) > max(5.0, baseline_next_noise * 0.05)
                strong_attack = _rms(next_head) > max(30.0, baseline_next_noise * 1.8)
                if end_is_voiced and start_is_voiced and not strong_attack:
                    output = output.append(seg, crossfade=crossfade_ms)  # type: ignore
                else:
                    output += seg  # type: ignore
            else:
                output += seg  # type: ignore
        else:
            output += seg  # type: ignore

        if gi < len(groups) - 1:
            add_ms = extra_silences[gi]
            if add_ms > 0:
                add_ms_q = int(quantize_ms(add_ms, fps))
                output += AudioSegment.silent(duration=add_ms_q)  # type: ignore

    # Export: ensure WAV 48kHz PCM16 for frame-based validation. MP3 reserved for preview.
    out_is_mp3 = str(out_path).lower().endswith(".mp3")
    # Build 48kHz PCM16 artifact for frame-based validation
    wav_48: AudioSegment = _ensure_wav48(output)

    if out_is_mp3:
        # Export preview MP3 as requested by UI (use wav_48 as the source to keep typing stable)
        try:
            safe_export(wav_48, out_path, cfg)
        except Exception:
            pass
        # Also export a sibling WAV for precise frame checking
        from pathlib import Path
        p = Path(out_path)
        wav_path = str(p.with_suffix("")) + "_frame.wav"
        safe_export(wav_48, wav_path, {**cfg, "export_format": "wav"})
    else:
        # Direct WAV path: ensure 48kHz
        safe_export(wav_48, out_path, {**cfg, "export_format": "wav"})

    # Final verification logs for periods
    for i in range(len(groups) - 1):
        if puncts[i] == ".":
            measured = (starts[i + 1] - ends[i]) + extra_silences[i]
            expected = target_period_ms
            if abs(measured - expected) > 5:
                print(f"[WARN] Pause off: expected={expected}ms, measured={measured}ms at junction {i}")
            else:
                print(f"[OK] Period pause ~{measured}ms at junction {i}")
