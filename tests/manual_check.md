# Manual Checks

## 1) Dấu chấm: nghỉ 24 frames (~800ms @30fps)
- Prepare two short sentences with a clear period between them.
- Run UI or CLI to synthesize. Inspect the pause after each '.'; it should be ~800ms (24 frames @30fps).
- Listen carefully at the end boundary: it should cut only after the voice fully decays (not clipping final consonants), thanks to snapping and true-silence seek.

## 2) Dấu phẩy: “xin chào, tôi tên là …”
- Use the exact phrase with a brief natural gap between “chào,” and “tôi”.
- Ensure the natural gap < `comma_soft_break_min_gap_ms` (default 250ms).
- Verify there is no forced cut that breaks the word flow. It should not sound glued (no merged syllables), and no extra pause unless the natural gap is smaller than the threshold.

## 3) Cụm dài không có dấu chấm
- Provide a long sentence without any '.' characters.
- The fallback grouping uses `gap_merge_ms` to merge by natural gaps.
- Validate no loss of leading/trailing consonants at group boundaries: pre/post padding, snapping to valleys, and crossfade when touching should avoid artifacts.
