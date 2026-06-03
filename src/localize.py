"""
Post-hoc temporal localization from the model's attention distributions.

Implements the inference procedure described in the paper (Section 3.5):
  1. Temperature-sharpen each modality's attention distribution.
  2. Align both modalities to a common resolution of N bins
     (audio via max-pooling, vision via linear interpolation).
  3. Combine with the learned modality gate weights.
  4. Pick the peak bin and expand left/right while attention exceeds the mean.
  5. Map the bin range to continuous timestamps within the segment.
"""
from typing import Tuple

import numpy as np

SEG_DUR = 5.0      # segment duration in seconds
NUM_BINS = 10      # temporal bins per segment
TEMPERATURE = 0.5  # sharpening temperature (lower => sharper)


def _sharpen(att: np.ndarray, tau: float = TEMPERATURE) -> np.ndarray:
    att = np.maximum(np.asarray(att, dtype=np.float64), 1e-12)
    sharpened = att ** (1.0 / tau)
    return sharpened / sharpened.sum()


def _to_bins(att: np.ndarray, num_bins: int) -> np.ndarray:
    """Align an attention vector to `num_bins` bins."""
    att = np.asarray(att, dtype=np.float64).flatten()
    if len(att) == num_bins:
        return att
    if len(att) > num_bins:
        step = len(att) // num_bins
        return att[: step * num_bins].reshape(num_bins, step).max(axis=1)
    return np.interp(
        np.linspace(0, 1, num_bins), np.linspace(0, 1, len(att)), att
    )


def _peak_expand(signal: np.ndarray, mean_val: float, seg_dur: float, num_bins: int):
    peak = int(np.argmax(signal))
    left, right = peak, peak
    while left > 0 and signal[left - 1] > mean_val:
        left -= 1
    while right < num_bins - 1 and signal[right + 1] > mean_val:
        right += 1
    bin_dur = seg_dur / num_bins
    return left * bin_dur, (right + 1) * bin_dur


def attention_to_interval(
    audio_att: np.ndarray,
    vision_att: np.ndarray,
    gates_att: np.ndarray,
    seg_dur: float = SEG_DUR,
    num_bins: int = NUM_BINS,
) -> Tuple[float, float]:
    """Convert per-modality attention + gate weights into a (start, end) interval."""
    if gates_att is None:
        gates = np.array([0.5, 0.5])
    else:
        gates = np.asarray(gates_att, dtype=np.float64).flatten()[:2]

    audio_binned = _sharpen(_to_bins(audio_att, num_bins))
    vision_binned = _sharpen(_to_bins(vision_att, num_bins))

    combined = audio_binned * gates[0] + vision_binned * gates[1]
    mean_val = audio_binned.mean() * gates[0] + vision_binned.mean() * gates[1]
    return _peak_expand(combined, mean_val, seg_dur, num_bins)


def iou(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def multi_gt_iou(pred: Tuple[float, float], gts) -> float:
    """Maximum IoU of a prediction against all ground-truth events in a sample."""
    if not gts:
        return 0.0
    return max(iou(pred, (g["start"], g["end"])) for g in gts)
