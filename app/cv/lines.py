"""
Line analysis on top of the vendored segmentation model.

The vendored `classify()` keeps only the 3 major lines (heart/head/life) nearest to
K-means centers and discards every other candidate. This module re-runs the candidate
extraction and additionally:

  * recovers the Fate line (Bhagya Rekha) as the best vertical/central candidate,
  * detects forks at line endpoints from skeleton junctions,
  * optionally bridges nearby fragments (dilate -> skeletonize) to reduce truncation.

All geometry is in the palm_lines.png space (square, side = resize_value).
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .vendor import classification as C


def _skeleton(palmline_path: str, bridge: int = 0) -> np.ndarray:
    """Binary skeleton (uint8 0/255). `bridge` dilates before skeletonizing to join fragments."""
    img = cv2.imread(palmline_path)
    if bridge > 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        binary = cv2.dilate(binary, np.ones((bridge, bridge), np.uint8), iterations=1)
        img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor((skeletonize(img).astype(np.uint8) * 255), cv2.COLOR_BGR2GRAY)


def _neighbor_count(skel: np.ndarray) -> np.ndarray:
    """Per-pixel count of 8-connected skeleton neighbours (junctions have >= 3)."""
    on = (skel > 0).astype(np.uint8)
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    return cv2.filter2D(on, -1, k) * on


def _endpoint_forks(skel: np.ndarray, line, radius: int = 4) -> bool:
    """True if the terminating end of `line` sits near a junction (a fork/branch)."""
    if not line or len(line) < 2:
        return False
    counts = _neighbor_count(skel)
    ey, ex = line[-1][:2]
    h, w = skel.shape
    y0, y1 = max(0, ey - radius), min(h, ey + radius + 1)
    x0, x1 = max(0, ex - radius), min(w, ex + radius + 1)
    return bool((counts[y0:y1, x0:x1] >= 3).any())


def _bounds(line):
    arr = np.array([(p[0], p[1]) for p in line], dtype=np.float32)
    ys, xs = arr[:, 0], arr[:, 1]
    return ys.min(), ys.max(), xs.min(), xs.max(), xs.mean()


def _select_fate(candidates, used_ids: set[int], h: int, w: int):
    """Pick the most vertical, centrally located candidate as the Fate line."""
    best, best_score = None, 0.0
    for idx, line in enumerate(candidates):
        if idx in used_ids or len(line) < 12:
            continue
        min_y, max_y, min_x, max_x, mean_x = _bounds(line)
        y_span, x_span = max_y - min_y, max_x - min_x
        # Fate line: tall, narrow, in the central third of the palm.
        if y_span < 0.22 * h:
            continue
        if not (0.30 * w <= mean_x <= 0.70 * w):
            continue
        verticality = y_span / (x_span + 1.0)
        if verticality < 1.4:
            continue
        score = y_span - 1.5 * x_span
        if score > best_score:
            best, best_score = line, score
    return best


def analyze(palmline_path: str, side: int, bridge: int = 0) -> dict:
    """Return {'heart','head','life','fate'} lines with points + fork flags (values may be None)."""
    centers = C.get_cluster_centers()
    skel = _skeleton(palmline_path, bridge=bridge)
    candidates = C.group(skel)

    result: dict[str, dict | None] = {"heart": None, "head": None, "life": None, "fate": None}
    if not candidates:
        return result

    major = C.classify_lines(centers, candidates, side, side)  # [heart, head, life] (may hold None)

    # Track which candidate indices were consumed by the 3 major lines.
    used_ids: set[int] = set()
    for line in major:
        if line is None:
            continue
        for idx, cand in enumerate(candidates):
            if cand is line:
                used_ids.add(idx)
                break

    names = ("heart", "head", "life")
    for name, line in zip(names, major):
        if line is not None:
            result[name] = {"points": line, "fork_end": _endpoint_forks(skel, line)}

    fate = _select_fate(candidates, used_ids, side, side)
    if fate is not None:
        result["fate"] = {"points": fate, "fork_end": _endpoint_forks(skel, fate)}
    return result
