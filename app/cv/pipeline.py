"""
CV pipeline: palm image -> FeatureVector.

Orchestrates the vendored, validated palmistry model (U-Net Context Fusion,
yeonsumia/palmistry, Apache-2.0) and derives a structured feature vector.

Pipeline stages (mirrors the original read_palm.py, but returns data, not an image):
  1. background removal   (tools.remove_background)
  2. rectification/warp   (rectification.warp  -> MediaPipe homography)
  3. resize to 256        (tools.resize)
  4. line segmentation    (detection.detect    -> U-Net)
  5. line classification  (classification.classify -> K-means into heart/head/life)
  6. feature derivation   (this module: length, curvature, extent, confidence)
"""
from __future__ import annotations

import math
import os
import tempfile

import cv2
import mediapipe as mp
import numpy as np
import torch

from app.models.schemas import FeatureVector, LineFeature, LineLength
from . import lines as line_analysis
from .vendor import detection, tools
from .vendor.model import UNet
from .vendor.rectification import warp

_CHECKPOINT = os.path.join(os.path.dirname(__file__), "checkpoint", "checkpoint_aug_epoch70.pth")
_RESIZE = 256
# Fragment-bridging: dilate by N px before skeletonizing to join broken line fragments.
# Tested on the sample set: dilation creates duplicate/parallel skeleton branches and did
# NOT reliably lengthen lines, so it is OFF. The truncation is inherent to the pretrained
# model; the real fix is fine-tuning/retraining (see ARCHITECTURE.md §2.4). Kept configurable.
_BRIDGE = 0

# Load the U-Net once at import (CPU) and reuse across requests.
_net = UNet(n_channels=3, n_classes=1)
_net.load_state_dict(torch.load(_CHECKPOINT, map_location=torch.device("cpu")))
_net.eval()


class PalmNotDetected(Exception):
    """Raised when no hand/palm can be located in the image."""


def _line_thresholds(warped_mini_path: str):
    """Replicate the original length-threshold logic from measurement.py."""
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.5) as hands:
        image = cv2.flip(cv2.imread(warped_mini_path), 1)
        h, w, _ = image.shape
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            raise PalmNotDetected("Hand landmarks not found in warped image.")
        lm = results.multi_hand_landmarks[0].landmark
        hand = "unknown"
        if results.multi_handedness:
            # image was horizontally flipped, so invert the reported label
            label = results.multi_handedness[0].classification[0].label
            hand = "left" if label == "Right" else "right"
        zero, one = lm[0].y, lm[1].y
        five, nine, thirteen = lm[5].x, lm[9].x, lm[13].x
        return {
            "heart_x": w * (1 - (nine + (five - nine) * 2 / 5)),
            "head_x": w * (1 - (thirteen + (nine - thirteen) / 3)),
            "life_y": h * (one + (zero - one) / 3),
            "hand": hand,
        }


def _curvature(points: list[tuple[float, float]]) -> bool | None:
    """True if the polyline bends noticeably from the straight chord between its ends."""
    if len(points) < 3:
        return None
    (x1, y1), (x2, y2) = points[0], points[-1]
    chord = math.hypot(x2 - x1, y2 - y1)
    if chord < 1e-6:
        return None
    max_dev = 0.0
    for (px, py) in points[1:-1]:
        dev = abs((y2 - y1) * px - (x2 - x1) * py + x2 * y1 - y2 * x1) / chord
        max_dev = max(max_dev, dev)
    return (max_dev / chord) > 0.12


def _line_feature(points_raw, length: LineLength, fork_end: bool | None = None) -> LineFeature:
    if not points_raw:
        return LineFeature(present=False, confidence=0.0)
    pts = [tuple(reversed(p[:2])) for p in points_raw]  # (y,x) -> (x,y)
    n = len(pts)
    # Confidence: more detected points => stronger, cleaner detection.
    confidence = max(0.3, min(0.95, n / 60.0))
    return LineFeature(
        present=True,
        length=length,
        curved=_curvature(pts),
        fork_end=fork_end,
        point_count=n,
        confidence=round(confidence, 2),
    )


def extract_features(image_path: str) -> FeatureVector:
    """Run the full CV pipeline on an image file and return a FeatureVector."""
    with tempfile.TemporaryDirectory() as work:
        clean = os.path.join(work, "clean.jpg")
        warped = os.path.join(work, "warped.jpg")
        warped_clean = os.path.join(work, "warped_clean.jpg")
        warped_mini = os.path.join(work, "warped_mini.jpg")
        warped_clean_mini = os.path.join(work, "warped_clean_mini.jpg")
        palm_lines = os.path.join(work, "palm_lines.png")

        tools.remove_background(image_path, clean)
        if warp(image_path, warped) is None:
            raise PalmNotDetected("Could not detect a hand to rectify the palm.")
        tools.remove_background(warped, warped_clean)
        tools.resize(warped, warped_clean, warped_mini, warped_clean_mini, _RESIZE)

        detection.detect(_net, warped_clean, palm_lines, _RESIZE)
        analyzed = line_analysis.analyze(palm_lines, _RESIZE, bridge=_BRIDGE)

        thr = _line_thresholds(warped_mini)
        fv = FeatureVector(hand=thr["hand"])

        # --- heart / head / life: length from landmark thresholds (original logic) ---
        heart, head, life = analyzed["heart"], analyzed["head"], analyzed["life"]
        if heart:
            tip = tuple(reversed(heart["points"][0][:2]))
            length = LineLength.long if tip[0] < thr["heart_x"] else LineLength.short
            fv.heart = _line_feature(heart["points"], length, heart["fork_end"])
        if head:
            tip = tuple(reversed(head["points"][-1][:2]))
            length = LineLength.long if tip[0] > thr["head_x"] else LineLength.short
            fv.head = _line_feature(head["points"], length, head["fork_end"])
        if life:
            tip = tuple(reversed(life["points"][-1][:2]))
            length = LineLength.long if tip[1] > thr["life_y"] else LineLength.short
            fv.life = _line_feature(life["points"], length, life["fork_end"])

        # --- fate (Bhagya Rekha): length from vertical extent (wrist -> fingers) ---
        fate = analyzed["fate"]
        if fate:
            ys = [p[0] for p in fate["points"]]
            y_span = max(ys) - min(ys)
            length = LineLength.long if y_span > 0.45 * _RESIZE else LineLength.short
            fv.fate = _line_feature(fate["points"], length, fate["fork_end"])

        present = [getattr(fv, n) for n in ("heart", "head", "life", "fate")]
        if not any(l.present for l in (fv.heart, fv.head, fv.life)):
            fv.quality_flags.append("major_lines_incomplete")

        confs = [l.confidence for l in present if l.present]
        fv.overall_confidence = round(sum(confs) / len(confs), 2) if confs else 0.0
        return fv
