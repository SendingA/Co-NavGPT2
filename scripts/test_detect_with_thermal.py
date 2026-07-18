"""End-to-end: detection pipeline picks up fire from thermal even when RGB
has no flame-coloured pixels.

We feed Object_Detection_and_Segmentation a uniform gray BGR image (no
flames at all in colour space) plus a thermal flame mask. The detector
should still emit a fire detection sourced from thermal.

Run with::

    python scripts/test_detect_with_thermal.py
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# We don't need YOLO/SAM weights for this test - stub them out before import.
import torch  # noqa: E402

class _DummyResult:
    class _Boxes:
        def __init__(self):
            self.conf = torch.zeros((0,))
            self.cls = torch.zeros((0,))
            self.xyxy = torch.zeros((0, 4))
        def cpu(self):
            return self

    def __init__(self):
        self.boxes = self._Boxes()


class _DummyYOLO:
    def __init__(self, *a, **k):
        pass

    def to(self, device):
        return self

    def set_classes(self, classes):
        self.classes = classes

    def predict(self, image, conf=0.1, verbose=False):
        return [_DummyResult()]


class _DummySAM:
    def __init__(self, *a, **k):
        pass

    def to(self, device):
        return self

    def predict(self, image, bboxes=None, verbose=False):
        return []


sys.modules["ultralytics"] = types.ModuleType("ultralytics")
sys.modules["ultralytics"].YOLO = _DummyYOLO
sys.modules["ultralytics"].SAM = _DummySAM

from utils.detection_segmentation import Object_Detection_and_Segmentation  # noqa: E402


def main() -> int:
    args = types.SimpleNamespace()
    classes = ["chair", "bed", "fire"]
    det = Object_Detection_and_Segmentation(args, classes, device="cpu")

    H, W = 240, 320
    # Pure gray BGR: no flame-orange pixels, HSV detector must return zero.
    image = np.full((H, W, 3), 180, dtype=np.uint8)

    mask = np.zeros((H, W), dtype=np.float32)
    mask[100:140, 150:190] = 1.0

    out = det.detect(image, thermal_flame_mask=mask)
    n = len(out.xyxy)
    print(f"detections={n}, class_ids={out.class_id.tolist()}, "
          f"confidence={[float(x) for x in out.confidence]}")

    if n != 1:
        print("FAIL: expected exactly one fire detection from thermal")
        return 1
    if int(out.class_id[0]) != classes.index("fire"):
        print("FAIL: detection class id is not 'fire'")
        return 1
    if float(out.confidence[0]) < 0.9:
        print("FAIL: thermal-derived confidence too low")
        return 1
    print("OK: thermal source gave a fire detection on a flame-free RGB")

    # ---- Person from thermal_human_mask (smoke-invariant) --------------
    classes_p = ["chair", "bed", "person", "fire"]
    det_p = Object_Detection_and_Segmentation(args, classes_p, device="cpu")
    human_mask = np.zeros((H, W), dtype=np.float32)
    human_mask[80:170, 140:180] = 1.0   # an upright person-ish blob
    out_p = det_p.detect(image, thermal_human_mask=human_mask)
    np_ = len(out_p.xyxy)
    print(f"person detections={np_}, class_ids={out_p.class_id.tolist()}")
    if np_ != 1 or int(out_p.class_id[0]) != classes_p.index("person"):
        print("FAIL: expected exactly one person detection from thermal")
        return 1
    print("OK: thermal source gave a person detection on a flame-free RGB")

    # ---- Both masks together ------------------------------------------
    out_both = det_p.detect(
        image, thermal_flame_mask=mask, thermal_human_mask=human_mask
    )
    ids = sorted(int(c) for c in out_both.class_id)
    expected = sorted([classes_p.index("person"), classes_p.index("fire")])
    print(f"combined class_ids={ids} (expect {expected})")
    if ids != expected:
        print("FAIL: expected one fire + one person detection")
        return 1
    print("OK: fire + person both detected from thermal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
