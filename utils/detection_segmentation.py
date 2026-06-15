import numpy as np
import torch
import torch.nn.functional as F
import os
import time
import cv2
import torchvision
import supervision as sv

from PIL import Image

from ultralytics import YOLO
from ultralytics import SAM


# ---------------------------------------------------------------------------
# HSV-based flame detector
# ---------------------------------------------------------------------------
# YOLO-World does not reliably pick up the synthetic flames rendered by the
# fire-scene simulator. The thermal sensor in
# ``utils/fire_sensors/sensors/thermal.py`` already isolates flames via two
# HSV ranges (covering the hue wrap-around for orange/red). We reuse the same
# ranges here so the RGB perception pipeline can produce a "fire" detection
# that the agents' goal-checking logic can consume just like a YOLO box.
FLAME_HSV_LOW1 = np.array([0, 100, 200], dtype=np.uint8)
FLAME_HSV_HIGH1 = np.array([35, 255, 255], dtype=np.uint8)
FLAME_HSV_LOW2 = np.array([160, 100, 200], dtype=np.uint8)
FLAME_HSV_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)

# Connected components below this many pixels are treated as noise.
_FLAME_MIN_AREA = 60
# Confidence assigned to HSV-derived detections. Set above the default
# ``sem_threshold`` (0.85) so flames pass the goal filter.
_FLAME_CONFIDENCE = 0.95


def detect_flames_hsv(image_rgb):
    """Return (boxes_xyxy, masks, scores) for HSV-detected flame regions.

    Args:
        image_rgb: (H, W, 3) uint8 RGB image.

    Returns:
        boxes: (N, 4) float32 xyxy, possibly empty.
        masks: (N, H, W) bool, possibly empty.
        scores: (N,) float32, possibly empty.
    """
    if image_rgb is None or image_rgb.size == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0, 0, 0), bool),
                np.zeros((0,), np.float32))

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, FLAME_HSV_LOW1, FLAME_HSV_HIGH1)
    mask2 = cv2.inRange(hsv, FLAME_HSV_LOW2, FLAME_HSV_HIGH2)
    flame_mask = cv2.bitwise_or(mask1, mask2)
    flame_mask = cv2.morphologyEx(
        flame_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    n_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(
        flame_mask, connectivity=8
    )

    H, W = flame_mask.shape
    boxes = []
    masks = []
    scores = []
    # label 0 is background
    for lbl in range(1, n_lbl):
        x, y, w, h, area = stats[lbl]
        if area < _FLAME_MIN_AREA:
            continue
        boxes.append([x, y, x + w, y + h])
        masks.append(lbl_img == lbl)
        scores.append(_FLAME_CONFIDENCE)

    if not boxes:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0, H, W), bool),
                np.zeros((0,), np.float32))

    return (
        np.asarray(boxes, dtype=np.float32),
        np.stack(masks, axis=0).astype(bool),
        np.asarray(scores, dtype=np.float32),
    )


class Object_Detection_and_Segmentation():
    r""" YOLO-World + SAM, augmented with an HSV flame fallback.

    If the class list passed in contains ``"fire"`` we additionally run a
    color-based flame detector on every frame and merge its boxes/masks into
    the YOLO-derived detections. This is more robust than relying on
    YOLO-World alone for the synthetic flames produced by the fire-scene
    simulator.
    """

    def __init__(self, args, classes, device):
        self.args = args
        self.device = device
        self.classes = list(classes)

        self.sam_predictor = SAM('mobile_sam.pt').to(self.device)

        # Initialize a YOLO-World model
        self.yolo_model_w_classes = YOLO('yolov8l-world.pt').to(self.device)
        self.yolo_model_w_classes.set_classes(self.classes)

        # Resolve fire id once. -1 means caller did not opt in.
        self.fire_class_id = (
            self.classes.index("fire") if "fire" in self.classes else -1
        )

    # ------------------------------------------------------------------
    def detect(self, image, thermal_flame_mask=None):
        """Run YOLO-World on ``image`` and merge fire detections.

        Args:
            image: HxWx3 uint8 BGR (matches the rest of the pipeline).
            thermal_flame_mask: optional float32 in [0, 1], shape (H, W).
                If provided and the ``fire`` class is registered, fire
                detections are derived from this mask (smoke-invariant)
                instead of running HSV on the smoky RGB.
        """
        # ----------------------- 1) YOLO-World ------------------------
        yolo_s_time = time.time()
        with torch.no_grad():
            yolo_results_w_classes = self.yolo_model_w_classes.predict(
                image, conf=0.1, verbose=False
            )
        yolo_e_time = time.time()

        confidences = yolo_results_w_classes[0].boxes.conf.cpu().numpy()
        detection_class_ids = (
            yolo_results_w_classes[0].boxes.cls.cpu().numpy().astype(int)
        )
        xyxy_tensor = yolo_results_w_classes[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        # ----------------------- 2) SAM masks for YOLO boxes ----------
        masks_np = None
        if len(confidences) > 0:
            with torch.no_grad():
                sam_out = self.sam_predictor.predict(
                    image, bboxes=xyxy_tensor, verbose=False
                )
            masks_tensor = sam_out[0].masks.data
            masks_np = masks_tensor.cpu().numpy().astype(bool)

        # ----------------------- 3) Fire fallback ---------------------
        # Prefer thermal: it is unaffected by smoke. Fall back to the HSV
        # detector on the (possibly smoky) RGB only when thermal is absent.
        if self.fire_class_id >= 0:
            if thermal_flame_mask is not None:
                from utils.smoke_perception import thermal_mask_to_detections

                target_hw = (
                    masks_np.shape[1:] if masks_np is not None else image.shape[:2]
                )
                fire_xyxy, fire_masks, fire_scores = thermal_mask_to_detections(
                    np.asarray(thermal_flame_mask, dtype=np.float32),
                    target_hw=target_hw,
                )
            else:
                fire_xyxy, fire_masks, fire_scores = detect_flames_hsv(image)

            if len(fire_xyxy) > 0:
                # Align mask spatial dims with YOLO/SAM masks if any.
                if masks_np is not None and masks_np.shape[1:] != fire_masks.shape[1:]:
                    target_h, target_w = masks_np.shape[1:]
                    resized = np.zeros(
                        (fire_masks.shape[0], target_h, target_w), dtype=bool
                    )
                    for i, m in enumerate(fire_masks):
                        resized[i] = cv2.resize(
                            m.astype(np.uint8),
                            (target_w, target_h),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    fire_masks = resized

                fire_class_ids = np.full(
                    (len(fire_xyxy),), self.fire_class_id, dtype=int
                )

                if len(confidences) > 0:
                    xyxy_np = np.concatenate([xyxy_np, fire_xyxy], axis=0)
                    confidences = np.concatenate([confidences, fire_scores])
                    detection_class_ids = np.concatenate(
                        [detection_class_ids, fire_class_ids]
                    )
                    masks_np = np.concatenate([masks_np, fire_masks], axis=0)
                else:
                    xyxy_np = fire_xyxy
                    confidences = fire_scores
                    detection_class_ids = fire_class_ids
                    masks_np = fire_masks

        # ----------------------- 4) Pack detections -------------------
        detections = sv.Detections(
            xyxy=xyxy_np if len(xyxy_np) > 0 else np.zeros((0, 4), np.float32),
            confidence=(
                confidences if len(confidences) > 0 else np.zeros((0,), np.float32)
            ),
            class_id=(
                detection_class_ids
                if len(detection_class_ids) > 0
                else np.zeros((0,), int)
            ),
            mask=masks_np,
        )
        return detections
