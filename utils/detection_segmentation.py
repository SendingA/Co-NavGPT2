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


class Object_Detection_and_Segmentation():
    r""" YOLO-World + SAM, augmented with a thermal flame detection.

    If the class list passed in contains ``"fire"`` we derive a smoke-
    invariant fire detection from the voxel thermal flame mask and merge
    its boxes/masks into the YOLO-derived detections. This is more robust
    than relying on YOLO-World alone for the synthetic flames produced by
    the fire-scene simulator.
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
                detections are derived from this (smoke-invariant) voxel
                thermal mask.
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

        # ----------------------- 3) Fire detection --------------------
        # Fire detections are derived from the voxel thermal flame mask,
        # which is unaffected by smoke.
        if self.fire_class_id >= 0 and thermal_flame_mask is not None:
            from utils.smoke_perception import thermal_mask_to_detections

            target_hw = (
                masks_np.shape[1:] if masks_np is not None else image.shape[:2]
            )
            fire_xyxy, fire_masks, fire_scores = thermal_mask_to_detections(
                np.asarray(thermal_flame_mask, dtype=np.float32),
                target_hw=target_hw,
            )

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
