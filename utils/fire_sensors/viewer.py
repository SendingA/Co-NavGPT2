"""Lightweight OpenCV dashboard window driven from the main thread.

Important: ``cv2.imshow`` uses Qt under the hood and **requires** the
GUI to live on the main thread, otherwise you get
``QApplication was not created in the main() thread`` and a segfault.

So this viewer does **not** spawn a worker thread anymore. Instead, the
main loop calls :meth:`update` once per step. Internally we throttle the
imshow rate so we don't burn CPU. The first failure (e.g. headless
host without a display, or a Qt clash with habitat-sim) disables the
window and falls back to writing the latest frame to ``<dump>/live.png``
which you can watch with e.g. ``feh --reload 0.1 live.png``.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import cv2
import numpy as np


class FireSensorViewer:
    def __init__(
        self,
        window_name: str = "Fire Sensors",
        fps: float = 10.0,
        fallback_path: Optional[str] = None,
    ) -> None:
        self.window_name = window_name
        self._period = 1.0 / max(fps, 1e-3)
        self._last_show = 0.0
        self._enabled = True
        self._created = False
        self._fallback_path = fallback_path
        if fallback_path is not None:
            os.makedirs(os.path.dirname(fallback_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    @classmethod
    def start(
        cls,
        window_name: str = "Fire Sensors",
        fps: float = 10.0,
        fallback_path: Optional[str] = None,
    ) -> "FireSensorViewer":
        """Backward-compatible factory. Window is created lazily on
        first :meth:`update` (must be called from the main thread)."""
        return cls(window_name=window_name, fps=fps, fallback_path=fallback_path)

    # ------------------------------------------------------------------
    def update(self, frame: Optional[np.ndarray]) -> None:
        """Display ``frame`` (BGR uint8) in the window. No-op when
        disabled or when the host has no GUI backend (falls back to
        writing ``fallback_path`` if configured)."""
        if frame is None:
            return
        now = time.monotonic()
        if now - self._last_show < self._period:
            return
        self._last_show = now

        if self._enabled:
            try:
                if not self._created:
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    self._created = True
                cv2.imshow(self.window_name, frame)
                cv2.waitKey(1)
                return
            except cv2.error as e:
                print(f"[FireSensorViewer] disabled (cv2 error: {e})")
                self._enabled = False

        if self._fallback_path is not None:
            try:
                cv2.imwrite(self._fallback_path, frame)
            except cv2.error:
                pass

    # ------------------------------------------------------------------
    def stop(self) -> None:
        if not self._created:
            return
        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass
        self._created = False
