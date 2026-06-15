"""Common base class for fire-scene sensor simulators."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..config import FireSensorConfig


class BaseSensor:
    """Tiny abstract base. Subclasses override :meth:`process`.

    Each sensor receives the **clean** RGB and metric depth produced by
    Habitat and returns a dict of named numpy arrays (one of which is
    typically a uint8 BGR image suitable for direct ``cv2.imwrite``).
    """

    #: short identifier used as a key in the suite output dict and for
    #: filenames written to disk.
    name: str = "sensor"

    def __init__(
        self,
        cfg: FireSensorConfig,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError
