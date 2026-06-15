"""Fire-scene multi-modal sensor simulator package.

Each modality lives in its own module under ``utils.fire_sensors.sensors``
and inherits from :class:`BaseSensor`. The :class:`FireSensorSuite`
orchestrates them, persists per-step files and renders a 2x3 dashboard
(clean RGB / clean Depth / smoky RGB / smoky Depth / radar / thermal).

Public API (kept stable for callers in main.py / main_vec.py)::

    from utils.fire_sensors import FireSensorConfig, FireSensorSuite
"""
from .config import FireSensorConfig
from .suite import FireSensorSuite
from .viewer import FireSensorViewer

__all__ = ["FireSensorConfig", "FireSensorSuite", "FireSensorViewer"]
