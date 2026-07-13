"""Fire-scene multi-modal sensor simulator package.

The smoky RGB and thermal images come from the voxel renderer
(:class:`~utils.fire_sensors.sensors.voxel_smoke.VoxelSmokeSensor`);
depth / radar / lidar model their own smoke degradation. The
:class:`FireSensorSuite` orchestrates them, persists per-step files and
renders a dashboard.

Public API (kept stable for callers in main.py / main_vec.py)::

    from utils.fire_sensors import FireSensorConfig, FireSensorSuite
"""
from .config import FireSensorConfig
from .suite import FireSensorSuite
from .viewer import FireSensorViewer

__all__ = ["FireSensorConfig", "FireSensorSuite", "FireSensorViewer"]
