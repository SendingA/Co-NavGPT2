"""Per-sensor simulator modules.

Each sensor implements :class:`BaseSensor` and exposes a single
``process()`` method returning a dict of named arrays. Keeping each
modality in its own file makes it easy to tune, swap or extend.
"""
from .base import BaseSensor
from .rgb_smoke import SmokeRGBSensor
from .depth_smoke import SmokeDepthSensor
from .radar import RadarSensor
from .thermal import ThermalSensor
from .lidar import LidarSensor
from .voxel_smoke import VoxelSmokeSensor

__all__ = [
    "BaseSensor",
    "SmokeRGBSensor",
    "SmokeDepthSensor",
    "RadarSensor",
    "ThermalSensor",
    "LidarSensor",
    "VoxelSmokeSensor",
]
