"""Fire-scene sensor modules.

Each modality inherits from :class:`BaseSensor`. The smoky RGB and
thermal images are produced by :class:`VoxelSmokeSensor` (ray-marching
the voxel fire world); depth / radar / lidar model their own smoke
degradation from the shared :class:`~utils.fire_sensors.config.SmokeConfig`.
"""
from .base import BaseSensor, density_to_k
from .depth_smoke import SmokeDepthSensor
from .radar import RadarSensor
from .lidar import LidarSensor
from .voxel_smoke import VoxelSmokeSensor

__all__ = [
    "BaseSensor",
    "density_to_k",
    "SmokeDepthSensor",
    "RadarSensor",
    "LidarSensor",
    "VoxelSmokeSensor",
]
