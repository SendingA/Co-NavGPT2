"""Wire up four yaw-rotated depth sensors on every Habitat agent so we
can stitch a true 360° LIDAR point cloud at runtime.

Why four 90°-HFOV slices and not an EquirectangularSensor?
  * Habitat's pinhole depth sensor has well-defined intrinsics, so the
    back-projection is straightforward (z * pixel_ray).
  * Equirectangular sensors require special back-projection and are not
    supported by every habitat-sim build.

UUIDs used (one set per agent; the same UUIDs are reused by every
agent because the local fork shares one `agent_config` for all
agents - see ``HabitatSim.create_sim_config``)::

    lidar_depth_front   yaw =   0   (looking +X)
    lidar_depth_left    yaw = +π/2  (looking +Y)
    lidar_depth_back    yaw =  π    (looking -X)
    lidar_depth_right   yaw = -π/2  (looking -Y)

The four slices share the same MIN_DEPTH / MAX_DEPTH / NORMALIZE_DEPTH
as the agent's primary depth sensor.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# Public mapping (uuid -> yaw radians). Order is also the docstring order.
LIDAR_DEPTH_YAW: Dict[str, float] = {
    "lidar_depth_front": 0.0,
    "lidar_depth_left":  np.pi / 2,
    "lidar_depth_back":  np.pi,
    "lidar_depth_right": -np.pi / 2,
}
LIDAR_DEPTH_UUIDS: List[str] = list(LIDAR_DEPTH_YAW.keys())


# ---------------------------------------------------------------------------
# Habitat config injection
# ---------------------------------------------------------------------------


def install_lidar_depth_sensors(
    config,
    base_depth_cfg,
    resolution: int = 320,
    num_agents: int = 1,
) -> None:
    """Add 4 yaw-rotated DEPTH sensors to ``config.SIMULATOR`` and to
    every agent's ``SENSORS`` list.

    Must be called between ``config.defrost()`` and ``config.freeze()``.
    The same sensor specs end up on every agent because the local
    Habitat fork reuses ``AGENT_0``'s sensor specs for all agents (see
    ``HabitatSim.create_sim_config``).
    """
    res = int(resolution)

    for uuid, yaw in LIDAR_DEPTH_YAW.items():
        # Clone the existing depth sensor config so we inherit
        # MIN/MAX/NORMALIZE.
        cfg_name = uuid.upper()  # e.g. LIDAR_DEPTH_FRONT
        sensor_cfg = base_depth_cfg.clone()
        sensor_cfg.UUID = uuid
        # Each slice is a square 90° HFOV camera so 4 slices = 360°.
        sensor_cfg.HFOV = 90
        sensor_cfg.WIDTH = res
        sensor_cfg.HEIGHT = res
        # Position: same height as the primary depth sensor.
        sensor_cfg.POSITION = list(base_depth_cfg.POSITION)
        # ORIENTATION uses Euler XYZ in radians: rotate around the Y
        # (up) axis to point each slice in a different yaw direction.
        sensor_cfg.ORIENTATION = [0.0, float(yaw), 0.0]
        sensor_cfg.TYPE = base_depth_cfg.TYPE
        setattr(config.SIMULATOR, cfg_name, sensor_cfg)

    # Append to AGENT_0's SENSORS list (other agents reuse this list).
    sensor_names = list(config.SIMULATOR.AGENT_0.SENSORS)
    for uuid in LIDAR_DEPTH_UUIDS:
        cfg_name = uuid.upper()
        if cfg_name not in sensor_names:
            sensor_names.append(cfg_name)
    config.SIMULATOR.AGENT_0.SENSORS = sensor_names

    # AGENT_i may have its own SENSORS list in some yaml variants - if
    # so, append there too. Multi-agent objectnav config in this repo
    # only defines AGENT_0.SENSORS, but be defensive.
    for i in range(num_agents):
        attr = f"AGENT_{i}"
        if hasattr(config.SIMULATOR, attr):
            agent_cfg = getattr(config.SIMULATOR, attr)
            if hasattr(agent_cfg, "SENSORS"):
                names = list(agent_cfg.SENSORS)
                changed = False
                for uuid in LIDAR_DEPTH_UUIDS:
                    cfg_name = uuid.upper()
                    if cfg_name not in names and cfg_name in [
                        s for s in dir(config.SIMULATOR) if s.startswith("LIDAR_DEPTH_")
                    ]:
                        names.append(cfg_name)
                        changed = True
                if changed:
                    agent_cfg.SENSORS = names


# ---------------------------------------------------------------------------
# 4-slice → 360° point cloud
# ---------------------------------------------------------------------------


def _depth_to_local_xyz(
    depth_m: np.ndarray,
    hfov_deg: float,
    max_range_m: float,
    stride: int = 2,
) -> np.ndarray:
    """Back-project a depth image to a (N, 3) cloud in *camera local* frame.

    Camera local frame: X forward, Y left, Z up.
    """
    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    H, W = depth_m.shape
    fx = (W / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx
    cx, cy = W / 2.0, H / 2.0

    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_m[::stride, ::stride]
    valid = (z > 0) & (z < max_range_m)
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]

    xc = (xs - cx) * z / fx
    yc = (ys - cy) * z / fy
    X = z
    Y = -xc
    Z = -yc
    return np.stack([X, Y, Z], axis=-1).astype(np.float32)


def _rotate_yaw(points_xyz: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Rotate (N, 3) points around the Z (up) axis by ``yaw_rad``."""
    if points_xyz.size == 0:
        return points_xyz
    c, s = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return points_xyz @ R.T


def stitch_lidar_360(
    obs: Dict[str, np.ndarray],
    max_range_m: float,
    stride: int = 2,
    normalize_depth: bool = True,
    min_depth_m: float = 0.0,
    depth_norm_max_m: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Stitch the four lidar_depth_* slices in ``obs`` into one cloud.

    Returns ``None`` when the obs dict has none of the lidar UUIDs (so
    callers can fall back to the legacy single-depth path).

    Args:
        obs: a single agent's observation dict from ``env.step``.
        max_range_m: clip-distance for back-projection (LIDAR's own
            maximum range; pixels beyond this are dropped).
        stride: depth pixel stride (>=1). Larger = sparser cloud.
        normalize_depth: True if Habitat returned depth in [0, 1]
            (matches NORMALIZE_DEPTH = True in the yaml).
        min_depth_m: lower bound when un-normalising depth.
        depth_norm_max_m: the MAX_DEPTH that Habitat used to normalise
            depth in the first place (= yaml DEPTH_SENSOR.MAX_DEPTH).
            Defaults to ``max_range_m`` for backward compatibility.
    """
    found = [uuid for uuid in LIDAR_DEPTH_UUIDS if uuid in obs]
    if not found:
        return None

    span = float(
        depth_norm_max_m if depth_norm_max_m is not None else max_range_m
    ) - float(min_depth_m)

    clouds = []
    for uuid in found:
        depth = np.asarray(obs[uuid])
        if normalize_depth:
            depth = depth * span + float(min_depth_m)
        local = _depth_to_local_xyz(
            depth, hfov_deg=90.0, max_range_m=max_range_m, stride=stride
        )
        rotated = _rotate_yaw(local, LIDAR_DEPTH_YAW[uuid])
        clouds.append(rotated)

    if not clouds:
        return None
    return np.concatenate(clouds, axis=0).astype(np.float32)
