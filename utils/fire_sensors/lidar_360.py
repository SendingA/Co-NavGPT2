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
    base_depth_cfg=None,
    resolution: int = 320,
    num_agents: int = 1,
) -> None:
    """Add 4 yaw-rotated DEPTH sensors under every agent in the config.

    Habitat-Lab 0.3.3 (DictConfig)::

        config.habitat.simulator.agents.<name>.sim_sensors.<uuid>

    Requires the caller to be inside a ``habitat.config.read_write``
    block so the DictConfig is mutable.

    ``base_depth_cfg`` is ignored for H3.3 (we always copy the main
    agent's ``depth_sensor`` block); it's kept in the signature for
    backwards compatibility with older callers.
    """
    from copy import deepcopy

    from omegaconf import OmegaConf

    if not hasattr(config, "habitat"):
        raise RuntimeError(
            "install_lidar_depth_sensors requires a Habitat 3.3 DictConfig; "
            "the legacy YACS support was dropped in the H3.3 migration."
        )

    res = int(resolution)
    sim_cfg = config.habitat.simulator
    for agent_name in sim_cfg.agents_order:
        agent_cfg = sim_cfg.agents[agent_name]
        if "depth_sensor" not in agent_cfg.sim_sensors:
            continue
        base = OmegaConf.to_container(
            agent_cfg.sim_sensors.depth_sensor, resolve=True
        )
        base_position = list(base.get("position", [0.0, 0.88, 0.0]))
        for uuid, yaw in LIDAR_DEPTH_YAW.items():
            sensor_cfg = deepcopy(base)
            sensor_cfg["uuid"] = uuid
            sensor_cfg["hfov"] = 90
            sensor_cfg["width"] = res
            sensor_cfg["height"] = res
            sensor_cfg["position"] = list(base_position)
            sensor_cfg["orientation"] = [0.0, float(yaw), 0.0]
            agent_cfg.sim_sensors[uuid] = OmegaConf.create(sensor_cfg)


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
        depth = np.asarray(obs[uuid], dtype=np.float32)
        if normalize_depth:
            # Habitat encodes no-return rays at the normalized upper bound.
            # Converting those 1.0 values to metric depth before filtering
            # creates an artificial ring at the camera's max depth.
            no_return = ~np.isfinite(depth) | (depth >= 1.0 - 1e-6)
            depth = depth * span + float(min_depth_m)
            depth = depth.copy()
            depth[no_return] = 0.0
        local = _depth_to_local_xyz(
            depth, hfov_deg=90.0, max_range_m=max_range_m, stride=stride
        )
        rotated = _rotate_yaw(local, LIDAR_DEPTH_YAW[uuid])
        clouds.append(rotated)

    if not clouds:
        return None
    return np.concatenate(clouds, axis=0).astype(np.float32)
