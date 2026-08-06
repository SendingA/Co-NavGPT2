"""Voxel-driven smoky-RGB / Thermal camera.

This sensor is the **observation** counterpart of the voxel fire world.
The world model (``utils.fire_world``) owns the flame/smoke/temperature
fields and how they evolve in time; this sensor takes a snapshot of
those fields at the current step and ray-marches it from the agent's
camera pose, returning a dict with the smoky RGB image plus thermal
channels.

This is the sole RGB / Thermal source used by
:class:`FireSensorSuite`, which requires a :class:`FireScene` to be
bound before observation.
"""
from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np

from .base import BaseSensor
from ..voxel_render import VoxelRenderParams, volumetric_composite


class VoxelSmokeSensor(BaseSensor):
    """Observation of a :class:`utils.fire_world.scene.FireScene`.

    The sensor needs the camera intrinsics (passed at construction) and
    a per-frame agent pose / step counter (passed via ``process()``
    keyword args). The fire scene itself is stored on the sensor so the
    suite can swap it without re-allocating the camera matrices.
    """

    name = "voxel_smoke"

    def __init__(
        self,
        cfg,
        rng: Optional[np.random.Generator] = None,
        *,
        camera_K=None,
        scene=None,
    ) -> None:
        super().__init__(cfg, rng)
        if camera_K is None:
            raise ValueError("VoxelSmokeSensor needs camera intrinsics K")
        self.camera_K = camera_K
        self.scene = scene  # FireScene or None until bound
        self._torch_failed = False
        self._torch_warning_emitted = False

    # ------------------------------------------------------------------
    def bind_scene(self, scene) -> None:
        """Attach (or replace) the fire-world scene this sensor observes."""
        self.scene = scene
        self._torch_failed = False
        self._torch_warning_emitted = False

    # ------------------------------------------------------------------
    def _wants_torch(self) -> bool:
        """Resolve the configured backend without importing Torch on NumPy."""
        voxel = getattr(self.cfg, "voxel", None)
        if voxel is None:
            return False
        backend = str(getattr(voxel, "render_backend", "auto")).lower()
        if backend not in {"auto", "numpy", "torch"}:
            raise ValueError(
                f"unsupported FireWorld render backend {backend!r}; "
                "expected auto, numpy or torch"
            )
        if backend == "numpy" or self._torch_failed:
            return False
        if backend == "torch":
            return True

        # Backend=auto with an explicit device is an intentional Torch
        # request. With device=auto, only select Torch when CUDA is visible;
        # Torch-on-CPU is useful for tests but slower than the NumPy reference.
        requested_device = str(
            getattr(voxel, "render_device", "auto")
        ).lower()
        if requested_device != "auto":
            return True
        try:
            import torch
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    # ------------------------------------------------------------------
    def _params(self) -> VoxelRenderParams:
        v = self.cfg.voxel
        return VoxelRenderParams(
            max_depth_m=float(self.cfg.max_depth_m),
            n_steps=int(v.n_steps),
            smoke_k_ext=float(v.smoke_k_ext),
            smoke_color_rgb=tuple(v.smoke_color_rgb),
            flame_threshold=float(v.flame_threshold),
            flame_emission_gain=float(v.flame_emission_gain),
            flame_k_ext=float(v.flame_k_ext),
            flame_glow_ksize=int(v.flame_glow_ksize),
            flame_glow_gain=float(v.flame_glow_gain),
            flame_smoke_passthrough=float(v.flame_smoke_passthrough),
            flame_smoke_displacement=float(v.flame_smoke_displacement),
            flame_surface_reveal=float(v.flame_surface_reveal),
            flame_highlight_compression=float(
                v.flame_highlight_compression
            ),
            thermal_color_blend=float(v.thermal_color_blend),
            thermal_surface_start=float(v.thermal_surface_start),
            thermal_air_coupling=float(v.thermal_air_coupling),
            render_scale=float(v.render_scale),
            flame_noise_strength=float(getattr(v, "flame_noise_strength", 0.75)),
            flame_edge_break=float(getattr(v, "flame_edge_break", 1.05)),
            flame_color_jitter=float(getattr(v, "flame_color_jitter", 0.32)),
            flame_time_speed=float(getattr(v, "flame_time_speed", 12.0)),
            smoke_noise_strength=float(getattr(v, "smoke_noise_strength", 0.24)),
        )

    # ------------------------------------------------------------------
    def process(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        *,
        agent_state=None,
        robot_step: int = 0,
        t_sim_s: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Render the current fire scene from the agent's camera pose.

        Falls back to a passthrough (clean RGB, no smoke / flame) when
        no ``scene`` is bound. This keeps the interface non-fatal for
        callers that toggle ``--fire_world=0``.
        """
        if self.scene is None or agent_state is None:
            H, W = (depth_m.shape[:2] if depth_m.ndim >= 2 else rgb.shape[:2])
            zeros = np.zeros((H, W), dtype=np.float32)
            return {
                "image": rgb.copy(),
                "transmittance": np.ones((H, W), dtype=np.float32),
                "flame_mask": zeros,
                "thermal_image": np.zeros((H, W, 3), dtype=np.uint8),
                "thermal_temperature": np.full((H, W),
                    float(getattr(self.scene, "ambient_c", 25.0)) if self.scene else 25.0,
                    dtype=np.float32),
                "t_sim_s": float(0.0),
                "robot_step": int(robot_step),
            }

        scene = self.scene
        cam_pos, R = scene.camera_pose(agent_state)
        # Use the unified ``t_sim()`` API so the renderer reads the
        # wallclock when the scene was built with mode="wallclock"
        # (default). Step mode falls back to the legacy mapping.
        t_sim = (
            scene.t_sim(int(robot_step))
            if t_sim_s is None
            else float(t_sim_s)
        )
        out = None
        if self._wants_torch():
            try:
                from ..voxel_render_torch import (
                    resolve_torch_device,
                    shared_scene_cache,
                    volumetric_composite_torch,
                )

                device = resolve_torch_device(
                    getattr(self.cfg.voxel, "render_device", "auto")
                )
                frame_index = scene.fw.frame_index(t_sim)
                # Use native FP16 timeline views here. Calling scene.query()
                # would allocate three FP32 CPU arrays before uploading them.
                flame_field = scene.fw.flame[frame_index]
                smoke_field = scene.fw.smoke[frame_index]
                temp_field = scene.fw.temp[frame_index]
                cache = shared_scene_cache(scene)
                out = volumetric_composite_torch(
                    rgb_clean=rgb,
                    depth_m=depth_m,
                    cam_pos_world=cam_pos.astype(np.float32),
                    R_cam2world=R.astype(np.float32),
                    flame_field=flame_field,
                    smoke_field=smoke_field,
                    temp_field=temp_field,
                    origin=scene.origin,
                    voxel_m=scene.voxel_m,
                    grid_shape=scene.shape,
                    ambient_c=scene.ambient_c,
                    camera_K=self.camera_K,
                    params=self._params(),
                    t_sim=float(t_sim),
                    device=str(device),
                    volume_dtype=str(
                        getattr(
                            self.cfg.voxel, "render_dtype", "float16"
                        )
                    ),
                    max_sample_points=int(
                        getattr(
                            self.cfg.voxel,
                            "max_sample_points",
                            2_000_000,
                        )
                    ),
                    cache=cache,
                    frame_key=(
                        scene.scene_id,
                        scene.plan_id,
                        int(frame_index),
                    ),
                )
                out["render_backend"] = "torch"
                out["render_device"] = str(device)
                out["render_frame_index"] = int(frame_index)
                out["render_cache_hits"] = int(cache.hits)
                out["render_cache_uploads"] = int(cache.uploads)
            except (ImportError, RuntimeError, ValueError) as exc:
                # A missing driver, unsupported GPU kernel or OOM must not
                # destroy an otherwise valid navigation episode. Disable the
                # Torch path for this sensor after the first failure.
                self._torch_failed = True
                if not self._torch_warning_emitted:
                    warnings.warn(
                        "FireWorld Torch rendering failed; falling back to "
                        f"NumPy for this sensor: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._torch_warning_emitted = True

        if out is None:
            flame_field, smoke_field, temp_field = scene.query(t_sim)
            out = volumetric_composite(
                rgb_clean=rgb,
                depth_m=depth_m,
                cam_pos_world=cam_pos.astype(np.float32),
                R_cam2world=R.astype(np.float32),
                flame_field=flame_field,
                smoke_field=smoke_field,
                temp_field=temp_field,
                origin=scene.origin,
                voxel_m=scene.voxel_m,
                grid_shape=scene.shape,
                ambient_c=scene.ambient_c,
                camera_K=self.camera_K,
                params=self._params(),
                t_sim=float(t_sim),
            )
            out["render_backend"] = "numpy"
            out["render_device"] = "cpu"
            if hasattr(scene, "fw"):
                out["render_frame_index"] = int(
                    scene.fw.frame_index(t_sim)
                )
        out["t_sim_s"] = float(t_sim)
        out["robot_step"] = int(robot_step)
        return out
