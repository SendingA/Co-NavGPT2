"""Torch backend for FireWorld voxel ray-marching.

The reference renderer in :mod:`utils.fire_sensors.voxel_render` remains the
portable NumPy implementation.  This module moves the expensive batched
trilinear sampling and front-to-back integration to a Torch device while
reusing the same CPU resize, glow and thermal-display post-processing.

Only the active timeline frame is cached on the device.  A cache is attached
to the shared :class:`FireScene`, so every robot viewing the same fire frame
reuses one uploaded volume instead of copying the full timeline to the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Hashable, Optional, Tuple

import numpy as np

from .voxel_render import (
    _FLAME_LUT_RGB,
    _FLAME_LUT_X,
    _hash_noise_field,
    VoxelRenderParams,
    finalize_volumetric_outputs,
)


def _torch_modules():
    """Import Torch lazily so the NumPy backend has no Torch dependency."""
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - project normally has torch
        raise RuntimeError(
            "Torch voxel rendering was requested but PyTorch is not installed"
        ) from exc
    return torch, functional


def resolve_torch_device(requested: str = "auto"):
    """Resolve ``auto``/``cpu``/``cuda:N`` to a usable Torch device.

    Explicit unavailable CUDA devices raise a descriptive error. The sensor
    wrapper catches that error and falls back to the NumPy renderer, so a
    benchmark can continue instead of failing halfway through an episode.
    """
    torch, _ = _torch_modules()
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        value = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"FireWorld render device {device} was requested, but "
                "torch.cuda.is_available() is false"
            )
        index = 0 if device.index is None else int(device.index)
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"FireWorld render device cuda:{index} does not exist; "
                f"visible CUDA device count is {torch.cuda.device_count()}"
            )
        device = torch.device("cuda", index)
    return device


def resolve_volume_dtype(device, requested: str = "float16"):
    """Use compact FP16 volumes on CUDA and safe FP32 sampling on CPU."""
    torch, _ = _torch_modules()
    value = str(requested or "float16").strip().lower()
    if value not in {"float16", "float32"}:
        raise ValueError(
            f"unsupported FireWorld render dtype {requested!r}; "
            "expected float16 or float32"
        )
    if value == "float16" and device.type == "cuda":
        return torch.float16
    return torch.float32


@dataclass
class _FrameEntry:
    key: Hashable
    volume: object


class TorchVolumeCache:
    """One active frame per device/dtype plus small procedural textures."""

    def __init__(self) -> None:
        self._frames: Dict[Tuple[str, str], _FrameEntry] = {}
        self._noise: Dict[Tuple[str, Tuple[int, int, int], int], object] = {}
        self.hits = 0
        self.uploads = 0

    @staticmethod
    def _pack_volume(
        flame_field: np.ndarray,
        smoke_field: np.ndarray,
        temp_field: np.ndarray,
        *,
        device,
        dtype,
    ):
        torch, _ = _torch_modules()
        # FireWorld layout is (X, Y, Z). grid_sample expects (D, H, W),
        # and its normalized coordinate order is (W=x, H=y, D=z).
        packed_xyz = np.stack(
            [smoke_field, flame_field, temp_field], axis=0
        )
        volume = torch.from_numpy(
            np.ascontiguousarray(packed_xyz)
        ).permute(0, 3, 2, 1).unsqueeze(0).contiguous()
        return volume.to(
            device=device,
            dtype=dtype,
            non_blocking=(device.type == "cuda"),
        )

    def get_volume(
        self,
        *,
        frame_key: Hashable,
        flame_field: np.ndarray,
        smoke_field: np.ndarray,
        temp_field: np.ndarray,
        device,
        dtype,
    ):
        slot = (str(device), str(dtype))
        entry = self._frames.get(slot)
        if entry is not None and entry.key == frame_key:
            self.hits += 1
            return entry.volume

        volume = self._pack_volume(
            flame_field,
            smoke_field,
            temp_field,
            device=device,
            dtype=dtype,
        )
        self._frames[slot] = _FrameEntry(frame_key, volume)
        self.uploads += 1
        return volume

    def get_noise(
        self,
        shape: Tuple[int, int, int],
        seed: int,
        device,
    ):
        torch, _ = _torch_modules()
        key = (str(device), tuple(shape), int(seed))
        field = self._noise.get(key)
        if field is None:
            field = torch.from_numpy(
                np.ascontiguousarray(_hash_noise_field(shape, seed))
            ).to(device=device, dtype=torch.float32)
            self._noise[key] = field
        return field

    def clear(self) -> None:
        self._frames.clear()
        self._noise.clear()


def shared_scene_cache(scene) -> TorchVolumeCache:
    """Return the cache shared by every sensor bound to ``scene``."""
    attr = "_fire_torch_volume_cache"
    cache = getattr(scene, attr, None)
    if cache is None:
        cache = TorchVolumeCache()
        setattr(scene, attr, cache)
    return cache


def _sample_world_volume(
    volume,
    points_world,
    *,
    origin,
    voxel_m: float,
    grid_shape: Tuple[int, int, int],
):
    """Sample stacked smoke/flame/temp at ``(..., 3)`` world positions."""
    torch, functional = _torch_modules()
    shape = torch.as_tensor(
        grid_shape, dtype=torch.float32, device=points_world.device
    )
    coord = (
        (points_world - origin) / float(voxel_m)
        - torch.tensor(0.5, device=points_world.device)
    )
    valid = ((coord >= -0.5) & (coord <= shape - 0.5)).all(dim=-1)

    # align_corners=False maps voxel-centre index i to
    # 2*(i+0.5)/size-1. ``coord`` is already the centre-index coordinate.
    norm = 2.0 * (coord + 0.5) / shape - 1.0
    grid = norm.reshape(1, 1, norm.shape[0], norm.shape[1], 3)
    if grid.dtype != volume.dtype:
        grid = grid.to(dtype=volume.dtype)
    sampled = functional.grid_sample(
        volume,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    # (1,C,1,R,N) -> (R,N,C); accumulation is always FP32.
    sampled = sampled[0, :, 0].permute(1, 2, 0).to(dtype=torch.float32)
    return sampled * valid.unsqueeze(-1).to(dtype=torch.float32)


def _sample_noise(
    points_world,
    *,
    shape: Tuple[int, int, int],
    frequency: float,
    time_phase: float,
    seed: int,
    cache: TorchVolumeCache,
):
    """Periodic trilinear sampling of the same noise texture as NumPy."""
    torch, _ = _torch_modules()
    field = cache.get_noise(shape, seed, points_world.device)
    dims = torch.as_tensor(
        shape, dtype=torch.float32, device=points_world.device
    )
    coord = points_world * float(frequency)
    offset = torch.tensor(
        [0.0, float(time_phase), 0.0],
        dtype=torch.float32,
        device=points_world.device,
    )
    coord = torch.remainder(coord + offset, dims)
    i0 = torch.floor(coord).to(dtype=torch.long)
    frac = coord - i0.to(dtype=torch.float32)
    dims_long = torch.as_tensor(
        shape, dtype=torch.long, device=points_world.device
    )
    i0 = torch.remainder(i0, dims_long)
    i1 = torch.remainder(i0 + 1, dims_long)

    ix0, iy0, iz0 = i0.unbind(dim=-1)
    ix1, iy1, iz1 = i1.unbind(dim=-1)
    fx, fy, fz = frac.unbind(dim=-1)
    c000 = field[ix0, iy0, iz0]
    c100 = field[ix1, iy0, iz0]
    c010 = field[ix0, iy1, iz0]
    c110 = field[ix1, iy1, iz0]
    c001 = field[ix0, iy0, iz1]
    c101 = field[ix1, iy0, iz1]
    c011 = field[ix0, iy1, iz1]
    c111 = field[ix1, iy1, iz1]
    c00 = c000 * (1.0 - fx) + c100 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx
    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy
    return c0 * (1.0 - fz) + c1 * fz


def _fractal_noise(
    points_world,
    *,
    time_phase: float,
    seed: int,
    cache: TorchVolumeCache,
):
    torch, _ = _torch_modules()
    kwargs = {"shape": (16, 32, 16), "cache": cache}
    n1 = _sample_noise(
        points_world,
        frequency=4.0,
        time_phase=time_phase * 1.6,
        seed=seed,
        **kwargs,
    )
    n2 = _sample_noise(
        points_world,
        frequency=8.0,
        time_phase=time_phase * 2.4,
        seed=seed + 1,
        **kwargs,
    )
    n3 = _sample_noise(
        points_world,
        frequency=16.0,
        time_phase=time_phase * 3.6,
        seed=seed + 2,
        **kwargs,
    )
    raw = 0.55 * n1 + 0.30 * n2 + 0.15 * n3
    return torch.clamp((raw - 0.5) * 2.0, -1.0, 1.0)


def _flame_lut(intensity):
    """Torch equivalent of NumPy's per-channel linear interpolation."""
    torch, _ = _torch_modules()
    x = torch.clamp(intensity, 0.0, 1.0)
    xp = torch.as_tensor(
        _FLAME_LUT_X, dtype=torch.float32, device=x.device
    )
    colors = torch.as_tensor(
        _FLAME_LUT_RGB, dtype=torch.float32, device=x.device
    )
    segment = torch.bucketize(x.contiguous(), xp[1:-1])
    x0 = xp[segment]
    x1 = xp[segment + 1]
    weight = (x - x0) / torch.clamp(x1 - x0, min=1e-6)
    return colors[segment] * (1.0 - weight[..., None]) + (
        colors[segment + 1] * weight[..., None]
    )


def volumetric_composite_torch(
    *,
    rgb_clean: np.ndarray,
    depth_m: np.ndarray,
    cam_pos_world: np.ndarray,
    R_cam2world: np.ndarray,
    flame_field: np.ndarray,
    smoke_field: np.ndarray,
    temp_field: np.ndarray,
    origin: np.ndarray,
    voxel_m: float,
    grid_shape: Tuple[int, int, int],
    ambient_c: float,
    camera_K,
    params: VoxelRenderParams,
    t_sim: float = 0.0,
    device: str = "auto",
    volume_dtype: str = "float16",
    max_sample_points: int = 2_000_000,
    cache: Optional[TorchVolumeCache] = None,
    frame_key: Optional[Hashable] = None,
) -> Dict[str, np.ndarray]:
    """Torch implementation of the FireWorld front-to-back composite."""
    torch, _ = _torch_modules()
    resolved_device = resolve_torch_device(device)
    resolved_dtype = resolve_volume_dtype(resolved_device, volume_dtype)
    cache = cache or TorchVolumeCache()
    if frame_key is None:
        frame_key = (
            id(flame_field),
            id(smoke_field),
            id(temp_field),
        )

    try:
        import cv2
    except Exception:  # pragma: no cover
        cv2 = None  # type: ignore

    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    depth_m = np.clip(
        depth_m.astype(np.float32), 0.0, params.max_depth_m
    )
    h_full, w_full = depth_m.shape
    scale = float(np.clip(params.render_scale, 0.05, 1.0))
    if scale < 1.0 and cv2 is not None:
        width = max(64, int(round(w_full * scale)))
        height = max(64, int(round(h_full * scale)))
        depth_used = cv2.resize(
            depth_m, (width, height), interpolation=cv2.INTER_AREA
        )
        camera_eff = SimpleNamespace(
            cx=camera_K.cx * (width / float(w_full)),
            cy=camera_K.cy * (height / float(h_full)),
            fx=camera_K.fx * (width / float(w_full)),
            fy=camera_K.fy * (height / float(h_full)),
        )
    else:
        depth_used = depth_m
        camera_eff = camera_K
        height, width = h_full, w_full

    with torch.inference_mode():
        depth = torch.as_tensor(
            depth_used, dtype=torch.float32, device=resolved_device
        )
        v = torch.arange(
            height, dtype=torch.float32, device=resolved_device
        )
        u = torch.arange(
            width, dtype=torch.float32, device=resolved_device
        )
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        x_cam = (grid_u - float(camera_eff.cx)) * depth / float(camera_eff.fx)
        y_cam = -(grid_v - float(camera_eff.cy)) * depth / float(camera_eff.fy)
        points_cam = torch.stack([x_cam, y_cam, -depth], dim=-1)

        cam_pos = torch.as_tensor(
            cam_pos_world, dtype=torch.float32, device=resolved_device
        )
        rotation = torch.as_tensor(
            R_cam2world, dtype=torch.float32, device=resolved_device
        )
        end = points_cam @ rotation.T + cam_pos
        start = cam_pos.reshape(1, 1, 3).expand(height, width, 3)
        origin_tensor = torch.as_tensor(
            origin, dtype=torch.float32, device=resolved_device
        )
        max_corner = origin_tensor + torch.as_tensor(
            grid_shape, dtype=torch.float32, device=resolved_device
        ) * float(voxel_m)
        segment_min = torch.minimum(start, end)
        segment_max = torch.maximum(start, end)
        ray_hits = (
            (segment_max[..., 0] >= origin_tensor[0])
            & (segment_min[..., 0] <= max_corner[0])
            & (segment_max[..., 1] >= origin_tensor[1])
            & (segment_min[..., 1] <= max_corner[1])
            & (segment_max[..., 2] >= origin_tensor[2])
            & (segment_min[..., 2] <= max_corner[2])
        )

        transmittance = np.ones((height, width), dtype=np.float32)
        smoke_color_acc = np.zeros(
            (height, width, 3), dtype=np.float32
        )
        flame_color_acc = np.zeros(
            (height, width, 3), dtype=np.float32
        )
        flame_seen = np.zeros((height, width), dtype=np.float32)
        temp_apparent = np.full(
            (height, width), float(ambient_c), dtype=np.float32
        )

        hit_count = int(ray_hits.sum().item())
        if hit_count:
            volume = cache.get_volume(
                frame_key=frame_key,
                flame_field=flame_field,
                smoke_field=smoke_field,
                temp_field=temp_field,
                device=resolved_device,
                dtype=resolved_dtype,
            )
            start_hits = start[ray_hits]
            end_hits = end[ray_hits]
            n_steps = max(2, int(params.n_steps))
            ts = torch.linspace(
                0.0, 1.0, n_steps,
                dtype=torch.float32,
                device=resolved_device,
            )
            chord = torch.linalg.vector_norm(
                end_hits - start_hits, dim=-1
            )
            step_m = chord / float(n_steps - 1)
            max_points = max(n_steps, int(max_sample_points))
            rays_per_chunk = max(1, max_points // n_steps)

            trans_hits = np.empty(hit_count, dtype=np.float32)
            smoke_color_hits = np.empty(
                (hit_count, 3), dtype=np.float32
            )
            flame_color_hits = np.empty(
                (hit_count, 3), dtype=np.float32
            )
            flame_hits = np.empty(hit_count, dtype=np.float32)
            temp_hits = np.empty(hit_count, dtype=np.float32)

            noise_phase = float(
                np.fmod(
                    float(t_sim) * float(params.flame_time_speed),
                    10_000.0,
                )
            )
            noise_strength = float(
                np.clip(params.flame_noise_strength, 0.0, 1.5)
            )
            edge_break = float(np.clip(params.flame_edge_break, 0.0, 1.5))
            color_jitter = float(
                np.clip(params.flame_color_jitter, 0.0, 1.0)
            )
            smoke_noise = float(
                np.clip(params.smoke_noise_strength, 0.0, 1.0)
            )
            threshold_lo = float(params.flame_threshold)
            threshold_hi = threshold_lo + max(0.05, threshold_lo)
            threshold_inv = 1.0 / max(
                threshold_hi - threshold_lo, 1e-3
            )
            smoke_k = float(params.smoke_k_ext)
            flame_k = float(params.flame_k_ext)
            flame_smoke_k = smoke_k * (
                1.0 - float(
                    np.clip(params.flame_smoke_passthrough, 0.0, 1.0)
                )
            )
            smoke_color = torch.as_tensor(
                params.smoke_color_rgb,
                dtype=torch.float32,
                device=resolved_device,
            ) / 255.0

            surface_start = float(
                np.clip(params.thermal_surface_start, 0.0, 0.95)
            )
            surface_phase = torch.clamp(
                (ts - surface_start) / max(1.0 - surface_start, 1e-6),
                min=0.0,
            )
            surface_weights = surface_phase * surface_phase
            surface_weight_sum = torch.clamp(
                surface_weights.sum(), min=1e-6
            )

            for begin in range(0, hit_count, rays_per_chunk):
                finish = min(hit_count, begin + rays_per_chunk)
                chunk_start = start_hits[begin:finish]
                chunk_end = end_hits[begin:finish]
                chunk_step = step_m[begin:finish]
                points = (
                    chunk_start[:, None, :] * (1.0 - ts[None, :, None])
                    + chunk_end[:, None, :] * ts[None, :, None]
                )
                ray_intensity_pattern = None
                ray_color_pattern = None
                if noise_strength > 0.0:
                    ray_intensity_pattern = _fractal_noise(
                        chunk_end,
                        time_phase=noise_phase,
                        seed=17,
                        cache=cache,
                    )
                if color_jitter > 0.0:
                    ray_color_pattern = _fractal_noise(
                        chunk_end,
                        time_phase=noise_phase * 0.7,
                        seed=23,
                        cache=cache,
                    )
                sampled = _sample_world_volume(
                    volume,
                    points,
                    origin=origin_tensor,
                    voxel_m=voxel_m,
                    grid_shape=grid_shape,
                )
                smoke = sampled[..., 0]
                flame = sampled[..., 1]
                temperature = torch.where(
                    sampled[..., 2] > 0.0,
                    sampled[..., 2],
                    torch.tensor(
                        float(ambient_c),
                        dtype=torch.float32,
                        device=resolved_device,
                    ),
                )

                if smoke_noise > 0.0:
                    smoke_pattern = _fractal_noise(
                        points,
                        time_phase=noise_phase * 0.4,
                        seed=11,
                        cache=cache,
                    )
                    smoke = torch.clamp(
                        smoke * (1.0 + smoke_noise * smoke_pattern),
                        0.0,
                        1.5,
                    )
                if noise_strength > 0.0:
                    intensity_pattern = _fractal_noise(
                        points,
                        time_phase=noise_phase,
                        seed=1,
                        cache=cache,
                    )
                    if ray_intensity_pattern is not None:
                        intensity_pattern = torch.clamp(
                            0.35 * intensity_pattern
                            + 0.65 * ray_intensity_pattern[:, None],
                            -1.0,
                            1.0,
                        )
                    flame_norm = torch.clamp(flame, 0.0, 1.0)
                    edge_weight = 4.0 * flame_norm * (1.0 - flame_norm)
                    detail_weight = torch.clamp(
                        noise_strength
                        * (0.30 + 0.70 * edge_weight),
                        0.0,
                        0.95,
                    )
                    density_texture = torch.clamp(
                        0.55 + 0.95 * intensity_pattern,
                        0.05,
                        1.45,
                    )
                    edge_texture = torch.clamp(
                        0.65 + edge_break * intensity_pattern,
                        0.05,
                        1.40,
                    )
                    core_detail = (
                        (1.0 - detail_weight)
                        + detail_weight * density_texture
                    )
                    silhouette_detail = (
                        (1.0 - edge_weight)
                        + edge_weight * edge_texture
                    )
                    flame = torch.clamp(
                        flame * core_detail * silhouette_detail,
                        0.0,
                        1.2,
                    )

                flame_used = torch.clamp(
                    (flame - threshold_lo) * threshold_inv, 0.0, 1.0
                ) * flame
                if color_jitter > 0.0:
                    color_pattern = _fractal_noise(
                        points,
                        time_phase=noise_phase * 0.7,
                        seed=5,
                        cache=cache,
                    )
                    if ray_color_pattern is not None:
                        color_pattern = torch.clamp(
                            0.35 * color_pattern
                            + 0.65 * ray_color_pattern[:, None],
                            -1.0,
                            1.0,
                        )
                    lut_input = torch.clamp(
                        flame_used + color_jitter * color_pattern,
                        0.0,
                        1.0,
                    )
                else:
                    lut_input = flame_used

                displacement = float(
                    np.clip(
                        params.flame_smoke_displacement,
                        0.0,
                        1.0,
                    )
                )
                flame_presence = torch.clamp(
                    flame_used / 0.35, 0.0, 1.0
                )
                visible_smoke = smoke * (
                    1.0 - displacement * flame_presence
                )

                step = chunk_step[:, None]
                scene_step = torch.exp(
                    -(
                        smoke_k * visible_smoke
                        + flame_k * flame_used
                    )
                    * step
                )
                flame_step = torch.exp(
                    -(
                        flame_smoke_k * visible_smoke
                        + flame_k * flame_used
                    )
                    * step
                )
                ones = torch.ones(
                    (finish - begin, 1),
                    dtype=torch.float32,
                    device=resolved_device,
                )
                scene_before = torch.cat(
                    [ones, torch.cumprod(scene_step[:, :-1], dim=1)],
                    dim=1,
                )
                flame_before = torch.cat(
                    [ones, torch.cumprod(flame_step[:, :-1], dim=1)],
                    dim=1,
                )
                emission = (
                    _flame_lut(lut_input)
                    * (
                        flame_used
                        * float(params.flame_emission_gain)
                        * step
                    )[..., None]
                )
                scatter = (
                    smoke_color[None, None, :]
                    * (visible_smoke * smoke_k * step)[..., None]
                )
                integrated_flame_color = (
                    flame_before[..., None] * emission
                ).sum(dim=1)
                integrated_smoke_color = (
                    scene_before[..., None] * scatter
                ).sum(dim=1)

                excess = torch.clamp(
                    temperature - float(ambient_c), min=0.0
                )
                surface_excess = (
                    excess * surface_weights[None, :]
                ).sum(dim=1) / surface_weight_sum
                path_excess = excess.mean(dim=1)
                apparent = float(ambient_c) + torch.maximum(
                    surface_excess,
                    path_excess * float(params.thermal_air_coupling),
                )

                trans_hits[begin:finish] = (
                    scene_step.prod(dim=1).cpu().numpy()
                )
                smoke_color_hits[begin:finish] = (
                    integrated_smoke_color.cpu().numpy()
                )
                flame_color_hits[begin:finish] = (
                    integrated_flame_color.cpu().numpy()
                )
                flame_hits[begin:finish] = (
                    flame_used.amax(dim=1).cpu().numpy()
                )
                temp_hits[begin:finish] = apparent.cpu().numpy()

            ray_hits_cpu = ray_hits.cpu().numpy()
            transmittance[ray_hits_cpu] = trans_hits
            smoke_color_acc[ray_hits_cpu] = smoke_color_hits
            flame_color_acc[ray_hits_cpu] = flame_color_hits
            flame_seen[ray_hits_cpu] = flame_hits
            temp_apparent[ray_hits_cpu] = temp_hits

    return finalize_volumetric_outputs(
        rgb_clean=rgb_clean,
        transmittance=transmittance,
        smoke_color_acc=smoke_color_acc,
        flame_color_acc=flame_color_acc,
        flame_seen=flame_seen,
        temp_apparent=temp_apparent,
        output_hw=(h_full, w_full),
        ambient_c=ambient_c,
        params=params,
    )
