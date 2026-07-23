"""Project humanoid pedestrians into the current camera and paint them
as warm silhouettes in the thermal image.

The thermal composer :func:`utils.fire_sensors.voxel_render.compose_thermal`
keeps physical apparent temperature separate from its dark structural
display. This module adds a per-pixel ``human_excess_c`` field to that
physical temperature and paints a high-contrast body signature so people
remain identifiable even beside a much hotter fire.

The projection uses the standard pinhole model and the current
camera pose (world->camera) derived from the agent's depth-sensor
state. Occlusion is optional and controlled by the caller (if a
metric depth image is passed, we clip out humans hidden behind
geometry).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import quaternion as _npq  # noqa: F401
    _HAS_QUAT = True
except Exception:
    _HAS_QUAT = False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
@dataclass
class HumanThermalTarget:
    """A single humanoid to render as a warm blob."""

    position: np.ndarray            # (3,) world-space center of mass (metres)
    height_m: float = 1.75          # skeleton height (radial extent below/above center)
    radius_m: float = 0.28          # radial half-width of the person in world-metres
    excess_c: float = 25.0          # skin temperature above ambient (K); exaggerated
                                    # a bit vs the physical 9 K so a "warm
                                    # object" detection threshold in the
                                    # 40 K range still fires reliably.


# ---------------------------------------------------------------------------
# Camera pose helpers
# ---------------------------------------------------------------------------
def _unpack_intrinsics(camera_K) -> Tuple[float, float, float, float]:
    """Return ``(fx, fy, cx, cy)`` from either a 3x3 intrinsics matrix or
    an object/Namespace exposing ``.fx/.fy/.cx/.cy``.

    ``utils.general_utils.get_camera_K`` hands out an ``argparse.Namespace``
    (attribute-style), while other callers may pass a plain 3x3 numpy
    matrix (subscript-style). Supporting both keeps the human-thermal
    overlay from silently failing on a ``TypeError`` (which used to be
    swallowed upstream, so humans never appeared in the IR image).
    """
    # Attribute-style (Namespace / SimpleNamespace / dataclass).
    if all(hasattr(camera_K, a) for a in ("fx", "fy", "cx", "cy")):
        return (
            float(camera_K.fx),
            float(camera_K.fy),
            float(camera_K.cx),
            float(camera_K.cy),
        )
    # Matrix-style (numpy array / nested list).
    arr = np.asarray(camera_K, dtype=np.float64)
    if arr.shape == (3, 3):
        return (
            float(arr[0, 0]),
            float(arr[1, 1]),
            float(arr[0, 2]),
            float(arr[1, 2]),
        )
    raise TypeError(
        "camera_K must be a 3x3 matrix or expose .fx/.fy/.cx/.cy; "
        f"got {type(camera_K)!r} with shape {getattr(arr, 'shape', None)}"
    )


def _rotation_matrix_from_state(sensor_state) -> np.ndarray:
    """Return R_cam2world as a (3,3) numpy matrix for either numpy.quaternion
    or a magnum quaternion / list."""
    rot = sensor_state.rotation
    if hasattr(rot, "x") and hasattr(rot, "w") and _HAS_QUAT:
        import quaternion as npq
        return npq.as_rotation_matrix(rot)
    arr = np.asarray(rot, dtype=np.float64)
    if arr.shape == (4,):
        w, x, y, z = arr
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
    return np.asarray(arr, dtype=np.float64)


def project_humans_to_thermal(
    humans: Iterable[HumanThermalTarget],
    agent_state,
    camera_K: np.ndarray,
    image_hw: Tuple[int, int],
    depth_m: Optional[np.ndarray] = None,
    *,
    max_depth_m: float = 8.0,
) -> np.ndarray:
    """Return a ``(H, W)`` float32 ``human_excess_c`` mask.

    Args:
        humans: list of :class:`HumanThermalTarget`.
        agent_state: object with ``sensor_states["depth"].position``
            and ``.rotation`` (Habitat 3 AgentState).
        camera_K: camera intrinsics, either a 3x3 matrix in pixel space
            or an object exposing ``.fx/.fy/.cx/.cy`` (e.g. the Namespace
            returned by :func:`utils.general_utils.get_camera_K`).
        image_hw: ``(H, W)`` of the target thermal image.
        depth_m: optional metric depth image (H, W); if provided the
            humans that fall behind a closer scene surface at their
            silhouette center are dropped from the pixel painting.
        max_depth_m: humans farther than this from the camera are
            skipped entirely (default 8 m).

    The mask is expressed in **Celsius above ambient**, so the caller
    can simply do ``temperature_c = temperature_c + mask`` before the
    auto-stretch.
    """
    H, W = int(image_hw[0]), int(image_hw[1])
    mask = np.zeros((H, W), dtype=np.float32)
    if not humans:
        return mask

    # Camera pose: world <- depth sensor. Habitat's sensor states are
    # in world coords already.
    sensor_state = agent_state.sensor_states.get("depth", agent_state)
    cam_pos = np.asarray(sensor_state.position, dtype=np.float64)
    R_cam2world = _rotation_matrix_from_state(sensor_state)

    # In Habitat the depth camera looks along -Z_cam. We want to
    # express world points in the camera frame such that a point in
    # front of the camera has positive depth.
    R_world2cam = R_cam2world.T
    fx, fy, cx, cy = _unpack_intrinsics(camera_K)

    # Prepare a depth map (metric) for optional occlusion tests.
    depth_2d = None
    if depth_m is not None:
        depth_2d = np.asarray(depth_m)
        if depth_2d.ndim == 3:
            depth_2d = depth_2d[..., 0]

    for target in humans:
        world_center = np.asarray(target.position, dtype=np.float64)
        cam_center = R_world2cam @ (world_center - cam_pos)
        # Habitat convention: +Z_cam points backwards, so depth from
        # camera is -z_cam.
        depth_from_cam = float(-cam_center[2])
        if not np.isfinite(depth_from_cam) or depth_from_cam <= 0.05:
            continue
        if depth_from_cam > max_depth_m:
            continue

        u = fx * (cam_center[0] / depth_from_cam) + cx
        v = fy * (-cam_center[1] / depth_from_cam) + cy   # invert Y

        if not (np.isfinite(u) and np.isfinite(v)):
            continue

        # Convert world-space radius / height into pixels using the
        # focal length divided by depth. Enforce a healthy minimum
        # (~8 px radius) so a distant person is still detectable.
        radius_px = max(8.0, float(fx * target.radius_m / depth_from_cam))
        height_px = max(radius_px * 3.0,
                        float(fy * target.height_m / depth_from_cam))

        cx_px, cy_px = int(round(u)), int(round(v))

        # ---- Preferred path: carve the TRUE silhouette from depth ------
        # First reject a target hidden behind a closer surface. Previously
        # the carve ran before this test, so a wall near the expected person
        # depth could itself become a large thermal "person" mask.
        if depth_2d is not None:
            if 0 <= cy_px < H and 0 <= cx_px < W:
                scene_depth = float(depth_2d[cy_px, cx_px])
                if (
                    scene_depth > 0
                    and scene_depth < 1.0 + 1e-3
                    and max_depth_m > 1.5
                ):
                    scene_depth *= max_depth_m
                if scene_depth > 0 and scene_depth + 0.3 < depth_from_cam:
                    continue

            carved = _carve_person_from_depth(
                mask,
                depth_2d=depth_2d,
                center=(cx_px, cy_px),
                radius_px=radius_px,
                height_px=height_px,
                depth_from_cam=depth_from_cam,
                depth_band_m=max(0.35, float(target.radius_m) * 2.5),
                excess_c=float(target.excess_c),
                max_depth_m=float(max_depth_m),
            )
            # Depth is authoritative. If no plausible humanoid silhouette is
            # present, the model is off-screen/occluded/not rendered; never
            # fall back to a goal-position ellipse over the RGB scene.
            if not carved:
                continue

        # Elliptical fallback is only for callers with no depth sensor at all.
        else:
            _paint_person_blob(
                mask,
                center=(cx_px, cy_px),
                radius_px=radius_px,
                height_px=height_px,
                excess_c=float(target.excess_c),
            )
    return mask


def _carve_person_from_depth(
    mask: np.ndarray,
    depth_2d: np.ndarray,
    center: Tuple[int, int],
    radius_px: float,
    height_px: float,
    depth_from_cam: float,
    depth_band_m: float,
    excess_c: float,
    max_depth_m: float,
) -> bool:
    """Extract the real human silhouette from the depth image.

    Within a generous bounding box around the projected person centre,
    keep the pixels whose metric depth lies within ``depth_band_m`` of
    the person's distance. That connected depth cluster is the body
    outline. Returns True when a plausible silhouette was written.
    """
    if cv2 is None:
        return False
    H, W = mask.shape
    cx, cy = int(center[0]), int(center[1])

    # Search window: a bit wider than the expected person so arms and a
    # side-step stance are captured, and tall enough for the full body.
    half_w = int(round(radius_px * 2.2))
    half_h = int(round(height_px * 0.6))
    x0 = max(0, cx - half_w)
    x1 = min(W, cx + half_w + 1)
    y0 = max(0, cy - half_h)
    y1 = min(H, cy + half_h + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False

    win = depth_2d[y0:y1, x0:x1].astype(np.float32)
    # Normalise defensively: Suite hands metres, but tolerate a [0,1]
    # depth by scaling with max_depth_m.
    if np.nanmax(win) <= 1.0 + 1e-3 and max_depth_m > 1.5:
        win = win * max_depth_m

    lo = depth_from_cam - depth_band_m
    hi = depth_from_cam + depth_band_m
    band = (win >= lo) & (win <= hi) & np.isfinite(win) & (win > 0.0)
    if band.sum() < max(20, int(0.02 * band.size)):
        # Too few in-band pixels: person likely occluded or mis-projected.
        return False

    band_u8 = band.astype(np.uint8)
    # Clean up speckle and close small gaps so the silhouette is solid.
    k = np.ones((3, 3), np.uint8)
    band_u8 = cv2.morphologyEx(band_u8, cv2.MORPH_OPEN, k, iterations=1)
    band_u8 = cv2.morphologyEx(band_u8, cv2.MORPH_CLOSE, k, iterations=2)

    # Keep only the connected component nearest the projected centre so
    # a wall at similar depth in a corner of the window doesn't leak in.
    n_lbl, lbl, stats, cents = cv2.connectedComponentsWithStats(band_u8, 8)
    if n_lbl <= 1:
        return False
    local_cx, local_cy = cx - x0, cy - y0
    best_lbl, best_d = -1, 1e18
    for li in range(1, n_lbl):
        area = stats[li, cv2.CC_STAT_AREA]
        if area < max(20, int(0.02 * band.size)):
            continue
        ccx, ccy = cents[li]
        d = (ccx - local_cx) ** 2 + (ccy - local_cy) ** 2
        if d < best_d:
            best_d, best_lbl = d, li
    if best_lbl < 0:
        return False

    sil = (lbl == best_lbl)
    area = int(stats[best_lbl, cv2.CC_STAT_AREA])
    component_w = int(stats[best_lbl, cv2.CC_STAT_WIDTH])
    component_h = int(stats[best_lbl, cv2.CC_STAT_HEIGHT])
    # Reject broad planar surfaces. A standing stop-pose humanoid is tall and
    # occupies only part of the projection window; the failure captured in the
    # benchmark filled almost the whole window with one wall component.
    if (
        area < 20
        or area > int(0.55 * band.size)
        or component_w > max(12, int(round(radius_px * 3.4)))
        or component_h < max(8, int(round(component_w * 1.05)))
    ):
        return False

    # Soft interior→edge falloff so the core reads hottest (skin over
    # the torso/face) and the outline tapers, like a real IR body.
    dist = cv2.distanceTransform(sil.astype(np.uint8), cv2.DIST_L2, 3)
    dmax = float(dist.max())
    if dmax > 1e-6:
        weight = 0.55 + 0.45 * (dist / dmax)   # 0.55 at edge, 1.0 at core
    else:
        weight = sil.astype(np.float32)
    contribution = (weight * float(excess_c)).astype(np.float32)
    contribution[~sil] = 0.0

    sub = mask[y0:y1, x0:x1]
    np.maximum(sub, contribution, out=sub)
    mask[y0:y1, x0:x1] = sub
    return True


def _paint_person_blob(
    mask: np.ndarray,
    center: Tuple[int, int],
    radius_px: float,
    height_px: float,
    excess_c: float,
) -> None:
    """Paint an upright elliptical human silhouette with a Gaussian falloff."""
    H, W = mask.shape
    cx, cy = int(center[0]), int(center[1])
    rx = max(1, int(round(radius_px)))
    ry = max(rx * 2, int(round(height_px / 2.0)))

    # Bounding box (clipped to image).
    x0 = max(0, cx - rx * 2)
    x1 = min(W, cx + rx * 2 + 1)
    y0 = max(0, cy - ry - int(rx * 0.5))
    y1 = min(H, cy + ry + int(rx * 0.5) + 1)
    if x1 <= x0 or y1 <= y0:
        return

    ys = np.arange(y0, y1)[:, None]
    xs = np.arange(x0, x1)[None, :]
    # Ellipse distance in normalized units.
    dx = (xs - cx) / max(rx, 1)
    dy = (ys - cy) / max(ry, 1)
    d2 = dx * dx + dy * dy
    # Gentle Gaussian falloff so the whole silhouette is visible;
    # exp(-0.7 d^2) is still ~0.5 at the ellipse boundary.
    falloff = np.exp(-0.7 * d2)
    contribution = (falloff * excess_c).astype(np.float32)

    # Take the max so overlapping humans don't sum artificially.
    mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], contribution)


# ---------------------------------------------------------------------------
# Helpers for callers
# ---------------------------------------------------------------------------
def humans_from_walker(walker, *, excess_c: float = 25.0) -> List[HumanThermalTarget]:
    """Extract :class:`HumanThermalTarget` records from a running
    :class:`envs.random_humanoid.RandomHumanoidWalker`.

    The walker exposes ``.humans`` (a list of live
    ``KinematicHumanoid`` objects). Each humanoid's ``base_pos`` is
    the world-space foot pin; we lift the projection point to the
    torso by adding half the humanoid's body height so the blob sits
    where a real IR image would see the warmest surface (upper body).
    """
    targets: List[HumanThermalTarget] = []
    if walker is None:
        return targets
    for humanoid in getattr(walker, "humans", []):
        try:
            if humanoid.sim_obj is None or not humanoid.sim_obj.is_alive:
                continue
        except Exception:
            continue
        base = np.array(humanoid.base_pos, dtype=np.float64)
        # Lift to torso height so the projection lands on the chest,
        # roughly where FLIR sees the highest apparent temperature.
        base[1] += 0.9
        targets.append(HumanThermalTarget(
            position=base,
            height_m=1.75,
            radius_m=0.28,
            excess_c=excess_c,
        ))
    return targets


def add_humans_to_thermal_image(
    thermal_image_bgr: np.ndarray,
    thermal_temperature: np.ndarray,
    humans: Sequence[HumanThermalTarget],
    agent_state,
    camera_K: np.ndarray,
    *,
    depth_m: Optional[np.ndarray] = None,
    max_depth_m: float = 8.0,
    color_blend: float = 1.0,
    human_core_bgr: Tuple[int, int, int] = (245, 250, 255),   # near-white
    human_mid_bgr: Tuple[int, int, int]  = (60, 210, 255),    # warm yellow
    human_halo_bgr: Tuple[int, int, int] = (30, 90, 220),     # warm red
    glow_gain: float = 0.35,
    glow_ksize: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Overlay warm human silhouettes on an already-composed thermal frame.

    The person mask comes from :func:`project_humans_to_thermal`, which
    carves the true body outline out of the depth image (head / torso /
    limbs), so the overlay is a real silhouette rather than a blob.

    Flame temperature spans hundreds of degrees, so a physical 9 °C human
    excess can still be visually subtle beside a fire. We therefore blit
    humans directly on top of ``thermal_image_bgr`` with a **hot-body
    three-band ramp** that matches a FLIR high-gain-on-warm-objects mode: a
    near-white core (skin over torso/face), a warm-yellow body, and a
    thin red edge from the camera's PSF around a hot target.

    We still update ``thermal_temperature`` in place so downstream
    detectors (:mod:`utils.smoke_perception.thermal_mask_to_detections`)
    can trigger on the higher absolute temperature.
    """
    if not humans:
        return thermal_image_bgr, thermal_temperature

    H, W = thermal_temperature.shape[:2]
    mask = project_humans_to_thermal(
        humans=humans,
        agent_state=agent_state,
        camera_K=camera_K,
        image_hw=(H, W),
        depth_m=depth_m,
        max_depth_m=max_depth_m,
    )
    if not np.any(mask > 0):
        return thermal_image_bgr, thermal_temperature

    # Additive temperature contribution for downstream detection.
    temp_new = thermal_temperature + mask.astype(np.float32)

    # Normalise mask so alpha is in [0, 1] and the peak covers the
    # torso, not just one voxel of skin.
    peak = float(mask.max())
    a_norm = mask / max(peak, 1e-6)

    # ---- Three-band hot-body ramp -------------------------------------
    # a_norm is a Gaussian: 1.0 at the center, ~0.5 at the ellipse
    # boundary, ~0.15 further out. Map that to a core / mid / halo
    # colour ramp with soft transitions to avoid banding.
    core = np.clip((a_norm - 0.55) / 0.25, 0.0, 1.0)      # 0 -> 1 as we approach center
    mid  = np.clip((a_norm - 0.15) / 0.30, 0.0, 1.0) - core
    mid  = np.clip(mid, 0.0, 1.0)
    halo = np.clip((a_norm - 0.02) / 0.20, 0.0, 1.0) - (core + mid)
    halo = np.clip(halo, 0.0, 1.0)

    core_c = np.asarray(human_core_bgr, dtype=np.float32).reshape(1, 1, 3)
    mid_c  = np.asarray(human_mid_bgr,  dtype=np.float32).reshape(1, 1, 3)
    halo_c = np.asarray(human_halo_bgr, dtype=np.float32).reshape(1, 1, 3)

    person_rgb = (
        core[..., None] * core_c
        + mid[..., None] * mid_c
        + halo[..., None] * halo_c
    )
    person_alpha = np.clip(core + mid + halo, 0.0, 1.0)
    person_alpha = np.power(person_alpha, 0.5)  # brighten mid-tones
    alpha_3 = np.repeat(person_alpha[..., None], 3, axis=-1)

    image_f = thermal_image_bgr.astype(np.float32)
    image_f = image_f * (1.0 - alpha_3) + person_rgb * alpha_3

    # ---- Optional Gaussian bloom so the person is unmistakable --------
    # Blur the alpha mask with a fat kernel, take the max with the
    # existing image, so a warm halo leaks around the silhouette even
    # in dark rooms.
    if glow_gain > 0.0 and cv2 is not None:
        k = max(3, int(glow_ksize) | 1)   # ensure odd
        halo_a = cv2.GaussianBlur(person_alpha.astype(np.float32),
                                  (k, k), 0)
        halo_a = np.clip(halo_a * float(glow_gain), 0.0, 1.0)
        halo_rgb = halo_a[..., None] * halo_c
        image_f = np.maximum(image_f, halo_rgb)

    image_bgr = np.clip(image_f, 0, 255).astype(np.uint8)
    return image_bgr, temp_new.astype(np.float32)
