"""Smoke test: human thermal overlay + physical temperature field.

Verifies the two bugs are fixed:
  1. camera_K as an argparse.Namespace (get_camera_K output) no longer
     breaks project_humans_to_thermal -> humans actually get painted.
  2. compose_thermal returns a physical temperature map (ambient where
     there is no heat; ~ambient+25 where a person is; not driven by RGB
     luma).

Run:  python scripts/test_human_thermal.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.general_utils import get_camera_K  # noqa: E402
from utils.fire_sensors.voxel_render import compose_thermal  # noqa: E402
from utils.fire_sensors.humans_thermal import (  # noqa: E402
    HumanThermalTarget,
    project_humans_to_thermal,
    add_humans_to_thermal_image,
)


def _fake_agent_state(cam_pos, look_dir_yaw=0.0):
    """Build an AgentState-like object. Camera at cam_pos, looking down
    -Z in world (identity rotation) so a person on -Z is in front."""
    # identity rotation quaternion (w, x, y, z)
    rot = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    sensor = SimpleNamespace(position=np.asarray(cam_pos, np.float64),
                             rotation=rot)
    return SimpleNamespace(sensor_states={"depth": sensor},
                           position=np.asarray(cam_pos, np.float64),
                           rotation=rot)


def main() -> int:
    H, W = 480, 640
    ambient = 25.0
    K = get_camera_K(W, H, 79.0)          # <-- Namespace, the real repo path
    assert not hasattr(K, "__getitem__"), "get_camera_K should be a Namespace"

    # Camera at origin looking along -Z; person 3 m in front, slightly
    # right and at torso height.
    agent_state = _fake_agent_state([0.0, 0.88, 0.0])
    person = HumanThermalTarget(
        position=np.array([0.3, 0.9, -3.0]),  # world, -Z in front of cam
        excess_c=25.0,
    )

    # Depth scene: far wall at 5 m, with a person-shaped cluster at 3 m
    # (an upright rectangle with a head bump) so the depth-carve has a
    # real silhouette to extract rather than a flat wall.
    depth_m = np.full((H, W), 5.0, dtype=np.float32)
    # project the person centre to find where to place the depth cluster
    u_c, v_c = 342, 240   # roughly where [0.3,0.9,-3] lands for this K
    # torso rectangle
    depth_m[v_c - 70:v_c + 90, u_c - 22:u_c + 22] = 3.0
    # head
    depth_m[v_c - 95:v_c - 70, u_c - 12:u_c + 12] = 3.0
    # arms
    depth_m[v_c - 55:v_c - 10, u_c - 40:u_c - 22] = 3.0
    depth_m[v_c - 55:v_c - 10, u_c + 22:u_c + 40] = 3.0

    # --- 1) projection carves the person silhouette from depth ----------
    mask = project_humans_to_thermal(
        humans=[person], agent_state=agent_state, camera_K=K,
        image_hw=(H, W), depth_m=depth_m, max_depth_m=8.0,
    )
    painted = float((mask > 0).mean())
    hot_px = int((mask > 0).sum())
    print(f"[1] human silhouette painted fraction = {painted:.4f}, "
          f"px = {hot_px}, peak = {mask.max():.1f} C")
    assert painted > 0.0, "FAIL: human not projected (Namespace K bug)"
    assert mask.max() > 15.0, "FAIL: human excess too weak"

    # The carved silhouette must follow the depth cluster: (a) roughly
    # match the person cluster area, and (b) leave the arm-gap columns
    # (between torso and arms) cold, which a fat ellipse would fill.
    cluster_px = int((np.abs(depth_m - 3.0) < 0.1).sum())
    ratio = hot_px / max(cluster_px, 1)
    print(f"[1] silhouette/cluster area ratio = {ratio:.2f} "
          f"(expect ~0.6..1.4, ellipse would be >>1.4)")
    assert 0.4 < ratio < 1.8, \
        "FAIL: mask does not follow the depth silhouette (blob, not outline)"
    # A gap column just outside the torso but inside an ellipse bbox
    # should be cold (depth there is the 5 m wall, not the person).
    gap_col = mask[v_c - 5:v_c + 5, u_c + 45:u_c + 55]
    assert float(gap_col.max()) == 0.0, \
        "FAIL: mask bleeds past the body outline into background"

    # --- 2) physical temperature field ---------------------------------
    # No fire: temp_max = ambient everywhere, flame = 0. Bright vs dark
    # RGB must NOT change the returned temperature.
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:, : W // 2] = 230        # bright half
    rgb[:, W // 2:] = 20          # dark half
    temp_max = np.full((H, W), ambient, np.float32)
    flame = np.zeros((H, W), np.float32)
    img, temp = compose_thermal(rgb, temp_max, flame, ambient_c=ambient,
                                color_blend=1.0)
    tmin, tmax = float(temp.min()), float(temp.max())
    print(f"[2] no-fire temperature range = [{tmin:.2f}, {tmax:.2f}] C "
          f"(expect flat ~{ambient})")
    assert abs(tmin - ambient) < 1e-3 and abs(tmax - ambient) < 1e-3, \
        "FAIL: temperature field polluted by RGB luma"

    # --- 3) overlay humans onto the composed frame + temp ---------------
    img2, temp2 = add_humans_to_thermal_image(
        thermal_image_bgr=img, thermal_temperature=temp, humans=[person],
        agent_state=agent_state, camera_K=K, depth_m=depth_m, max_depth_m=8.0,
    )
    hot = float(temp2.max())
    print(f"[3] after human overlay, peak temperature = {hot:.1f} C "
          f"(expect >= {ambient + 15:.0f})")
    assert hot >= ambient + 15.0, "FAIL: human did not raise temperature map"
    # display image must have changed where the person is
    changed = int((img2 != img).any(axis=-1).sum())
    print(f"[3] display pixels changed by human = {changed}")
    assert changed > 200, "FAIL: human not visible in display image"

    # --- 4) with a real fire, person still recorded in temp field -------
    temp_max_fire = np.full((H, W), ambient, np.float32)
    temp_max_fire[200:260, 300:360] = 620.0     # a hot fire patch
    flame_fire = np.zeros((H, W), np.float32)
    flame_fire[200:260, 300:360] = 0.8
    img_f, temp_f = compose_thermal(rgb, temp_max_fire, flame_fire,
                                    ambient_c=ambient, color_blend=1.0)
    print(f"[4] fire temperature peak = {float(temp_f.max()):.0f} C "
          f"(expect ~600+)")
    assert temp_f.max() > 500.0, "FAIL: fire not hot in temperature map"

    print("\nOK: humans light up in IR and the temperature field is physical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
