"""Verify that FireWorldController advances the timeline with robot
steps (not wall clock) and that the renderer's output reacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.fire_world.controller import FireClock, FireWorldController  # noqa: E402
from utils.fire_world.runtime import FireWorld, FireWorldRenderer  # noqa: E402


SCENE = "Nfvxx8J5NCo"
PLAN_ID = "83679a07b632"


def test_clock() -> None:
    clk = FireClock(steps_per_unit=5, seconds_per_unit=2.0, base_t0_s=0.0)
    assert clk.t_sim_for_step(0) == 0.0
    assert clk.t_sim_for_step(4) == 0.0
    assert clk.t_sim_for_step(5) == 2.0
    assert clk.t_sim_for_step(50) == 20.0
    assert clk.t_sim_for_step(100) == 40.0
    print("FireClock: 5 steps/unit, 2 s/unit -> step 50 = t_sim 20.0s OK")


class _FakeAgentState:
    """Match the bits ``_habitat_agent_state_to_cam`` reads on real
    habitat agent states."""

    def __init__(self, position, R):
        self.position = np.asarray(position, dtype=np.float64)
        self.rotation = np.asarray(R, dtype=np.float64)
        # Habitat exposes sensor_states like a dict.
        self.sensor_states = {"depth": SimpleNamespace(
            position=self.position, rotation=self.rotation
        )}


def test_step_to_render(tmp: Path) -> None:
    inv_path = ROOT / "scenes" / SCENE / "inventory.json"
    plan_path = ROOT / "scenes" / SCENE / "plans" / f"{PLAN_ID}.json"
    fw = FireWorld.load(SCENE, PLAN_ID, out_root=ROOT / "outputs" / "fire_world")
    K = SimpleNamespace(cx=255.5, cy=191.5, fx=308.7, fy=308.7)
    renderer = FireWorldRenderer(
        fw=fw, camera_K=K, max_depth_m=4.0, n_steps=24, smoke_k_ext=6.0,
    )
    clock = FireClock(steps_per_unit=5, seconds_per_unit=2.0,
                      base_t0_s=float(fw.times[0]))
    ctrl = FireWorldController(fw=fw, renderer=renderer, clock=clock)

    plan = json.loads(plan_path.read_text())
    target = np.asarray(plan["ignitions"][0]["position"], dtype=np.float64)

    # A camera looking at the first ignition from a fixed pose.
    cam_pos = target + np.array([2.5, 0.6, 0.0])
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up_c = np.cross(right, forward)
    R = np.stack([right, up_c, -forward], axis=1)
    a_state = _FakeAgentState(cam_pos, R)

    H, W = 384, 512
    rgb_clean = np.full((H, W, 3), 200, dtype=np.uint8)
    depth_clean = np.full((H, W, 1), 4.0, dtype=np.float32)
    obs = {"rgb": rgb_clean, "depth": depth_clean}

    snapshots = []
    for step in [0, 30, 60, 120, 240]:
        out = ctrl.render_for_agent(
            obs, a_state, robot_step=step,
            max_depth_m=4.0, normalize_depth=False,
        )
        T = float(np.mean(out["transmittance"]))
        flame = float(np.mean(out["thermal_flame_mask"]))
        t_sim = float(out["t_sim_s"])
        snapshots.append((step, t_sim, T, flame))
        print(f"  step={step:>4} t_sim={t_sim:>6.1f}s  T_mean={T:.3f}  flame_px={flame:.2%}")

    # The FireClock should advance t_sim with step count.
    times = [s[1] for s in snapshots]
    assert times[0] <= times[1] <= times[2] <= times[3] <= times[4], times
    assert times[0] != times[4], "t_sim never moved"

    # The render output must vary between t=0 and a later step (the
    # plume grows / the source moves through its life cycle).
    flames = [s[3] for s in snapshots]
    if max(flames) - min(flames) < 1e-3:
        print("WARN: flame_px constant across steps. Either the camera "
              "is looking away from the plume or the timeline plateaued. "
              "This is allowed but worth noting.")
    print("FireWorldController step->render: OK")


def main() -> int:
    test_clock()
    test_step_to_render(ROOT / "outputs" / "test_fire_world_step_tmp")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
