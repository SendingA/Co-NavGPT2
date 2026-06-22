"""Lightweight reaction-diffusion + buoyancy + ceiling-jet fire propagation.

Stage-3 of the fire_world pipeline. Loads ``inventory.json`` + ``plan.json``,
discretises the world into a 3D voxel grid, integrates the fields over time,
and writes a deterministic ``timeline.npz`` to ``outputs/fire_world/...``.

Physics is intentionally *simulation-grade*, not research-grade: the goal
is a plausible, smooth, reproducible 3D field that downstream stages
(top-down validation video and the runtime renderer) can consume.

Per voxel state:
    fuel            in [0, 1] - dimensionless fuel mass fraction
    temp            in deg C
    flame           in [0, 1] - dimensionless flame intensity
    smoke           in [0, 1] - dimensionless soot density

Update per dt step (default 0.5 s):
    1) Heat diffusion       T += alpha * Laplacian(T) * dt
    2) Vertical buoyancy    T, smoke shifted up by v_buoy * dt
    3) Ceiling jet          horizontal spread of T, smoke under the
                            scene ceiling
    4) Reaction             where T > T_ignite and fuel > thresh:
                              flame += k * fuel * dt
                              fuel  -= k * fuel * dt
                              T     += Q * fuel * dt
                              smoke += yield * fuel * dt
    5) Decay                flame relaxes to 0 once fuel runs out, smoke
                            slowly settles
    6) Cooling              T -> ambient with a tiny rate constant

All stages use plain numpy arrays; on a 100x40x70 grid (10x4x7 m at
voxel=0.10 m) one step is ~6 ms which is plenty for 600s @ 0.5s = 1200
steps in a few seconds.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from .voxel_world import VoxelWorld


# ---------------------------------------------------------------------------
# Numeric helpers (small, hot, kept inlined-friendly)
# ---------------------------------------------------------------------------
def _laplacian_3d(field: np.ndarray) -> np.ndarray:
    """6-point Laplacian with zero-flux (Neumann) boundary conditions."""
    f = field
    out = np.zeros_like(f)
    # +-x
    out[1:-1, :, :] += f[2:, :, :] - 2 * f[1:-1, :, :] + f[:-2, :, :]
    out[0, :, :] += f[1, :, :] - f[0, :, :]
    out[-1, :, :] += f[-2, :, :] - f[-1, :, :]
    # +-y
    out[:, 1:-1, :] += f[:, 2:, :] - 2 * f[:, 1:-1, :] + f[:, :-2, :]
    out[:, 0, :] += f[:, 1, :] - f[:, 0, :]
    out[:, -1, :] += f[:, -2, :] - f[:, -1, :]
    # +-z
    out[:, :, 1:-1] += f[:, :, 2:] - 2 * f[:, :, 1:-1] + f[:, :, :-2]
    out[:, :, 0] += f[:, :, 1] - f[:, :, 0]
    out[:, :, -1] += f[:, :, -2] - f[:, :, -1]
    return out


def _shift_up_y(field: np.ndarray, frac: float) -> np.ndarray:
    """Linear shift of a (Nx, Ny, Nz) field upward by ``frac`` cell.

    Mass-conserving with reflective boundaries:
        out[y]   = (1 - f) * in[y]  + f * in[y - 1]   for y >= 1
        out[0]  += f * in[0]                          (no underflow)
        out[Ny-1] += f * in[Ny-1]                     (no escape at ceiling)
    """
    if frac <= 0.0:
        return field
    f = float(np.clip(frac, 0.0, 1.0))
    out = (1.0 - f) * field
    out[:, 1:, :] += f * field[:, :-1, :]
    out[:, 0, :] += f * field[:, 0, :]
    out[:, -1, :] += f * field[:, -1, :]
    return out


def _ceiling_jet(field: np.ndarray, ceiling_y: int, frac: float,
                 thickness: int = 4) -> np.ndarray:
    """Horizontal spread of ``field`` under the ceiling layer.

    Real ceiling jets advect along the ceiling at ~0.3 m/s; we approximate
    that with an XZ box-blur applied to the top ``thickness`` Y layers
    (the buoyancy step pushes the plume into this band). ``frac`` is the
    weight of the blurred layer in [0, 1]; the rest stays put.
    """
    if frac <= 0.0:
        return field
    Nx, Ny, Nz = field.shape
    cy = max(1, min(int(ceiling_y), Ny - 1))
    y0 = max(0, cy - thickness + 1)
    layers = field[:, y0:cy + 1, :]
    # 3x3 box blur in XZ, broadcast over the kept Y layers.
    pad = np.pad(layers, ((1, 1), (0, 0), (1, 1)), mode="edge")
    blur = (
        pad[:-2, :, :-2] + pad[:-2, :, 1:-1] + pad[:-2, :, 2:]
        + pad[1:-1, :, :-2] + pad[1:-1, :, 1:-1] + pad[1:-1, :, 2:]
        + pad[2:, :, :-2] + pad[2:, :, 1:-1] + pad[2:, :, 2:]
    ) / 9.0
    field[:, y0:cy + 1, :] = (1.0 - frac) * layers + frac * blur
    return field


# ---------------------------------------------------------------------------
# Propagation engine
# ---------------------------------------------------------------------------
class FirePropagation:
    """Owns the voxel world, integrates one fire scenario forward."""

    def __init__(
        self,
        world: VoxelWorld,
        rules: Dict,
        seed: int = 0,
    ) -> None:
        self.world = world
        self.rules = rules
        self.rng = np.random.default_rng(seed)

        # Coefficients (read once for speed).
        self.alpha = float(rules.get("thermal_diffusivity", 0.05))
        self.t_ignite = float(rules.get("ignition_temp_c", 350.0))
        self.flammable_threshold = float(rules.get("flammable_threshold", 0.4))
        self.spread = float(rules.get("spread_speed_m_per_s", 0.04))
        self.buoy = float(rules.get("buoyancy_v_m_per_s", 0.5))
        self.cj = float(rules.get("ceiling_jet_speed_m_per_s", 0.30))
        self.ambient = float(rules.get("ambient_temp_c", 25.0))
        # Reaction rate (kept low + linear-with-fuel for stability and so
        # the source survives several minutes, matching real fire growth).
        self.k_burn = float(rules.get("k_burn_per_s", 1.0 / 240.0))
        # Heat released per (fuel-fraction * dt) at the source voxel.
        self.q_release_c = float(rules.get("q_release_c", 350.0))

        Nx, Ny, Nz = world.shape
        # Locate the ceiling Y as the world's top layer. If we ever add a
        # building-aware ceiling map, this becomes per-(x,z).
        self.ceiling_y = int(Ny - 1)

        # Active ignitions whose source we keep warm each step until they
        # burn out. Each entry: (slice_tuple, falloff, source_temp_c, t_end).
        self._sources: List[Tuple[Tuple[slice, slice, slice], np.ndarray, float, float]] = []

    # ------------------------------------------------------------------
    def add_source(
        self,
        sl: Tuple[slice, slice, slice],
        falloff: np.ndarray,
        source_temp_c: float,
        sustain_s: float,
        t_now: float,
    ) -> None:
        """Register a sustained heat source. Each step until t_now+sustain_s
        the source voxels are pinned to at least
        ``ambient + (T_src - ambient) * falloff``.
        """
        self._sources.append((sl, falloff, float(source_temp_c), t_now + float(sustain_s)))

    # ------------------------------------------------------------------
    def step(self, dt: float, t_now: float = 0.0) -> None:
        """One forward step. ``dt`` in seconds, ``t_now`` end-of-step time."""
        w = self.world
        v = w.voxel

        # Solid (impermeable to heat/smoke). Wall + ceiling block the
        # plume from leaking through; floor blocks downward leakage.
        solid: Optional[np.ndarray] = None
        for m in (w.walls, w.ceilings, w.floors):
            if m is not None:
                solid = m if solid is None else (solid | m)

        # 1) Heat diffusion (sub-stepped to respect CFL: alpha*dt/v^2 <= 0.16).
        max_diff_dt = 0.16 * v * v / max(self.alpha, 1e-6)
        n_sub = max(1, int(np.ceil(dt / max_diff_dt)))
        sub_dt = dt / n_sub
        coeff = self.alpha * sub_dt / max(v * v, 1e-6)
        for _ in range(n_sub):
            lap = _laplacian_3d(w.temp)
            w.temp = w.temp + coeff * lap
            if solid is not None:
                w.temp[solid] = self.ambient
        np.clip(w.temp, -50.0, 1500.0, out=w.temp)

        # 2) Vertical buoyancy: hot temp + smoke drift upward.
        frac_b = float(np.clip(self.buoy * dt / v, 0.0, 0.95))
        hot_mask = w.temp > self.ambient + 5.0
        if frac_b > 0.0 and hot_mask.any():
            t_excess = (w.temp - self.ambient) * hot_mask
            shifted = (1.0 - frac_b) * t_excess
            shifted[:, 1:, :] += frac_b * t_excess[:, :-1, :]
            shifted[:, 0, :] += frac_b * t_excess[:, 0, :]
            w.temp = self.ambient + shifted
            if solid is not None:
                w.temp[solid] = self.ambient

            smoke_hot = w.smoke * hot_mask.astype(np.float32)
            smoke_cold = w.smoke - smoke_hot
            shifted_s = _shift_up_y(smoke_hot, frac_b)
            w.smoke = np.clip(smoke_cold + shifted_s, 0.0, 1.0)
            if solid is not None:
                w.smoke[solid] = 0.0

        # 3) Ceiling jet (under the actual ceiling surface if available).
        frac_cj = float(np.clip(self.cj * dt / v, 0.0, 0.95))
        if frac_cj > 0.0:
            for _ in range(2):
                _ceiling_jet(w.temp, self.ceiling_y, frac_cj)
                _ceiling_jet(w.smoke, self.ceiling_y, frac_cj)
            if solid is not None:
                w.temp[solid] = self.ambient
                w.smoke[solid] = 0.0

        # 4) Reaction.
        ignitable = (w.fuel > self.flammable_threshold) & (w.temp > self.t_ignite)
        if ignitable.any():
            burn_rate = self.k_burn * dt * w.fuel * ignitable
            burn_rate = np.minimum(burn_rate, 0.25)
            w.flame = np.clip(w.flame + burn_rate, 0.0, 1.0)
            w.fuel = np.clip(w.fuel - burn_rate, 0.0, 1.0)
            heat = np.minimum(self.q_release_c * burn_rate, 200.0)
            w.temp = np.clip(w.temp + heat, -50.0, 1500.0)
            w.smoke = np.clip(w.smoke + 0.5 * burn_rate, 0.0, 1.0)

        # Surface flame spread along the fuel field.
        if self.spread > 0.0:
            kernel_frac = float(np.clip(self.spread * dt / v, 0.0, 0.30))
            if kernel_frac > 0.0:
                lap = _laplacian_3d(w.flame)
                w.flame = np.clip(
                    w.flame + kernel_frac * lap * (w.fuel > 0).astype(np.float32),
                    0.0, 1.0,
                )
                if solid is not None:
                    w.flame[solid] = 0.0

        # 5) Decay.
        decay = 0.05 * dt
        w.flame *= np.where(w.fuel > 0.05, 1.0 - 0.2 * decay, 1.0 - 5.0 * decay)
        w.flame = np.clip(w.flame, 0.0, 1.0)
        # Smoke settles slowly. The 0.005/s rate gives a half-life of
        # ~140s, long enough for a 10-min episode to fill a small house.
        w.smoke *= 1.0 - 0.005 * dt
        w.smoke = np.clip(w.smoke, 0.0, 1.0)

        # 6) Cooling toward ambient (small).
        w.temp = self.ambient + (w.temp - self.ambient) * np.exp(-0.005 * dt)

        # 7) Sustained sources: pin source voxels to a hot floor, and
        # inject a steady stream of smoke + small fuel + flame so the
        # plume keeps growing throughout the lifetime of the ignition.
        if self._sources:
            keep: List[Tuple[Tuple[slice, slice, slice], np.ndarray, float, float]] = []
            for sl, falloff, src_t, t_end in self._sources:
                if t_now >= t_end:
                    continue
                target = self.ambient + (src_t - self.ambient) * falloff
                w.temp[sl] = np.maximum(w.temp[sl], target)
                w.fuel[sl] = np.maximum(w.fuel[sl], 0.7 * falloff)
                w.flame[sl] = np.maximum(w.flame[sl], 0.6 * falloff)
                # Smoke source rate ~ 0.10/s at the source core; tied to
                # falloff so the boundary contributes less.
                w.smoke[sl] = np.minimum(
                    w.smoke[sl] + 0.10 * dt * falloff, 1.0
                )
                keep.append((sl, falloff, src_t, t_end))
            self._sources = keep


# ---------------------------------------------------------------------------
# High-level driver
# ---------------------------------------------------------------------------
def run_propagation(
    inventory: Dict,
    plan: Dict,
    voxel_m: float = 0.15,
    dt: float = 0.5,
    save_dt: float = 1.0,
    out_dir: Optional[Path] = None,
    keep_in_memory: bool = False,
    verbose: bool = True,
) -> Dict:
    """Integrate ``plan`` over time and write a timeline.

    Returns a dict with metadata; the actual fields go into a compressed
    .npz at ``out_dir/timeline.npz``. If ``keep_in_memory`` is True, also
    returns the per-frame arrays (used by the test harness).
    """
    rules = plan["propagation_rules"]
    duration = float(plan["duration_s"])
    seed = int(plan.get("seed", 0))

    world = VoxelWorld.from_aabb(plan["world_aabb"], voxel=voxel_m,
                                 ambient_c=float(rules.get("ambient_temp_c", 25.0)))
    # Prefer schema v2 'instances' (full inventory); fall back to legacy
    # 'objects' so older plan/inventory pairs keep working.
    items = inventory.get("instances") or inventory.get("objects", [])
    world.stamp_object_aabbs(items)
    # Attach structural masks if the inventory points to them. Walls and
    # ceilings then act as zero-flux barriers in propagation.step().
    struct = inventory.get("structural") or {}
    world.attach_structural_masks(
        wall_path=struct.get("wall_voxel_path"),
        floor_path=struct.get("floor_voxel_path"),
        ceiling_path=struct.get("ceiling_voxel_path"),
    )

    sim = FirePropagation(world, rules, seed=seed)

    # Schedule of ignitions sorted by time.
    ignitions = sorted(plan["ignitions"], key=lambda ig: float(ig["ignite_time_s"]))
    next_ig = 0
    n_steps = int(np.ceil(duration / dt))
    save_every = max(1, int(round(save_dt / dt)))

    n_frames = (n_steps + save_every - 1) // save_every + 1
    Nx, Ny, Nz = world.shape

    # Allocate timeline buffers as float16 to keep disk small.
    flame_t = np.zeros((n_frames, Nx, Ny, Nz), dtype=np.float16)
    smoke_t = np.zeros((n_frames, Nx, Ny, Nz), dtype=np.float16)
    temp_t = np.zeros((n_frames, Nx, Ny, Nz), dtype=np.float16)
    times = np.zeros((n_frames,), dtype=np.float32)

    def snapshot(frame_idx: int, t_now: float) -> None:
        flame_t[frame_idx] = world.flame.astype(np.float16)
        smoke_t[frame_idx] = world.smoke.astype(np.float16)
        # Temperature is centred on ambient and clipped before fp16 cast
        # so we don't lose precision in the saturation regime.
        temp_t[frame_idx] = np.clip(world.temp, -50.0, 1500.0).astype(np.float16)
        times[frame_idx] = float(t_now)

    # Apply any ignitions scheduled at t=0 before snapshot 0.
    while next_ig < len(ignitions) and ignitions[next_ig]["ignite_time_s"] <= 0.0:
        ig = ignitions[next_ig]
        sl, fall = world.kindle_ignition(
            np.asarray(ig["position"]),
            radius_m=float(ig["source_radius_m"]),
            temp_c=float(ig["source_temp_c"]),
            smoke_yield=float(ig.get("smoke_yield", 0.5)),
        )
        sustain = float(ig.get("sustain_s", 0.5 * duration))
        sim.add_source(sl, fall, float(ig["source_temp_c"]), sustain, t_now=0.0)
        next_ig += 1
    snapshot(0, 0.0)

    t0 = time.time()
    frame_i = 1
    for step_i in range(1, n_steps + 1):
        t_now = step_i * dt
        # Apply any ignitions whose time has arrived this step.
        while next_ig < len(ignitions) and ignitions[next_ig]["ignite_time_s"] <= t_now:
            ig = ignitions[next_ig]
            sl, fall = world.kindle_ignition(
                np.asarray(ig["position"]),
                radius_m=float(ig["source_radius_m"]),
                temp_c=float(ig["source_temp_c"]),
                smoke_yield=float(ig.get("smoke_yield", 0.5)),
            )
            sustain = float(ig.get("sustain_s", 0.5 * (duration - t_now)))
            sim.add_source(sl, fall, float(ig["source_temp_c"]), sustain, t_now=t_now)
            next_ig += 1

        sim.step(dt, t_now=t_now)

        if step_i % save_every == 0:
            if frame_i < n_frames:
                snapshot(frame_i, t_now)
                frame_i += 1
    # Trim trailing unused frames.
    flame_t = flame_t[:frame_i]
    smoke_t = smoke_t[:frame_i]
    temp_t = temp_t[:frame_i]
    times = times[:frame_i]

    elapsed = time.time() - t0
    if verbose:
        print(f"[propagation] {plan['scene_id']} {plan['plan_id']}: "
              f"shape={world.shape} dt={dt}s save_dt={save_dt}s "
              f"frames={frame_i} sim_time={duration:.0f}s wall={elapsed:.1f}s")

    meta = {
        "scene_id": plan["scene_id"],
        "plan_id": plan["plan_id"],
        "shape": list(world.shape),
        "voxel_m": float(voxel_m),
        "origin": world.origin.tolist(),
        "world_aabb": list(plan["world_aabb"]),
        "dt": float(dt),
        "save_dt": float(save_dt),
        "n_frames": int(frame_i),
        "duration_s": float(duration),
        "ambient_c": float(world.ambient_c),
        "ceiling_y_idx": int(sim.ceiling_y),
    }

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save meta as a 0-d numpy *unicode* array, never as pickled
        # object. This way an npz baked with one numpy minor version
        # loads cleanly under another (older numpys 1.x can't depickle
        # objects whose qualname mentions ``numpy._core`` because that
        # private path is new in 2.x). The plain unicode path needs no
        # pickle at all.
        meta_json_str = json.dumps(meta)
        np.savez_compressed(
            out_dir / "timeline.npz",
            flame=flame_t,
            smoke=smoke_t,
            temp=temp_t,
            times=times,
            meta_json=np.array(meta_json_str),
        )
        (out_dir / "timeline_meta.json").write_text(json.dumps(meta, indent=2))

    out: Dict[str, object] = {"meta": meta, "out_dir": str(out_dir) if out_dir else None}
    if keep_in_memory:
        out.update({
            "flame": flame_t,
            "smoke": smoke_t,
            "temp": temp_t,
            "times": times,
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Run fire propagation for a (scene, plan_id) pair."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--plan_id", required=True)
    parser.add_argument("--scenes_root", default="scenes")
    parser.add_argument("--out_root", default="outputs/fire_world")
    parser.add_argument("--voxel_m", type=float, default=0.15)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--save_dt", type=float, default=1.0)
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    inv = json.loads((scenes_root / args.scene / "inventory.json").read_text())
    plan_path = scenes_root / args.scene / "plans" / f"{args.plan_id}.json"
    plan = json.loads(plan_path.read_text())

    out_dir = Path(args.out_root) / args.scene / args.plan_id
    res = run_propagation(
        inventory=inv,
        plan=plan,
        voxel_m=args.voxel_m,
        dt=args.dt,
        save_dt=args.save_dt,
        out_dir=out_dir,
    )
    print(f"[propagation] wrote {res['out_dir']}/timeline.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
