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


def _box_sum_axis(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Cumulative-sum box SUM of width ``2*k+1`` along ``axis``.

    Unlike :func:`_box_blur_axis` this does NOT divide by the window
    size, so each input voxel contributes its full value to every
    output voxel within ``k`` cells. Used for radiative heating where
    each flame voxel acts as an independent radiation source and the
    receiver accumulates contributions from all flames within range.
    """
    if k <= 0:
        return a.copy()
    pad_widths = [(0, 0)] * a.ndim
    pad_widths[axis] = (k, k)
    pad = np.pad(a, pad_widths, mode="constant")
    csum = np.cumsum(pad, axis=axis)
    sl_hi = [slice(None)] * a.ndim
    sl_lo = [slice(None)] * a.ndim
    sl_hi[axis] = slice(2 * k, None)
    sl_lo[axis] = slice(0, -2 * k)
    return csum[tuple(sl_hi)] - csum[tuple(sl_lo)]


def _box_sum_3d(a: np.ndarray, k: int) -> np.ndarray:
    """Separable 3D box sum (no normalisation)."""
    if k <= 0:
        return a.copy()
    out = _box_sum_axis(a, k, 0)
    out = _box_sum_axis(out, k, 1)
    out = _box_sum_axis(out, k, 2)
    return out


def _box_blur_axis(a: np.ndarray, k: int, axis: int) -> np.ndarray:
    """Cumulative-sum box blur of width ``2*k+1`` along ``axis``."""
    if k <= 0:
        return a
    pad_widths = [(0, 0)] * a.ndim
    pad_widths[axis] = (k, k)
    pad = np.pad(a, pad_widths, mode="edge")
    csum = np.cumsum(pad, axis=axis)
    sl_hi = [slice(None)] * a.ndim
    sl_lo = [slice(None)] * a.ndim
    sl_hi[axis] = slice(2 * k, None)
    sl_lo[axis] = slice(0, -2 * k)
    blurred = csum[tuple(sl_hi)] - csum[tuple(sl_lo)]
    return blurred / float(2 * k + 1)


def _box_blur_3d(a: np.ndarray, k: int) -> np.ndarray:
    """Separable 3D box blur. Used for fuel-abundance and radiative gain."""
    if k <= 0:
        return a
    out = _box_blur_axis(a, k, 0)
    out = _box_blur_axis(out, k, 1)
    out = _box_blur_axis(out, k, 2)
    return out


def _gaussian_blur_3d(a: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Approximate isotropic 3D Gaussian via 3 stacked box blurs.

    Three box blurs of width ``2k+1`` approximate a Gaussian with
    sigma ~= k / sqrt(3). We pick k from the requested sigma. This is
    O(N) per axis and avoids the SciPy dependency.
    """
    if sigma_cells <= 0.0:
        return a
    k = max(1, int(round(float(sigma_cells) * np.sqrt(3.0))))
    out = a
    for _ in range(3):
        out = _box_blur_3d(out, k)
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
        self.spread = float(rules.get("spread_speed_m_per_s", 0.12))
        self.buoy = float(rules.get("buoyancy_v_m_per_s", 0.5))
        self.cj = float(rules.get("ceiling_jet_speed_m_per_s", 0.30))
        self.ambient = float(rules.get("ambient_temp_c", 25.0))
        # Reaction rate (kept low + linear-with-fuel for stability and so
        # the source survives several minutes, matching real fire growth).
        self.k_burn = float(rules.get("k_burn_per_s", 1.0 / 240.0))
        # Heat released per (fuel-fraction * dt) at the source voxel.
        self.q_release_c = float(rules.get("q_release_c", 350.0))
        # Radiative pre-heating: 0 disables; positive values let the
        # flame heat any fuel within `radiative_radius_cells` voxels
        # by `radiative_gain_c * dt * blurred_flame` per step. This is
        # what bridges the gap between disjoint pieces of furniture so
        # a kitchen fire can ignite a chair 1 m away.
        self.radiative_gain_c = float(rules.get("radiative_gain_c", 200.0))
        self.radiative_radius_cells = int(rules.get("radiative_radius_cells", 4))

        # ---- Structural transport ----------------------------------------
        # Walls and ceilings remain zero-flux thermal barriers. Floors are
        # "semi-transparent" - heat does conduct vertically across them
        # but multiplied by ``floor_thermal_attenuation``. This lets a
        # fire on one storey slowly raise the ceiling/floor assembly's
        # temperature on the storey below, matching the behaviour
        # documented in NFPA 921 §5.10 (fire-rated assemblies still
        # transmit heat over minutes).
        self.floor_thermal_attenuation = float(
            rules.get("floor_thermal_attenuation", 0.20)
        )
        # Should the flame field also leak across floors? Default off
        # because real flames don't physically tunnel through structural
        # decking; the heat-only path is enough to ignite fuel below.
        self.flame_through_floors = bool(
            int(rules.get("flame_through_floors", 0))
        )

        # ---- Fuel-abundance-driven flame growth -------------------------
        # Local fuel abundance modulates burn rate and the upper cap on
        # flame intensity, so a corner of the room with lots of cushions
        # / blankets / curtains burns brighter than an isolated chair.
        self.fuel_neighborhood_cells = int(
            rules.get("fuel_neighborhood_cells", 2)
        )
        self.fuel_abundance_min = float(rules.get("fuel_abundance_min", 0.5))
        self.fuel_abundance_max = float(rules.get("fuel_abundance_max", 2.5))

        # ---- Spread kernel: 'laplacian' | 'gaussian' --------------------
        # The legacy laplacian gives 1-voxel-radius transport per step;
        # gaussian uses a 3-iteration separable box-blur approximation
        # whose sigma is set by ``spread_speed_m_per_s * dt`` (in voxel
        # units). Gaussian spreads more isotropically and is closer to
        # the smoothed plume profiles seen in NIST FDS validation tests.
        self.spread_kernel = str(rules.get("spread_kernel", "laplacian")).lower()

        # ---- Sustained / inextinguishable sources ----------------------
        # When 1, the per-source sustain_s timer is ignored and each
        # source keeps replenishing its own solid fuel every step, so
        # a single point ignition can drive a room-filling fire over
        # the full episode. Set 0 to model a finite fuel package that
        # eventually burns itself out.
        self.inextinguishable_sources = bool(
            int(rules.get("inextinguishable_sources", 1))
        )

        # ---- Floor-as-fuel + flame column height -----------------------
        # When the temperature on a floor voxel exceeds
        # ``floor_ignite_temp_c`` we deposit a synthetic fuel layer
        # there of magnitude ``floor_fuel_value``. This is how the
        # carpet / hardwood ignites and lets fire crawl across the
        # floor between two pieces of furniture, instead of always
        # needing the agent to be in line of sight of a source.
        self.floor_ignite_temp_c = float(rules.get("floor_ignite_temp_c", 250.0))
        self.floor_fuel_value = float(rules.get("floor_fuel_value", 0.7))
        # Direct contact ignition: any floor voxel within
        # ``floor_ignite_radius_cells`` of an existing flame voxel
        # whose intensity exceeds ``floor_flame_contact_thresh`` gets
        # the same synthetic fuel deposit. This bypasses heat
        # diffusion's smoothing (which kills the local thermal spike
        # in a single step) and matches the physical observation that
        # flames touching the floor ignite the carpet directly.
        self.floor_ignite_radius_cells = int(
            rules.get("floor_ignite_radius_cells", 2)
        )
        self.floor_flame_contact_thresh = float(
            rules.get("floor_flame_contact_thresh", 0.2)
        )
        # Range of per-voxel seed flame magnitudes used to break the
        # flat-sheet look on the ignited floor. Each new floor voxel
        # gets a uniform draw in this range, scaled by floor_fuel_value.
        self.floor_seed_flame_min = float(rules.get("floor_seed_flame_min", 0.1))
        self.floor_seed_flame_max = float(rules.get("floor_seed_flame_max", 0.65))
        # Flame column height: every active flame voxel projects an
        # upward flame plume of up to ``flame_column_cells`` voxels
        # (default 6 cells ~= 0.9 m on a 0.15 m grid). The plume
        # intensity decays linearly with height so the top of the
        # flame is wispier than the base. This is purely a renderer
        # cue - the upper voxels of the column don't consume fuel.
        self.flame_column_cells = int(rules.get("flame_column_cells", 6))
        self.flame_column_decay = float(rules.get("flame_column_decay", 0.75))

        # ---- Smoke transport tuning -------------------------------------
        # smoke_alpha is a *self-diffusion* coefficient (m^2/s), independent
        # of thermal_diffusivity. Without a smoke Laplacian the plume
        # would be locked to the hot mask and pile under the ceiling
        # forever. 0.02 m^2/s on a 0.15 m grid spreads smoke ~0.6 m in
        # 30 s of fire-time, enough to fill a small room while still
        # keeping the source peak high.
        self.smoke_alpha = float(rules.get("smoke_diffusivity", 0.02))
        # CFL safety factor for the smoke Laplacian. 6-point stencil
        # stability requires alpha*dt/v^2 < 1/6 ~ 0.166, but values that
        # close to the limit aggressively wash out bright source pulses
        # in a single substep. 0.08 keeps the source peak coherent.
        self.smoke_cfl = float(rules.get("smoke_cfl", 0.08))
        # Fraction of the buoyancy step applied to the smoke field. <1.0
        # keeps smoke near the source for a few seconds before it pins
        # to the ceiling, which is what allows the eye-height visibility
        # drop to actually show up in renders.
        self.smoke_buoy_frac = float(rules.get("smoke_buoy_fraction", 0.5))
        # Reaction-driven smoke production multiplier. The default 4.0 is
        # tuned so a flame voxel produces enough smoke per second to
        # cross the visibility threshold (smoke=0.3) within ~30 s.
        self.smoke_reaction_gain = float(rules.get("smoke_reaction_gain", 4.0))
        # Multiplier on the per-source sustained smoke injection rate
        # (`source_inject_rate = smoke_source_gain * smoke_yield`). 4.0
        # pumps a kitchen source (smoke_yield=0.85) at 3.4/s peak, which
        # saturates a voxel within one second and lets self-diffusion
        # carry the plume outward. Raising this is the right knob if a
        # specific scene still feels too "thin".
        self.smoke_source_gain = float(rules.get("smoke_source_gain", 4.0))
        # First-order smoke decay rate (1/s). 0.0008 = ~860 s half-life,
        # so a 900 s episode keeps a visible plume after the fire dies.
        # The original 0.005 (140 s half-life) silently scrubbed the
        # whole house by t=720 s.
        self.smoke_decay_per_s = float(rules.get("smoke_decay_per_s", 0.0008))

        Nx, Ny, Nz = world.shape
        # Locate the ceiling Y as the world's top layer. If we ever add a
        # building-aware ceiling map, this becomes per-(x,z).
        self.ceiling_y = int(Ny - 1)

        # Active ignitions whose source we keep warm each step until they
        # burn out. Each entry: (slice_tuple, falloff, source_temp_c,
        # t_end, smoke_yield). We carry smoke_yield through so the plan
        # value really controls how dense the room ends up.
        self._sources: List[Tuple[Tuple[slice, slice, slice], np.ndarray, float, float, float]] = []

    # ------------------------------------------------------------------
    def add_source(
        self,
        sl: Tuple[slice, slice, slice],
        falloff: np.ndarray,
        source_temp_c: float,
        sustain_s: float,
        t_now: float,
        smoke_yield: float = 0.5,
    ) -> None:
        """Register a sustained heat source.

        Each step until t_now+sustain_s the source voxels are pinned to
        at least ``ambient + (T_src - ambient) * falloff`` AND a
        smoke-injection rate proportional to ``smoke_yield`` is added.
        Plumbing ``smoke_yield`` through here is what actually links the
        plan field of the same name to the rendered smoke density: the
        previous version hard-coded 0.10/s for every source.
        """
        self._sources.append((
            sl, falloff,
            float(source_temp_c), t_now + float(sustain_s),
            float(np.clip(smoke_yield, 0.0, 1.0)),
        ))

    # ------------------------------------------------------------------
    def step(self, dt: float, t_now: float = 0.0) -> None:
        """One forward step. ``dt`` in seconds, ``t_now`` end-of-step time."""
        w = self.world
        v = w.voxel

        # Solid (impermeable to heat/smoke). Wall + ceiling block the
        # plume from leaking through; floors are handled separately
        # below as semi-transparent thermal barriers.
        solid: Optional[np.ndarray] = None
        for m in (w.walls, w.ceilings):
            if m is not None:
                solid = m if solid is None else (solid | m)
        # Original solid mask used by the smoke / spread transport
        # operators - smoke can still pile under a floor (we just don't
        # render it through one), so floors stay in the smoke-side mask
        # to avoid silently leaking soot into the layer below.
        solid_strict: Optional[np.ndarray] = solid
        if w.floors is not None:
            solid_strict = (
                w.floors if solid_strict is None else (solid_strict | w.floors)
            )

        # 1) Heat diffusion (sub-stepped to respect CFL: alpha*dt/v^2 <= 0.16).
        # Floors are NOT zero-flux; we run the Laplacian as if the floor
        # were air, then attenuate the *change* across floor voxels by
        # ``floor_thermal_attenuation``. This gives a slow vertical heat
        # transfer between storeys while still keeping the steady-state
        # temperature on the cool side bounded.
        max_diff_dt = 0.16 * v * v / max(self.alpha, 1e-6)
        n_sub = max(1, int(np.ceil(dt / max_diff_dt)))
        sub_dt = dt / n_sub
        coeff = self.alpha * sub_dt / max(v * v, 1e-6)
        floor_attn = float(np.clip(self.floor_thermal_attenuation, 0.0, 1.0))
        for _ in range(n_sub):
            lap = _laplacian_3d(w.temp)
            if w.floors is not None and floor_attn < 1.0:
                lap = np.where(w.floors, lap * floor_attn, lap)
            w.temp = w.temp + coeff * lap
            if solid is not None:
                w.temp[solid] = self.ambient
        np.clip(w.temp, -50.0, 1500.0, out=w.temp)

        # 1b) Smoke self-diffusion (independent of temperature).
        # Without this, smoke gets stuck inside the hot mask and never
        # actually fills the room, even though the source pumps more in
        # every step. The CFL substep target is ``smoke_cfl < 1/6``.
        # We deliberately do NOT zero smoke in solid voxels here -
        # zeroing inside a transport operator silently leaks mass at
        # every wall touch and produces a spurious ~30 s smoke
        # half-life. The renderer never samples through solid voxels,
        # so smoke "trapped" in a wall is invisible but mass-conserved.
        if self.smoke_alpha > 0.0:
            cfl = float(np.clip(self.smoke_cfl, 1e-3, 0.16))
            max_diff_dt_s = cfl * v * v / max(self.smoke_alpha, 1e-6)
            n_sub_s = max(1, int(np.ceil(dt / max_diff_dt_s)))
            sub_dt_s = dt / n_sub_s
            coeff_s = self.smoke_alpha * sub_dt_s / max(v * v, 1e-6)
            for _ in range(n_sub_s):
                lap_s = _laplacian_3d(w.smoke)
                w.smoke = np.clip(w.smoke + coeff_s * lap_s, 0.0, 1.0)

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

            # Smoke buoyancy is a fraction of the temperature buoyancy:
            # smoke is denser than air at room temperature once it cools,
            # so it should not jet to the ceiling as fast as the hot
            # gas does. Empirically smoke_buoy_frac=0.5 keeps a visible
            # plume at eye height for several seconds before it pins to
            # the ceiling.
            frac_b_smoke = frac_b * float(np.clip(self.smoke_buoy_frac, 0.0, 1.0))
            if frac_b_smoke > 0.0:
                # Apply buoyancy to ALL smoke voxels (not only hot ones)
                # so the plume keeps rising after it leaves the hot mask.
                w.smoke = _shift_up_y(w.smoke, frac_b_smoke)
                w.smoke = np.clip(w.smoke, 0.0, 1.0)
                # NOTE: we deliberately do NOT zero smoke inside solid
                # voxels here. ``_shift_up_y`` reflects mass at the top
                # of the grid; for inner ceilings (e.g. mezzanines)
                # zeroing would silently delete mass every step,
                # producing the spurious ~30 s smoke half-life users
                # observed. Renderer rays terminate at the geometry's
                # depth, so smoke "trapped" in a solid voxel is never
                # actually sampled and is harmless visually.

        # 3) Ceiling jet (under the actual ceiling surface if available).
        frac_cj = float(np.clip(self.cj * dt / v, 0.0, 0.95))
        if frac_cj > 0.0:
            for _ in range(2):
                _ceiling_jet(w.temp, self.ceiling_y, frac_cj)
                _ceiling_jet(w.smoke, self.ceiling_y, frac_cj)
            if solid is not None:
                # Same reasoning as the buoyancy step: only the
                # *temperature* field is constrained back to ambient at
                # solid voxels; the smoke field is allowed to remain so
                # the box-blur cannot silently bleed mass into walls.
                w.temp[solid] = self.ambient
        # Diffusion accumulated some smoke in solid voxels too; zero it
        # only once per step, at the source-injection step below, so
        # downstream operators don't keep depositing fresh mass and
        # leaking it.

        # 4) Reaction.
        # Local fuel abundance gives a corner of the room with lots of
        # cushions / blankets a faster burn rate and a higher flame cap
        # than an isolated chair, so flame magnitude actually scales
        # with how much there is to burn. abundance is in [min, max]
        # (default [0.5, 2.5]) - 0.5 means very sparse fuel, 2.5 means
        # the burn voxel is surrounded by fuel on every side.
        ignitable = (w.fuel > self.flammable_threshold) & (w.temp > self.t_ignite)
        if ignitable.any():
            k_neigh = max(1, int(self.fuel_neighborhood_cells))
            fuel_avg = _box_blur_3d(w.fuel, k_neigh)
            abundance = np.clip(
                fuel_avg / max(self.flammable_threshold, 1e-3),
                self.fuel_abundance_min, self.fuel_abundance_max,
            )
            burn_rate = self.k_burn * dt * w.fuel * ignitable * abundance
            burn_rate = np.minimum(burn_rate, 0.6)
            flame_cap = np.minimum(1.0, 0.85 + 0.15 * abundance)
            grown = w.flame + burn_rate
            new_flame = np.minimum(grown, flame_cap)
            # Preserve voxels that were already above cap (e.g. sources
            # pinned by step 7).
            w.flame = np.maximum(w.flame, new_flame).astype(np.float32)
            w.fuel = np.clip(w.fuel - burn_rate, 0.0, 1.0)
            heat = np.minimum(self.q_release_c * burn_rate, 200.0)
            w.temp = np.clip(w.temp + heat, -50.0, 1500.0)
            w.smoke = np.clip(w.smoke + self.smoke_reaction_gain * burn_rate, 0.0, 1.0)

        # 4b) Floor ignition.
        # The floor mask is normally a zero-flux thermal barrier for
        # walls / ceilings, but realistic room fires spread across the
        # carpet / hardwood once it gets hot enough. There are two
        # ways for a floor voxel to ignite:
        #
        #   (a) Direct contact: any flame voxel within
        #       ``floor_ignite_radius_cells`` of the floor cell. This
        #       is how the fire crawls outward from a furniture base.
        #       Without this term, heat diffusion's 1-step thermal
        #       smoothing kills the sharp temperature spike at the
        #       newly-pre-heated voxel before the reaction loop can
        #       use it (the spike gets spread to neighbours, cooling
        #       the centre below ignition_temp_c).
        #   (b) Sustained heating: the integrated temperature on a
        #       floor voxel exceeds ``floor_ignite_temp_c``, the
        #       traditional autoignition path that handles a fire
        #       above the ceiling igniting a floor on the next
        #       storey.
        if w.floors is not None and self.floor_fuel_value > 0.0:
            ignite_radius = max(1, int(self.floor_ignite_radius_cells))
            # (a) direct contact via local flame sum
            f_local = _box_sum_3d(
                (w.flame > self.floor_flame_contact_thresh).astype(np.float32),
                ignite_radius,
            )
            contact_ignite = w.floors & (f_local > 0.5)
            # (b) sustained autoignition
            hot_ignite = w.floors & (w.temp > self.floor_ignite_temp_c)
            ignite = contact_ignite | hot_ignite
            if ignite.any():
                w.fuel[ignite] = np.maximum(
                    w.fuel[ignite], float(self.floor_fuel_value)
                )
                # Seed flame with PER-VOXEL random magnitude so the
                # ignited floor doesn't look like a flat sheet of light.
                # Each newly-ignited voxel gets a value uniformly in
                # [seed_flame_min, seed_flame_max], creating
                # tongue-like heterogeneity that propagation then
                # preserves through subsequent steps.
                ignite_count = int(ignite.sum())
                seed_base = float(self.floor_fuel_value)
                noise01 = self.rng.random(ignite_count, dtype=np.float32)
                seed_flame_per = (
                    seed_base * (self.floor_seed_flame_min
                                 + (self.floor_seed_flame_max
                                    - self.floor_seed_flame_min) * noise01)
                )
                w.flame[ignite] = np.maximum(w.flame[ignite], seed_flame_per)
                # And push the temperature up to t_ignite so the
                # reaction-loop predicate fires next step. Temperature
                # is also randomised so subsequent burn_rate is
                # spatially heterogeneous (faster patches stay hotter).
                temp_jitter = self.rng.random(ignite_count, dtype=np.float32) * 100.0
                w.temp[ignite] = np.maximum(
                    w.temp[ignite],
                    float(self.t_ignite) + 50.0 + temp_jitter,
                )

        # Surface flame spread along the fuel field.
        # spread_mask = (fuel > 0) OR (temp > t_ignite). The latter
        # term lets flame jump *across* short air gaps that have been
        # pre-heated by the radiative-gain step, so fire actually
        # crosses from one piece of furniture to the next instead of
        # being trapped inside a single AABB.
        if self.spread > 0.0:
            kernel_frac = float(np.clip(self.spread * dt / v, 0.0, 0.30))
            if kernel_frac > 0.0:
                spread_mask = (
                    (w.fuel > 0) | (w.temp > self.t_ignite)
                ).astype(np.float32)
                if self.spread_kernel == "gaussian":
                    # Gaussian transport: sigma = spread * dt / v
                    # voxels per step. We blend the original flame with
                    # its blurred copy weighted by ``kernel_frac`` so the
                    # operator stays mass-conserving (kernel sums to 1)
                    # and reduces to ``flame`` when kernel_frac = 0.
                    sigma_cells = float(self.spread * dt / max(v, 1e-6))
                    blurred = _gaussian_blur_3d(w.flame, sigma_cells)
                    delta = (blurred - w.flame) * spread_mask * kernel_frac
                    w.flame = np.clip(w.flame + delta, 0.0, 1.0)
                else:
                    lap = _laplacian_3d(w.flame)
                    w.flame = np.clip(
                        w.flame + kernel_frac * lap * spread_mask,
                        0.0, 1.0,
                    )
                # Walls / ceilings still block flame; floors only block
                # if the user opts out of cross-floor flame travel.
                if solid is not None:
                    w.flame[solid] = 0.0
                if (
                    not self.flame_through_floors
                    and w.floors is not None
                ):
                    w.flame[w.floors] = 0.0

        # Conduction: dump heat from flame voxels into the 6 face
        # neighbours so adjacent fuel can cross the ignition_temp
        # threshold even when bulk diffusion is slow. Without this
        # step the only thermal path to neighbours is the alpha
        # Laplacian, which keeps T<350C in 99% of voxels and stalls
        # spread completely.
        if self.q_release_c > 0.0:
            f = np.clip(w.flame, 0.0, 1.0)
            heat_pad = self.q_release_c * 0.6 * dt * f
            cond = np.zeros_like(w.temp)
            cond[1:, :, :]  += heat_pad[:-1, :, :]
            cond[:-1, :, :] += heat_pad[1:, :, :]
            cond[:, 1:, :]  += heat_pad[:, :-1, :]
            cond[:, :-1, :] += heat_pad[:, 1:, :]
            cond[:, :, 1:]  += heat_pad[:, :, :-1]
            cond[:, :, :-1] += heat_pad[:, :, 1:]
            cond /= 6.0
            # Deposit conduction heat into fuel-bearing voxels AND
            # floor voxels. Floors don't carry the inventory's
            # flammability stamp but we still want them to heat up so
            # the floor_ignite_temp_c threshold gets crossed and the
            # synthetic floor fuel kicks in.
            burnable = (w.fuel > 0.05)
            if w.floors is not None:
                burnable = burnable | w.floors
            w.temp = np.clip(w.temp + cond * burnable.astype(np.float32),
                             -50.0, 1500.0)

        # Radiative pre-heating: each flame voxel is treated as a point
        # source that radiates into a cubic neighbourhood. We use a
        # *sum* (not mean) box filter so a chunk of fire surrounded by
        # many lit voxels heats up much faster than an isolated source
        # - exactly what physically happens when fire enters a room
        # full of burning furniture. The summed contribution is then
        # normalised by a target effective radius so the gain scales
        # with absolute flame coverage rather than relative density.
        if self.radiative_gain_c > 0.0:
            r = max(1, int(self.radiative_radius_cells))
            f = np.clip(w.flame, 0.0, 1.0)
            # Box-sum: each lit voxel deposits its full flame value in
            # every output voxel within +/- r cells. Output magnitude
            # scales with the *number* of flame voxels in range, so a
            # well-developed fire saturates nearby air much faster.
            ff = _box_sum_3d(f, r)
            # Normalise by sqrt(window) instead of full window so the
            # heat stays bounded but a fully-lit window still produces
            # a strong gradient at the room edge.
            ff /= np.sqrt(float((2 * r + 1) ** 3))
            burnable = (w.fuel > 0.05)
            if w.floors is not None:
                burnable = burnable | w.floors
            w.temp = np.clip(
                w.temp + self.radiative_gain_c * dt * ff * burnable.astype(np.float32),
                -50.0, 1500.0,
            )

        # 5) Decay.
        decay = 0.05 * dt
        # Voxels carrying fuel (or recently deposited floor fuel) keep
        # their flame; flame stranded in cool air without fuel decays
        # quickly.
        w.flame *= np.where(w.fuel > 0.05, 1.0 - 0.2 * decay, 1.0 - 5.0 * decay)
        w.flame = np.clip(w.flame, 0.0, 1.0)
        # Smoke decays much more slowly than the previous 0.005/s rate
        # (~140 s half-life), which scrubbed the room before the agent
        # could navigate. The default 0.001/s gives a ~700 s half-life,
        # so a 900 s episode keeps a visible plume.
        w.smoke *= 1.0 - self.smoke_decay_per_s * dt
        w.smoke = np.clip(w.smoke, 0.0, 1.0)

        # 6) Cooling toward ambient (small).
        w.temp = self.ambient + (w.temp - self.ambient) * np.exp(-0.005 * dt)

        # 7) Sustained sources: pin source voxels to a hot floor, and
        # inject a steady stream of smoke + small fuel + flame so the
        # plume keeps growing throughout the lifetime of the ignition.
        # When ``inextinguishable_sources`` is enabled, the per-source
        # ``sustain_s`` timer is ignored - the source burns until the
        # simulation ends, mimicking a real fire that found a
        # continuous fuel supply (e.g. natural gas line, large
        # furniture load, structural lumber).
        if self._sources:
            keep: List[Tuple[Tuple[slice, slice, slice], np.ndarray, float, float, float]] = []
            for entry in self._sources:
                sl, falloff, src_t, t_end, smoke_yield = entry
                if t_now >= t_end and not self.inextinguishable_sources:
                    continue
                target = self.ambient + (src_t - self.ambient) * falloff
                w.temp[sl] = np.maximum(w.temp[sl], target)
                # Inextinguishable mode: keep replenishing solid fuel at
                # the source so the point ignition never burns itself
                # out. The reaction step still consumes fuel each
                # frame; this just makes sure the source has more to
                # burn next frame. Set to 0 in the plan to model a
                # finite fuel package (the source then dies once it
                # runs out, even before sustain_s is up).
                if self.inextinguishable_sources:
                    w.fuel[sl] = np.maximum(w.fuel[sl], 0.85 * falloff)
                else:
                    w.fuel[sl] = np.maximum(w.fuel[sl], 0.7 * falloff)
                # Source flame is pinned to the abundance-aware ceiling
                # (0.85 + 0.15 * 2.5 = 1.225 in dense fuel zones,
                # clamped at 1.0). Without this, the previous 0.6
                # ceiling capped the rendered flame intensity even in
                # rooms full of cushions where reaction step would
                # otherwise drive it close to 1.
                w.flame[sl] = np.maximum(w.flame[sl], 0.95 * falloff)
                # Plan-aware smoke injection. The source's
                # ``smoke_yield`` is multiplied by ``smoke_source_gain``
                # to get a per-second injection rate at the source core
                # (kitchen smoke_yield=0.85 with default gain=4.0 ->
                # 3.4/s, saturating a voxel in <1s); falloff scales it
                # spatially. The voxel is clipped at 1.0 so this is
                # safe even with very high gains.
                inject_rate = self.smoke_source_gain * smoke_yield
                w.smoke[sl] = np.minimum(
                    w.smoke[sl] + inject_rate * dt * falloff, 1.0
                )
                keep.append(entry)
            self._sources = keep

        # 8) Flame column: project flame upward from each burning voxel
        # so the rendered fire has a visible vertical plume instead of
        # rendering as a flat sheet at the fuel's height. For each y
        # layer above an active flame (within ``flame_column_cells``)
        # we copy the source flame intensity decayed by a geometric
        # factor (``flame_column_decay``) per layer and OR it with
        # whatever is already there. Walls / ceilings block the
        # column (zero out at solid voxels). Top of the column also
        # contributes to the smoke field so the plume bridges into
        # the buoyancy step on the next iteration.
        if self.flame_column_cells > 0 and self.flame_column_decay > 0.0:
            base = np.clip(w.flame, 0.0, 1.0)
            current = base.copy()
            col_acc = base.copy()
            for h in range(1, int(self.flame_column_cells) + 1):
                # Shift the flame up by one voxel in y; reflect at top.
                lifted = np.zeros_like(current)
                lifted[:, h:, :] = base[:, :-h, :]
                lifted *= float(self.flame_column_decay) ** h
                if solid is not None:
                    lifted[solid] = 0.0
                col_acc = np.maximum(col_acc, lifted)
            # Apply only where the new column intensity exceeds the
            # existing flame value so the reaction step's fuel-anchored
            # peaks aren't smeared.
            w.flame = np.maximum(w.flame, col_acc).astype(np.float32)
            # Top of the column also produces a little smoke. Linear
            # in column height so a tall plume = darker smoke head.
            extra_smoke = col_acc * 0.05 * dt
            w.smoke = np.clip(w.smoke + extra_smoke, 0.0, 1.0)


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
        sustain = float(ig.get("sustain_s", duration - 0.0))
        sim.add_source(sl, fall, float(ig["source_temp_c"]), sustain,
                       t_now=0.0,
                       smoke_yield=float(ig.get("smoke_yield", 0.5)))
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
            sustain = float(ig.get("sustain_s", duration - t_now))
            sim.add_source(sl, fall, float(ig["source_temp_c"]), sustain,
                           t_now=t_now,
                           smoke_yield=float(ig.get("smoke_yield", 0.5)))
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
