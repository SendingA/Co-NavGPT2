# FireWorld: A Benchmark for Multi-Robot Object-Goal Navigation under Dynamic Fire and Sensor Degradation

> This document is both an engineering reference for the `utils/fire_world`
> and `utils/fire_sensors` subsystems **and** a writing template for the
> benchmark paper. §0–§6 describe the implementation as it exists in the
> repository; §7 is a ready-to-adapt Methodology / Benchmark-Construction
> section; the appendices give a file map and a minimal reproduction recipe.
>
> All quantitative claims below were checked against the source. Where a
> value is a tunable default it is stated as such and the owning symbol is
> named, so the paper can cite the exact knob.
>
> The optional navigation-time risk layer (evaluator-only, privileged oracle,
> and sensed belief modes; hazard-aware frontier/FMM planning; CHE outputs) is
> documented separately in [`docs/risk_assessment.md`](risk_assessment.md).

---

## 0. Motivation and Contributions

Embodied navigation benchmarks (Habitat, HM3D, MP3D) assume a **static**
world observed through **clean** sensors. Real emergency-response and
search-and-rescue settings violate both assumptions simultaneously: the
environment changes over time (fire and smoke spread), and the sensor
stack degrades in a modality-dependent way (RGB is blinded by smoke,
depth/LiDAR lose returns, while thermal IR and mmWave radar remain
largely usable).

**FireWorld** augments Habitat–HM3D multi-robot ObjectGoal Navigation
with two additions:

1. a **deterministic, time-varying 3D fire/smoke/temperature field**
   grounded in HM3D semantic geometry (the *world model*), and
2. a **multi-modal sensor-degradation model** that renders how each
   robot perceives that field (the *observation model*).

The central design principle is that the world model and the observation
model are **fully decoupled and independently ablatable**:

| Layer | Responsibility | Code |
| --- | --- | --- |
| World model — `utils/fire_world` | 3D voxel fields (flame / smoke / temperature), propagation physics, fire-time clock, camera-pose conversion | `scene_scan.py`, `templates.py`, `planner.py`, `propagation.py`, `voxel_world.py`, `runtime.py`, `scene.py` |
| Observation model — `utils/fire_sensors` | volumetric camera, smoke-degraded depth, mmWave radar, 360° LiDAR, thermal IR, dashboard | `voxel_render.py`, `sensors/*`, `lidar_360.py`, `suite.py`, `config.py` |

**Claimed contributions (paper §1):**

- **C1 — A reproducible dynamic-hazard benchmark.** Every artefact is a
  deterministic function of its inputs and content-addressed by a
  SHA1-truncated `plan_id`, so a full experimental run is bit-for-bit
  reproducible from three scalars `(fire_type, intensity, seed)`.
- **C2 — Semantically grounded fire.** Ignition sources, fuel loads, and
  smoke yields are derived from HM3D per-instance semantics, not painted
  by hand, so the hazard is spatially consistent with the scene.
- **C3 — A decoupled multi-modal degradation model.** RGB, depth, LiDAR,
  radar, and thermal are each degraded by a physically-motivated model
  keyed to the same underlying smoke field, enabling per-modality
  ablations (e.g. "does LWIR rescue detection in dense smoke?").
- **C4 — An explicit fire-to-agent time coupling.** The rate at which the
  fire evolves relative to agent actions is a first-class, reportable
  experimental axis.

### Pipeline at a glance

```
HM3D scene (semantic.glb + semantic.txt)
      │
      ▼
Stage 1  scene_scan ──────► scenes/<id>/inventory.json (schema v2)
      │                       + structural/{walls,floors,ceilings}.npy
      ▼
Stage 2  planner ─────────► scenes/<id>/plans/<plan_id>.json
      │
      ▼
Stage 3  propagation ─────► outputs/fire_world/<id>/<plan_id>/timeline.npz
      │                       + timeline_meta.json
      ▼
Stage 4  FireScene ───────► runtime world-model facade (timeline + clock + pose)
      │
      ▼
Stage 5  FireSensorSuite ─► per-step multi-modal observations
      │
      ▼
Stage 6  main.py / main_vec.py / scripts/keyboard_teleop_fire.py
         (navigation loop, obs integration, dashboard)
      │
      └── optional risk assessment (main.py only)
              ─► planner belief + independent GT metrics
```

Every stage writes a serialisable, cacheable, deterministic artefact.

---

## 1. Stage 1 — Scene scan: HM3D semantics → structured inventory

**Goal.** Parse a single HM3D scene's ~600 semantic instances (furniture,
walls, floors, doors, …) into a fire-simulation-ready description
(`inventory.json`, schema v2). Entry point: `utils/fire_world/scene_scan.py`.

**Technical points (all verified in code):**

1. **sRGB decoding of vertex colours.** HM3D `*.semantic.glb` encodes
   instance IDs in the per-vertex `COLOR_0` attribute (uint16, linear
   intensity). Recovering the 24-bit colour that matches `*.semantic.txt`
   requires passing the linear value through the IEC 61966-2-1 sRGB OETF;
   a naïve linear remap misses essentially every instance. Decoding lives
   in `utils/fire_world/hm3d_semantic.py`.
2. **Face-level instance aggregation.** HM3D mesh primitives are built for
   view-frustum culling, not per-instance grouping, so one primitive can
   contain many instance colours. We aggregate geometry at **face**
   granularity (`aggregate_instances`), not mesh granularity, and record
   each instance's AABB, centroid, vertex/face counts.
3. **Coordinate frame.** GLB is +Z-up, Habitat is +Y-up; AABBs are
   converted to the Habitat world frame during aggregation.
4. **Material assignment.** Each category is mapped to
   `(flammability, smoke_yield) ∈ [0,1]²` via `MATERIAL_TABLE` in
   `scene_scan.py` (e.g. `bed=(0.75,0.80)`, `curtain=(0.85,0.75)`,
   `stove=(0.85,0.70)`, `oven and stove=(0.85,0.70)`,
   `sink=(0.05,0.05)`). Every category named by a scenario template has
   an explicit table entry using the exact lowercase spelling from
   HM3D `semantic.txt`; detector-only aliases such as `tv_monitor` are
   not accepted. Categories in
   `STRUCTURAL_CATEGORIES` (wall/floor/ceiling/door/window/column/…) are
   forced to `(0,0)` and tagged `structural=True` so they can never be
   selected as ignition sources.
5. **Structural voxel rasterisation.** Wall / floor / ceiling AABBs are
   stamped into three `(Nx,Ny,Nz)` boolean masks at `voxel_m=0.10` (default)
   and saved as `.npy`. Because HM3D floor/ceiling instances have AABBs
   that span the full storey height, only the **bottom** `floor_slab_cells`
   (resp. top `ceiling_slab_cells`) voxels are stamped
   (`rasterise_structural_voxels`); stamping the raw AABB would fill the
   whole room volume and trap the fire. These masks act as zero-flux
   thermal boundaries and ceiling-jet surfaces in Stage 3.
6. **Floor clustering.** A 1-D chain-merge over instance y-extents
   (`cluster_floors`, gap threshold 1.5 m) groups instances into storeys,
   so the planner can restrict fire spread to a single floor.

**Output (`inventory.json`, schema v2):** `scene_id`, `world_aabb`,
`voxel_m`, `floors[]`, a full `instances[]` list (with `flammability`,
`smoke_yield`, `structural`, `is_goal`, `floor_id`), a legacy
`objects[]` alias (goal-flagged instances only, for backward compat), a
`semantic_summary` of per-region histograms, and `structural` voxel-path
pointers.

**Paper framing.** We recover **all** instance categories (not only the
ObjectGoal targets) and ground the world model in HM3D's semantic ground
truth, which is what makes the hazard spatially consistent (C2).

---

## 2. Stage 2 — Planner: (fire_type, intensity, seed) → reproducible plan

**Goal.** From the inventory, select only the objects ignited at `t=0`,
assign their source temperature and smoke yield, and emit the propagation
rule set. Later object ignition is an output of the solver. Code:
`utils/fire_world/templates.py` (scenario logic) and
`utils/fire_world/planner.py` (CLI + hashing).

**Deterministic identity.** A plan is content-addressed by a
SHA1-truncated hash of its inputs (scene, fire_type, intensity, seed,
template version, ignition-selection version, and optional explicit initial
count), so the same inputs always yield the same `plan_id` and the same
ignitions (C1).

**Template-based source selection** (`TEMPLATES` in `templates.py`). Each
template samples one or more initial sources by flammability from its declared
primary/fallback categories and ignites all of them at `t=0`. Template v8
does not choose secondary objects, future ignition times, parent-child links,
or routes. There is no arbitrary-object fallback: if a scene lacks the
required initial semantic objects, planning fails explicitly instead of
silently substituting an unrelated object.
`TEMPLATE_CATEGORY_GROUPS` is regression-checked against all 36 installed
HM3D `semantic.txt` files and `MATERIAL_TABLE`.

| Template | Initial semantic categories (fallback) | Initial placement policy |
| --- | --- | --- |
| `kitchen_grease_fire` | `stove`, `stovetop`, `oven and stove`, `oven`, `cooker` (→ exact kitchen appliances/hoods) | flammability-weighted exact count |
| `bedroom_textile` | `bed`, `bed small`, `bedframe`, `pillow`, `blanket`, `bed sheet` (→ exact bedroom textiles/furniture) | flammability-weighted exact count |
| `living_room_electric` | `tv`, `led tv`, `wall tv`, `monitor`, `computer`, tower/laptop variants (→ exact AV electronics) | flammability-weighted exact count |
| `multi_origin` | low floor furniture: beds, sofas/seating, tables/desks, rugs/carpets and ottomans | two or more spatially distributed initial nodes on one floor |

**Intensity presets** (`INTENSITIES`, three tiers):

| intensity | initial sources (min–max) | source_temp_c | fuel_kg | duration_s |
| --- | --- | --- | --- | --- |
| light | 1–1 | 550 | 2 | 300 |
| medium | 1–2 | 750 | 5 | 600 |
| severe | 2–3 | 950 | 10 | 900 |

Initial-only generation is the only planner mode. When `--num_ignitions` is
omitted, the intensity row above supplies the initial-source count. Passing
`--num_ignitions N` overrides that range with an **exact initial-source
count**. The plan then contains exactly N distinct entries, all with
`ignite_time_s = 0` and `ignition_role = "initial"`. Which other furniture
objects ignite later is decided inside `FirePropagation.step` from fuel,
temperature, conduction and radiation.

An explicit request fails instead of silently reducing the scenario when
there are fewer than N initial candidates. `multi_origin` requires `N >= 2`.
All plans store the resolved count as `num_initial_ignitions` plus
`ignition_selection_mode = "initial_only"` and
`ignition_selection_version`; an explicit override additionally stores
`num_initial_ignitions_requested` and participates in `plan_id`.
`TEMPLATE_VERSION = 10` gives vertically bounded plans new IDs, so their
timelines cannot be confused with template-v9. V10 retains the enlarged
duration-aware radial floor hazard from v9 and adds metric limits for both
free flame columns and vertical object-BBox visual fill.

**Plan schema v3 (`plans/<plan_id>.json`).** Top level: `scene_id`,
`world_aabb`, `fire_type`, `intensity`, `seed`, `template_version`,
`duration_s`, `num_initial_ignitions`, `ignition_selection_mode`, and
`ignition_selection_version`, plus
optional `num_initial_ignitions_requested`. Each `ignitions[i]`:
`object_id`, `category`, `position`,
`ignite_time_s`, `source_radius_m`, `source_temp_c`, `fuel_kg`,
`smoke_yield`, and `ignition_role = "initial"`. There are no secondary,
parent or delay fields. Plus a global
`propagation_rules` block
(`_default_propagation_rules`) whose defaults are:

- `flammable_threshold = 0.4`, `ignition_temp_c = 200` (lowered from the
  ~350 °C literature autoignition value so the discrete grid reaches it
  via radiative pre-heating; documented inline in `templates.py`),
- `spread_speed_m_per_s = 0.12` (light ×0.7, severe ×1.5),
  `spread_kernel = "laplacian"`,
- `ceiling_jet_speed_m_per_s = 0.30`, `buoyancy_v_m_per_s = 0.5`,
  `thermal_diffusivity = 0.05`, `ambient_temp_c = 25`,
- `radiative_gain_c = 200`, `radiative_radius_cells = 4` (severe gain 300;
  enough to bridge a short furniture gap without heating an entire room),
- `floor_thermal_attenuation = 0.20` (semi-transparent floors, NFPA 921
  §5.10 rationale), `flame_through_floors = 1`, floor-ignition and
  fuel-abundance knobs,
- floor flame uses `floor_fuel_value = 0.52`, a two-cell contact radius,
  heterogeneous seed intensity 0.10–0.58, and intensity-dependent bounded
  envelopes: light `0.0033 m/s minimum, 1.43 m`; medium
  `0.0044 m/s minimum, 2.20 m`; severe
  `0.0055 m/s minimum, 2.97 m`. `floor_spread_reach_fraction = 0.90`
  raises a source's constant effective speed only when the preset speed would
  fail to reach that radius within 90% of its remaining simulation duration.
- solver-driven object ignition uses a slightly wider dynamic local domain:
  light `0.00495 m/s, 1.188 m`; medium `0.0066 m/s, 1.35 m`; severe
  `0.00825 m/s, 1.512 m`. This domain masks reaction space but never names an
  object ID; actual ignition still requires the object's fuel voxel to cross
  `ignition_temp_c`.
- after that physical ignition, a deterministic six-connected front fills
  voxels carrying the same `object_id_field` value at light/medium/severe
  speeds `0.006/0.008/0.010 m/s`. The dynamic domain gates first ignition;
  after ignition the front may fill the lower part of that object's exact
  mask, but cannot cross into a neighbouring object's Bounding Box or ignite
  a new object outside the domain. To keep tall AABBs from carrying visible
  flame to their top, `object_bbox_max_vertical_spread_m` limits the fill
  above the object's lowest occupied voxel to `0.55/0.75/0.90 m` for
  light/medium/severe. Temperature and smoke transport remain unrestricted.
- `limit_flame_to_source_envelope = 1` clips synthetic floor and air flame
  to those growing XZ envelopes. Fuel-supported inventory object voxels are
  exempt, so furniture ignited by the solver remains visible outside an
  initial envelope. Temperature and smoke remain unconstrained.
- `max_flame_column_height_m = 0.20/0.30/0.35` for
  light/medium/severe is a metric hard cap on the visual plume. The actual
  number of layers is `min(flame_column_cells,
  floor(max_flame_column_height_m / voxel_m))`, so the cap cannot be exceeded
  at a different bake resolution. `object_flame_extra_height_cells = 0`
  removes the former resolution-dependent extra object layer.
- radial floor intensity follows
  `G_r(d,t)=exp(-d²/(2·(0.58·r(t))²))` inside the bounded source radius.
- `floor_min_visible_flame = 0.06` keeps the dim radial edge above the
  renderer threshold; no planner-authored directional link is added.

**Paper framing.** (a) deterministic hashing ⇒ full reproducibility;
(b) templates map onto recognisable real-world fire classes;
(c) `plan.json` is human-readable and hand-editable (a user can inject
an extra ignition), which supports controlled stress-test experiments.

---

## 3. Stage 3 — Propagation: 3D voxel reaction–diffusion–buoyancy solver

**Goal.** Integrate a plan forward for `duration_s` and emit a per-frame
`(flame, smoke, temperature)` voxel timeline. Code:
`utils/fire_world/propagation.py` (engine) and `voxel_world.py` (state).

**Scope statement (important for the paper).** This is a
**simulation-grade**, not research-grade CFD, solver. The goal is not to
predict a real fireground but to provide a *physically plausible,
smooth, deterministic, and cheap* space–time hazard that navigation
algorithms must contend with. The docstring in `propagation.py` states
this explicitly.

**Per-voxel state.** `fuel ∈ [0,1]` (dimensionless fuel-mass fraction,
rasterised from inventory flammability), `temp` (°C), `flame ∈ [0,1]`,
`smoke ∈ [0,1]`.

**Single step** (`FirePropagation.step`, default `dt = 0.5 s`). The
update is richer than a textbook reaction–diffusion; the actual ordered
stages are:

1. **Heat diffusion** — CFL-substepped 6-point Laplacian,
   `T ← T + α·Δt·∇²T`. Walls/ceilings are zero-flux; floors are
   *semi-transparent* (the Laplacian is attenuated by
   `floor_thermal_attenuation` across floor voxels rather than zeroed),
   giving slow inter-storey heat transfer.
2. **Smoke self-diffusion** — an independent smoke Laplacian with its own
   coefficient `smoke_diffusivity` and CFL factor `smoke_cfl`, so the
   plume actually fills the room instead of staying pinned to the hot mask.
3. **Vertical buoyancy** — the supra-ambient temperature field and the
   smoke field are advected upward by `frac = clip(v_buoy·dt/v, 0, 0.95)`;
   smoke uses a reduced fraction `smoke_buoy_fraction` (denser-than-air
   once cooled). Mass is reflected at the ceiling.
4. **Ceiling jet** — a 3×3 XZ box blur applied to the top voxel layers
   (`_ceiling_jet`), approximating horizontal advection under the ceiling.
5. **Reaction** — where `fuel > flammable_threshold` and `T > T_ignite`,
   fuel is consumed and flame/heat/smoke are produced. The local burn
   rate and flame cap are modulated by a **fuel-abundance** term (a
   box-blurred fuel neighbourhood), so a cushion-dense corner burns
   brighter than an isolated chair. Surface flame spread is a
   Laplacian (or Gaussian) operator masked by `𝟙(fuel>0 ∨ T>T_ignite)`.
6. **Continuous bounded floor ignition** — floor voxels can ignite by **direct flame
   contact** (a flame voxel within `floor_ignite_radius_cells`) or by
   **sustained heating** (`T > floor_ignite_temp_c`) only inside the union
   of source-centred envelopes. For source `i`, let
   `R_i = floor_max_spread_radius_m × floor_spread_scale_i` and
   `v_i = max(v_preset,
   (R_i-source_radius_i)/(floor_spread_reach_fraction ×
   (duration_s-ignite_time_i)))`. The front is then
   `r_i(t) = min(R_i, source_radius_i + v_i × source_age_i)`.
   On ignition floor voxels receive
   synthetic fuel `floor_fuel_value` and a per-voxel randomised seed
   flame. The larger light/medium/severe caps are 1.43/2.20/2.97 m; a
   constant per-source speed reaches each cap before the source's remaining
   duration expires. The hard radius still prevents recursive room-wide
   spread. With `limit_flame_to_source_envelope = 1`, synthetic floor/air flame
   outside the envelope is cleared after surface spread and flame-column
   projection. Fuel-supported object voxels inside the wider dynamic object
   domain are exempt, so a real object heated above the ignition threshold
   remains visibly burning. Smoke and temperature transport are intentionally
   not radius-limited.
   Within the radius, flame magnitude follows the radial Gaussian above, so
   the centre stays bright while the moving edge fades smoothly. No object or
   route is preselected as the destination of spread.
7. **Radiative pre-heating** — flame voxels heat fuel within
   `radiative_radius_cells` by `radiative_gain_c·dt·(blurred flame)`,
   bridging short air gaps while keeping pre-heating local.
8. **Decay + cooling** — flame relaxes once fuel is exhausted; smoke
   decays first-order at `smoke_decay_per_s` (default ≈ 860 s half-life);
   temperature relaxes exponentially toward ambient.
9. **Sustained sources** — each unexpired ignition pins its spherical
   neighbourhood to `≥ ambient + (T_src − ambient)·falloff` and injects
   smoke at a rate proportional to its `smoke_yield`. With
   `inextinguishable_sources = 1` (default) sources never expire. The visible
   fire stays local because of the source envelope, while the resulting smoke
   and temperature field may still affect the wider room.

A visualization-oriented **flame-column** cue has both a legacy cell-count
upper bound (two cells for light, three for medium/severe) and the v10 metric
caps `0.20/0.30/0.35 m`, with geometric decay of 0.45–0.55. At the default
0.15 m resolution this yields one/two/two vertical layers
(`0.15/0.30/0.30 m`). Unsupported column voxels are cleared before the next
physical solver step, so columns cannot stack toward the ceiling or amplify
radiation; they are regenerated from the current burning surface for each
saved frame. After projection, buoyancy-transported visible flame above the
same effective limit is clipped as well; heat and smoke are not
height-clipped.

**Output.** `timeline.npz` holds `flame / smoke / temp` as compressed
fp16 volumes plus `times` (fp32 seconds) and a `meta_json` unicode array;
a sibling `timeline_meta.json` sidecar carries the same metadata so the
loader can bypass numpy's pickle path across version boundaries. Default
propagation grid is `voxel_m = 0.15` (coarser than the 0.10 m inventory
grid, for speed). Metadata records both configured and effective flame-column
height as well as the object-BBox vertical limit, so baked assets remain
auditable.

**Paper framing.** Cite NIST FDS / OpenFOAM as the full-physics
reference and position this solver as the deliberately-simplified
reaction–diffusion + buoyancy + ceiling-jet approximation that produces
occupant-plausible plume evolution at a few tens of seconds of wall-clock
per 900 s scene. A log-time × room spread curve makes a good figure: it
shows the hazard reaches different rooms with a genuine time gradient.

---

## 4. Stage 4 — FireScene: the runtime world-model facade

**Goal.** Wrap `timeline + clock + camera-pose conversion` into one
per-episode object so downstream sensors only need `query(t_sim)` and
`camera_pose(agent_state)`. Code: `utils/fire_world/scene.py`
(`FireScene`, `FireClock`) over `utils/fire_world/runtime.py`
(`FireWorld` data layer).

**Fire-time clock (`FireClock`).** Two modes are supported; the default
is **wallclock** (this changed from the older step-only design):

- **`mode="wallclock"` (default).** Fire time advances with real time,
  scaled by `speedup` (fire-seconds per real-second):
  `t_sim = base_t0 + (now − origin)·speedup`. The fire evolves
  continuously regardless of how fast the agent acts or how long an LLM
  call blocks; `pause()`/`resume()` bracket evaluation stalls, and
  `start()` (called once per episode reset in `main.py`) fixes the
  origin. Default `speedup = 1.0`.
- **`mode="step"` (legacy, for step-count-reproducible benchmarking).**
  `t_sim = ⌊N_step / s⌋·τ + base_t0`, with `s = steps_per_unit` (default
  5) and `τ = seconds_per_unit` (default 2.0). Independent of wall clock.

Beyond the last frame the timeline clamps to its burnt-out state
(`FireWorld.frame_index` clips into range).

**Pose conversion.** `habitat_agent_state_to_cam` turns the Habitat depth
sensor's `(position, quaternion)` into `(cam_pos_world, R_cam2world)`,
with `R` columns = camera right / up / −forward, exactly what the voxel
ray-marcher expects.

**Stale-cache guard.** `FireWorld.load` warns loudly if `plan.json` is
newer than `timeline.npz`, so hand-edited plans are not silently rendered
from an outdated field.

**Paper framing.** Decoupling *when the world evolves* from *when the
robot observes* makes the **fire-to-robot clock ratio** an explicit,
reportable ablation axis (C4). In wallclock mode the ratio is `speedup`;
in step mode it is `τ / s` seconds of fire per action.

---

## 5. Stage 5 — FireSensorSuite: the multi-modal observation model

`FireSensorSuite` (`utils/fire_sensors/suite.py`) observes a bound
`FireScene` from the agent's viewpoint and produces every modality. In
the primary runtime path (`main.py`), **RGB and Thermal are produced
exclusively by the voxel renderer** when a scene is bound; when no scene
is bound the suite returns a clean-RGB / ambient-thermal passthrough so
depth / radar / LiDAR remain usable. Depth, radar, and LiDAR model their
own smoke degradation from the shared `cfg.smoke` parameters.

### 5.1 Volumetric RGB — `VoxelSmokeSensor` / `voxel_render.py`

Per-pixel front-to-back emission–absorption ray-march. Each ray carries
**two** transmittance accumulators so flame radiance survives dense smoke,
plus separate smoke-scatter and flame-radiance buffers:

- **Scene channel** `T_scene = ∏ exp(−(σ_smoke,visible + σ_flame)·ds)`,
  used for the clean scene RGB and smoke scatter, with
  `σ_smoke = k_ext · smoke_voxel` (`smoke_k_ext`, default 4.0 /m).
- **Flame-local smoke displacement.** A strong flame removes up to `0.52`
  of the co-located smoke extinction and grey scatter. This models hot
  combustion gas displacing cooler soot and prevents orange flame structure
  from being flattened into one grey-white fog bank.
- **Flame channel** `T_flame = ∏ exp(−((1−p)·σ_smoke + σ_flame)·ds)`,
  where `p = flame_smoke_passthrough` (default 0.95) is the fraction of
  smoke extinction that flame self-emission ignores — consistent with
  the visible-band flame-through-smoke imagery in
  [Starr & Lattimer, 2014, Fig. 7].

Flame emission is coloured by a multi-stop LUT (deep red → orange →
yellow → chromatic yellow-white core) indexed by voxel flame intensity;
`flame_threshold = 0.04` lets trilinearly-interpolated edge voxels
participate. The default `flame_emission_gain = 3.2`,
`flame_k_ext = 0.50`, and peak-preserving highlight compression keep long
rays through a burning furniture Bounding Box below display-white without
discarding their red/orange colour ratios. A bounded `0.13` clean-surface
reveal then retains upholstery, wood grain and object edges behind the
translucent emissive volume.

High-quality mode combines 3-D turbulence with a surface-anchored noise
sample. The latter is shared along each ray, so flame bands and broken
edges survive integration instead of averaging back into a smooth blob.
The defaults are `flame_noise_strength = 0.75`,
`flame_edge_break = 1.05`, `flame_color_jitter = 0.32`, and a reduced
`flame_glow_gain = 0.18`. This detail is visualisation-only and does not
change flame voxels, temperature, thermal masks or risk maps. It remains
disabled under `--fire_fast 1`; use `--fire_fast 0` for teleoperation and
qualitative figures.

### 5.2 Smoke-degraded depth — `SmokeDepthSensor`

Following [Starr & Lattimer, 2014, Fig. 5], the clean metric depth is
degraded by: (i) range- and density-dependent Gaussian noise
`σ = σ_base + σ_range·d + σ_smoke·density·d`; (ii) cm-level quantisation;
(iii) density-proportional dropout (`p_drop ≤ dropout_max`); and (iv) a
visibility cutoff at Jin's law `V = 2.3 / k`, `k = density·k_max`, beyond
which pixels are clipped to the smoke layer. Parameters live in
`DepthDegradeConfig` / `SmokeConfig`.

### 5.3 mmWave radar — `RadarSensor`

A `range_bins × az_bins = 256 × 64` range–azimuth heatmap with
sinc-shaped azimuth sidelobes, thresholded (`threshold = 0.18`) to a
point cloud. In `mode="learned"` (default) it bypasses the raw heatmap
and emits a LiDAR-like 3D cloud with `learned_noise_m = 0.10` Gaussian
noise and 4× stride subsampling, tuned to roughly match RadarHD's
reported post-training median Hausdorff error. mmWave is treated as
nearly smoke-invariant. All shared navigation/teleoperation previews
carry physical axes: the BEV uses lateral `Y [m]` horizontally and
forward `X [m]` vertically, while the range–azimuth and range–elevation
plots use angle `[deg]` horizontally and range `[m]` vertically. These
axes decorate only the preview images; the raw heatmap and point-cloud
arrays consumed by navigation are unchanged.

### 5.4 360° LiDAR — `LidarSensor` / `lidar_360.py`

A true 360° cloud is stitched from **four yaw-rotated 90°-HFOV depth
sensors** (`lidar_depth_{front,left,back,right}`) injected into the
Habitat agent config at startup (`install_lidar_depth_sensors`). Pinhole
depth is chosen over an equirectangular sensor because its intrinsics
give a clean back-projection and it is supported by every habitat-sim
build. Smoke-aware noise and dropout mirror the depth sensor.

### 5.5 Thermal IR — voxel temperature channel

Thermal sensing is **surface-dominant**. The final 28% of each clean-depth
ray estimates the visible surface temperature; mean hot air along the
whole ray contributes only `thermal_air_coupling = 0.025`. The old
per-ray maximum was incorrect: one hot plume voxel could assign hundreds
of degrees to an otherwise ambient wall and turn a near-fire frame
uniformly yellow.

The physical `thermal_temperature` map is ambient plus this localized
surface/hot-air estimate, with visible flame raised toward 600 °C. RGB
luminance is never added to the Celsius values. For display only, cold
scene RGB is darkened to retain doors/furniture/room structure, while a
fixed logarithmic heat response progressively blends warm surfaces and
flames toward INFERNO (`thermal_color_blend`, default `0.85`). Fixed
temperature anchors replace frame-percentile auto-stretch, so a fire
cannot repaint the exposure of every ambient pixel. Projected humanoids
then receive a physical warm-body contribution and an explicit
high-contrast silhouette overlay. LWIR (7.5–14 µm) remains treated as
smoke-invariant.

### 5.6 Dashboard

`render_dashboard` composites eight panels (clean RGB, clean depth,
thermal, LiDAR BEV, smoky RGB, smoky depth, radar BEV, radar
range–azimuth) plus a radar range–elevation panel into one image, with
`step` and `t_sim` printed in the title for figure annotation.

---

## 6. Stage 6 — Robot integration and visualisation

`utils/fire_pipeline.step_fire_observation()` glues the suite into the
nav loop each frame:

1. read `obs['rgb'/'depth']` and convert to metric depth;
2. call `suite.process(rgb, depth, agent_state, robot_step)` to get the
   synthesised sensor dict;
3. `apply_clean_depth_and_thermal` writes the results back into `obs`:
   voxel/smoky RGB → `obs['rgb']`; clean or noisy depth → `obs['depth']`
   (governed by `--depth_use_clean`); thermal image + flame mask →
   `obs['thermal*']`.

| Entry point | Purpose |
| --- | --- |
| `main.py` / `main_vec.py` | FireWorld automated evaluation; complete dynamic risk assessment is currently wired only in `main.py` and `main_vec.py` rejects `--risk_enabled=1` |
| `scripts/keyboard_teleop_fire.py` | manual WASD driving with a first-person view + suite dashboard, for qualitative inspection |

---

## 7. Paper-ready method description (adapt directly)

> Drop this into §3 (Methodology) or §4 (Benchmark Construction). Align
> variable names and figure numbers to your paper.

### 7.1 Overview

We introduce **FireWorld**, a benchmark that augments Habitat–HM3D
ObjectGoal Navigation with a deterministic, time-varying 3D fire and
smoke field, paired with a multi-modal sensor-degradation model. The
benchmark is produced by a six-stage pipeline (Fig. X) that decouples a
*world model* of fire propagation from an *observation model* of sensor
degradation, so that each can be ablated independently. Every
intermediate artefact is content-addressed by a SHA1-truncated
`plan_id`, making complete experimental runs bit-for-bit reproducible
from a `(fire_type, intensity, seed)` triple.

### 7.2 Scene parsing and material assignment

Given an HM3D scene we recover a *full* instance-level inventory by
walking each `*.semantic.glb` primitive, decoding its per-vertex linear
`COLOR_0` (uint16) through the IEC 61966-2-1 sRGB OETF, and matching the
resulting 24-bit colour to `*.semantic.txt` instance IDs. Because HM3D
primitives are grouped for view-frustum culling rather than by instance,
a single primitive routinely contains many instance colours; we therefore
aggregate at face granularity. Each instance is annotated with an
axis-aligned bounding box in Habitat world coordinates, a floor index
from 1-D clustering of instance y-extents (1.5 m gap threshold), and two
material parameters — *flammability* and *smoke yield* — from a
category-indexed table. Structural categories (walls, floors, ceilings,
doors, windows, columns) are tagged separately and rasterised into a
0.10 m voxel mask used as zero-flux boundaries by the solver; floor and
ceiling instances are stamped as thin slabs rather than full-height AABBs.

### 7.3 Plan generation

A *plan* selects ignitions and propagation rules from the inventory under
a `(fire_type, intensity, seed)` triple. We expose four template
generators (kitchen-grease, bedroom-textile, living-room-electric,
multi-origin) that select only the initial ignition objects at `t=0`.
Single-type templates use flammability-weighted sampling within their
declared categories; multi-origin uses spatially distributed low furniture
on one floor. Three intensity presets (light / medium / severe) set the source
temperature (550 / 750 / 950 °C), the per-ignition fuel mass
(2 / 5 / 10 kg), and the simulated duration (300 / 600 / 900 s). The
output `plan.json` records each initial object's world position, source
radius, source temperature and smoke yield together with the global
propagation rule set. Later object ignition is an emergent solver result, not
a plan entry. Each plan is content-addressed by a SHA1-truncated hash.

### 7.4 Voxel propagation

Each plan is integrated forward by a custom 3D voxel solver whose per-voxel
state is `(fuel, temperature, flame, smoke)`. One `dt = 0.5 s` step
applies, in order: CFL-substepped Laplacian heat diffusion with
semi-transparent floors and zero-flux walls/ceilings; independent smoke
self-diffusion; buoyancy-driven vertical advection of supra-ambient
temperature and smoke; a ceiling-jet box blur under the ceiling; a
reaction step in which voxels above the fuel and ignition-temperature
thresholds consume fuel and emit flame, heat, and smoke, with the burn
rate modulated by local fuel abundance and surface spread modelled as a
masked Laplacian; floor ignition by flame contact or sustained heating;
radiative pre-heating that couples nearby fuel across air gaps; flame and
smoke decay with exponential cooling toward ambient; and a sustained-source
term that pins each active ignition's neighbourhood and injects smoke at a
rate set by its yield. The solver runs at `voxel = 0.15 m` and writes a
compressed `timeline.npz` of fp16 `(flame, smoke, temperature)` volumes
with fire-time stamps; a 900 s scene bakes in a few tens of seconds.

### 7.5 World–observation decoupling at runtime

At evaluation time we do *not* re-run the solver. A `FireScene` object
loads the cached timeline and exposes two operators:
`query(t_sim) → (flame, smoke, temperature)` and
`camera_pose(state) → (position, rotation)`. Fire time is advanced by a
`FireClock`. In the default *wallclock* mode fire time flows with real
time scaled by a `speedup` factor (fire-seconds per real-second), so the
hazard evolves continuously irrespective of agent or planner latency; a
*step* mode maps `t_sim = ⌊N_step / s⌋·τ` for step-count-reproducible
runs. This makes the fire-to-agent time coupling a first-class,
reportable experimental axis.

### 7.6 Sensor model

Every modality is produced by a `FireSensorSuite` that observes the
FireScene from the agent's viewpoint:

* **Volumetric RGB.** A ray-march compositor overlays the flame / smoke
  field on the clean scene RGB. To preserve flame radiance through dense
  smoke it maintains two transmittance accumulators — a scene channel
  `T_scene = ∏ exp(−(σ_smoke + σ_flame)·ds)` and a flame channel
  `T_flame = ∏ exp(−((1−p)·σ_smoke + σ_flame)·ds)` with `p = 0.95` — the
  passthrough coefficient being consistent with visible-band imagery in
  [Starr & Lattimer, 2014, Fig. 7]. Flame is coloured by a multi-stop LUT.
* **Smoke-degraded depth.** Following [Starr & Lattimer, 2014, Fig. 5],
  depth is degraded by range- and density-dependent Gaussian noise,
  cm-level quantisation, density-proportional dropout, and a visibility
  cutoff at Jin's law `V = 2.3 / k`.
* **Thermal IR.** Visible/near-surface voxel temperature dominates each
  clean-depth ray; line-of-sight hot air has weak coupling, preventing a
  single plume voxel from saturating a background wall. Cold structure is
  dark RGB context and localized heat uses a fixed logarithmic
  grayscale/INFERNO response. LWIR (7.5–14 µm) is treated as
  smoke-invariant.
* **mmWave radar / LiDAR.** A 256×64 range–azimuth heatmap with
  sinc-shaped sidelobes, optionally converted to a 3D point cloud
  matching RadarHD's reported error; LiDAR stitches four yaw-rotated
  depth sensors into a 360° cloud with smoke-aware noise and dropout.

An eight-panel dashboard composites all modalities for offline
inspection.

### 7.7 Determinism, reproducibility, and configurability

Every artefact is deterministic given its inputs: `inventory.json` is a
function of HM3D content; `plan.json` of `(fire_type, intensity, seed)`;
`timeline.npz` of `(plan, voxel_m, dt)`; and per-step observations of
`(timeline, agent_state, step_index)` in step mode. Hand-edited plans are
supported, and a stale-cache detector warns when the propagation step must
be re-run. The pipeline is configured through CLI flags in
`arguments.py`, grouped into fire-world selection (`--fire_world`,
`--fire_world_plan_id`), fire-time coupling (`--fire_clock_mode`,
`--fire_speedup`, `--fire_steps_per_unit`, `--fire_seconds_per_unit`),
volumetric rendering (`--fire_world_n_steps`, `--fire_world_render_scale`,
`--fire_world_smoke_k_ext`, `--fire_fast`, `--fire_flame_noise`), and
perception (`--depth_use_clean`, `--use_thermal_perception`,
`--smoke_density`).

### 7.8 Suggested ablations (paper §5)

| Axis | Lever | Measures |
| --- | --- | --- |
| Fire-to-agent time coupling | `--fire_speedup` (wallclock) or `--fire_steps_per_unit`/`--fire_seconds_per_unit` (step) | how aggressively the fire grows relative to the agent's action rate |
| Depth path | `--depth_use_clean` 0/1 | navigation cost added by smoke-degraded depth |
| Thermal perception | `--use_thermal_perception` 0/1 | whether LWIR rescues fire detection in dense smoke |
| Smoke density | `--smoke_density` | sensitivity of each modality to optical thickness |
| Flame radiance model | `flame_smoke_passthrough` (0–1, `VoxelSmokeConfig`) | detection sensitivity to the physically-grounded flame-through-smoke model |
| Render fidelity | `--fire_fast` 0/1, `--fire_world_n_steps` | whether ray-march fidelity affects the agent (vs. render cost only) |

---

## Appendix A — File map

| Path | Role |
| --- | --- |
| `utils/fire_world/scene_scan.py` | Stage 1: build `inventory.json` + structural voxels |
| `utils/fire_world/hm3d_semantic.py` | Stage 1: sRGB OETF + GLB parsing helpers |
| `utils/fire_world/templates.py` | Stage 2: ignition templates + intensity presets + default rules |
| `utils/fire_world/planner.py` | Stage 2: CLI + `plan_id` hashing |
| `utils/fire_world/propagation.py` | Stage 3: voxel solver |
| `utils/fire_world/voxel_world.py` | Stage 3: `VoxelWorld` state container |
| `utils/fire_world/runtime.py` | Stage 4: `FireWorld` timeline loader (+ stale-cache warning) |
| `utils/fire_world/scene.py` | Stage 4: `FireScene` + `FireClock` |
| `utils/fire_sensors/voxel_render.py` | Stage 5: ray-march + flame LUT + dual-transmittance |
| `utils/fire_sensors/sensors/voxel_smoke.py` | Stage 5: voxel RGB/Thermal sensor |
| `utils/fire_sensors/sensors/{depth_smoke,radar,lidar}.py` | Stage 5: per-modality degradation |
| `utils/fire_sensors/lidar_360.py` | Stage 5: 4-sensor 360° LiDAR injection |
| `utils/fire_sensors/config.py` | Stage 5: all sensor knobs (`FireSensorConfig` + nested) |
| `utils/fire_sensors/suite.py` | Stage 5: `FireSensorSuite` orchestration |
| `utils/fire_pipeline.py` | Stage 6: `step_fire_observation` glue |
| `main.py` / `main_vec.py` | Stage 6 FireWorld evaluation; risk runtime only in `main.py` |
| `scripts/keyboard_teleop_fire.py` | Stage 6: manual driving + dashboard |
| `arguments.py` | all CLI flags |
| `docs/main_usage.md` | flag + keyboard reference |

## Appendix B — Minimal reproduction of a new plan

```bash
# 1) Build the inventory (once per scene)
python -m utils.fire_world.scene_scan --scene Nfvxx8J5NCo

# 2) Auto-generate a plan (or hand-write plans/<id>.json)
python -m utils.fire_world.planner \
    --scene Nfvxx8J5NCo \
    --fire_type multi_origin --intensity medium --seed 7 \
    --num_ignitions 4

# N means exactly four initial sources at t=0. The planner does not select
# any later object; propagation decides that from the voxel heat/fuel fields.
# The planner prints the new count-specific plan_id. Inspect without writing:
python -m utils.fire_world.planner \
    --scene Nfvxx8J5NCo \
    --fire_type multi_origin --intensity medium --seed 7 \
    --num_ignitions 4 --print_only

# 3) Run propagation with the plan_id printed by step 2
# (6df964ec1f4c for the exact template-v8 inputs above)
python -m utils.fire_world.propagation \
    --scene Nfvxx8J5NCo --plan_id 6df964ec1f4c \
    --voxel_m 0.15

# 4) Automated evaluation (wallclock clock, real-time fire)
python main.py --num_agents 2 --nav_mode co_ut \
    --fire_world 1 --fire_world_plan_id 6df964ec1f4c \
    --fire_clock_mode wallclock --fire_speedup 1.0 \
    --depth_use_clean 1

# 4') Or drive manually to inspect the field (step clock for reproducibility)
python scripts/keyboard_teleop_fire.py \
    --task-config configs/multi_objectnav_hm3d.yaml \
    --scene-id Nfvxx8J5NCo --plan-id 6df964ec1f4c \
    --clock-mode step --steps-per-unit 1 --seconds-per-unit 5.0 \
    --depth_use_clean 1 --show-dashboard 1
```

## Appendix C — Preparing complete FireWorld asset matrices

`scripts/prepare_fire_world_scene.py` runs scene scan, deterministic planning,
propagation, timeline validation and checksumming as one resumable command.
By default it prepares all four templates at all three intensities:

```bash
python scripts/prepare_fire_world_scene.py \
    --scene Nfvxx8J5NCo \
    --fire-types all --intensities all --seeds 42
```

Use comma-separated selectors to prepare a subset. `--dry-run` performs
semantic planning and size estimation without writing; the default resume
mode validates an existing timeline header, times and metadata before
skipping it. `--force` explicitly rebakes valid assets.

The dataset coordinator uses the same runner and canonical runtime paths:

```bash
# Read-only task and storage preflight
python scripts/prepare_fire_world_dataset.py \
    --dataset-root data/scene_datasets/hm3d_v0.2 --splits val \
    --fire-types all --intensities all --seeds 42 \
    --jobs 1 --dry-run

# Production run; automatically resumes validated assets
python scripts/prepare_fire_world_dataset.py \
    --dataset-root data/scene_datasets/hm3d_v0.2 --splits val \
    --fire-types all --intensities all --seeds 42 \
    --jobs 2
```

Only folders containing matching `basis.glb`, `semantic.glb` and
`semantic.txt` files are executable. Incomplete dataset folders and
template/category shortages are recorded as explicit skip statuses rather
than silently substituted. Before propagation, the coordinator computes an
uncompressed upper bound for every missing timeline, checks free disk, and
caps requested workers using the largest estimated process memory.

Canonical assets remain:

```text
scenes/<scene>/inventory.json
scenes/<scene>/structural/{walls,floors,ceilings}.npy
scenes/<scene>/plans/<plan_id>.json
outputs/fire_world/<scene>/<plan_id>/timeline.npz
outputs/fire_world/<scene>/<plan_id>/timeline_meta.json
```

Operational records are stored separately:

```text
outputs/fire_world/<scene>/asset_index.json
outputs/fire_world/runs/<run_id>/run_config.json
outputs/fire_world/runs/<run_id>/preflight.json
outputs/fire_world/runs/<run_id>/resource_plan.json
outputs/fire_world/runs/<run_id>/manifest.jsonl
outputs/fire_world/runs/<run_id>/summary.json
outputs/fire_world/runs/<run_id>/logs/<scene>/<scenario>.log
```

New timelines are baked under `outputs/fire_world/.staging/`, validated, and
then atomically installed. A replaced directory is retained under
`.replaced/`; lock files under `.locks/` prevent duplicate concurrent work.
