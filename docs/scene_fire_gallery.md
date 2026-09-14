# Scene-grouped medium multi-origin gallery

`scripts/capture_scene_fire_gallery.py` creates one merged plan and one baked
timeline for each of the 36 scenes in the formal PersonNav validation split.
It selects four navigable camera views per scene and exports actual Habitat
images and FireSensor observations.

## Plan and timeline provenance

The input set is the canonical medium kitchen, textile, electrical and
multi-origin plans in `scenes/<scene_id>/plans/`. Initial fire sources are
merged by `object_id`. On overlap, the existing multi-origin source parameters
win; otherwise the original template's ignition entry is retained. All sources
ignite at time zero. The existing multi-origin medium propagation rules,
source lifetimes and 600-second duration remain unchanged.

These merged plans use a **custom gallery source distribution**, not the
canonical multi-origin 4–6-region / 1–2-source sampling contract. Their source
count is the union of the input sets. Every plan records source membership,
base plan IDs and unavailable semantic templates. An unavailable type gets a
view of another actual burning area, explicitly labelled as an additional
origin. No kitchen or electrical object is invented.

All derived plans are saved under the output directory; canonical plan JSONs
and existing benchmark timelines are not rewritten. `inputs.json` records
SHA-256 hashes of the original plans and inventories. Each timeline also has
a `provenance.json` recording the solver hash and bake settings.

The bake runs the original solver for 600 seconds at `dt=0.5 s`, using
`voxel_m=0.15`. It saves every 10 seconds (61 frames). This is a gallery
sampling interval, not the navigation benchmark's usual 1-second cache.
All selected capture times coincide with saved frames.

## Rendering

- Native Habitat textured geometry, RGB, metric depth and semantic images.
- Camera: 960×720 pixels, 79-degree horizontal FOV, 1.3 m above the navigable
  agent position, pitched toward the selected semantic source; depth range
  12 m. The four LiDAR depth cameras remain horizontal and cover 360 degrees.
- Native FireSensorSuite: RGB with volumetric fire/smoke, thermal, clean and
  degraded depth, LiDAR, and radar BEV/range-angle views. The gallery camera
  differs from the normal navigation camera; its actual pose is recorded.
- Final Torch/CUDA ray marching at full resolution with 96 samples. Preview
  search uses 24 samples and 240×180 integration. After the initial five scenes,
  native camera-search frames use 320×240 to accelerate semantic visibility
  checks; final captures remain native 960×720 in all scenes.
- Source visibility in the native semantic image rejects occluded views.
  Fire visibility and framing choose among candidate poses and simulation
  times. Raised sources are anchored to navigable floors below them, avoiding
  an upstairs camera for a downstairs wall-mounted TV. Additional origins
  prefer regions not already represented; if that region's only source is
  occluded, another visible source is used and the relaxed region preference
  is recorded in the preview metrics.
- Current version 3 uses smoke extinction 3.0, flame emission gain 8.0 and
  flame threshold 0.015. Version 2 used 0.75, 3.2 and 0.04 respectively.
  The lower threshold reveals more of the existing flame envelope; increased
  emission and smoke extinction make flame contours and smoke layers visible.
  This is an optical rendering change, not increased physical spread or heat.
  Procedural flame noise 0.9, edge break 1.15, color jitter 0.28, glow kernel 13,
  glow gain 0.18, surface reveal 0.08, highlight compression 0.85; smoke noise
  0.24. The shared depth/LiDAR/radar sensor smoke-density
  setting remains 0.6. These are recorded gallery rendering parameters.

The requested gray haze is a depth-aware **RGB presentation layer**. It uses
gray `(140,140,140)`, density `0.045/m`, and maximum opacity 0.22. A smooth gate
suppresses it where physical smoke is already strong and around visible
flames. It does not change the fire fields, physical transmittance,
temperature or depth/radar/LiDAR outputs. Both the unstyled RGB and the haze
alpha are retained. The dashboard RGB and exported styled RGB are identical.

## Commands

Run from the repository root with the `co-nav3` environment:

```bash
PY=/home/liushe10/miniconda3/envs/co-nav3/bin/python
$PY scripts/capture_scene_fire_gallery.py --stage prepare
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY scripts/capture_scene_fire_gallery.py --stage bake --workers 2
HABITAT_SIM_LOG=error MAGNUM_LOG=quiet OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $PY scripts/capture_scene_fire_gallery.py --stage render
$PY scripts/capture_scene_fire_gallery.py --stage index
$PY scripts/capture_scene_fire_gallery.py --stage validate
```

`--scene SCENE_ID` restricts baking/rendering to selected scenes; repeat the
option to select several. `--wait-for-timelines 1800` lets a render process use
ready scenes first while waiting for concurrent bakes. `--force-render`
regenerates selected views while retaining their single baked timeline.
`--output-dir PATH` overrides the default
`outputs/fire_gallery_medium_36_20260911`.

GPU rendering needs access to the host NVIDIA devices. Device-isolating
sandboxes may report no CUDA even though the host driver is operational.

## Stronger matched revision

The original version 2 gallery is retained in
`outputs/fire_gallery_medium_36_20260911`. The stronger version 3 is in
`outputs/fire_gallery_medium_36_stronger_20260912`, with a `timelines` symlink
to the original 36 bakes, so no duplicate bake storage is needed. Keep the
original directory when moving or archiving the stronger gallery.

The revised gallery uses exactly the original four source IDs, camera poses
and simulation times per scene. All sensor observations are processed again;
the suite's voxel optical changes affect RGB/transmittance and its coupled
thermal render. The medium hazard fields and shared depth/LiDAR/radar model
parameters remain unchanged. Do not interpret increased visual flame area
as a larger physically burning floor area. `sensor_config.json` records the
complete revised configuration and every view records its reference gallery.

```bash
HABITAT_SIM_LOG=error MAGNUM_LOG=quiet OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/liushe10/miniconda3/envs/co-nav3/bin/python scripts/capture_scene_fire_gallery.py \
  --stage render --output-dir outputs/fire_gallery_medium_36_stronger_20260912 \
  --reuse-views-from outputs/fire_gallery_medium_36_20260911
```

Compare the resulting observations with:

```bash
/home/liushe10/miniconda3/envs/co-nav3/bin/python scripts/compare_scene_fire_galleries.py \
  --reference outputs/fire_gallery_medium_36_20260911 \
  --revised outputs/fire_gallery_medium_36_stronger_20260912
```

`comparison.html` pairs all 144 before/after images. `comparison.json` checks
source IDs, poses, timestamps, clean RGB/depth and shared timeline identity,
and records per-view smoke attenuation and visible flame area. Add
`--completed-only` during rendering for explicitly partial reports.

The render command uses the prepared `inputs.json`, copied scene plan files and
linked timelines in the revision directory. `--force-render` also replaces
already completed captures; without it, matching completed versions resume.

## Products

```text
outputs/fire_gallery_medium_36_20260911/
  index.html                       # Browse all scenes
  inputs.json                      # Original input paths and hashes
  validation.json
  <scene_id>/
    index.html                     # Four full-size views and sensor links
    plan.json                      # One shared merged medium plan
    manifest.json                  # Four camera poses, times and sources
    four_templates.png             # 2×2 RGB overview
    four_sensor_observations.jpg    # 2×2 sensor dashboard overview
    timeline/                      # Link to this scene's single bake
    01_kitchen_grease_fire/         # Other slots have analogous directories
      template_scene.png
      fire_sensor_observation.png
      fire_sensor_arrays.npz
      rgb_clean.png
      rgb_fire_smoke.png            # Same pixels as template_scene.png
      rgb_fire_smoke_physical.png   # Before presentation haze
      presentation_haze.npz         # Alpha and unstyled RGB
      source_object_mask.png       # Native selected-source semantic mask
      depth_clean.png
      depth_smoke.png
      thermal.png
      lidar_bev.png
      radar_bev.png
      radar_range_azimuth.png
      radar_range_elevation.png
      sensor_dashboard.png
      manifest.json
  timelines/<scene_id>/<plan_id>/
    timeline.npz
    timeline_meta.json
    provenance.json
```

The inherited category directory names denote requested slots. The per-view
`label`, `semantic_fallback` and `selected_source` fields specify what is
actually depicted. Read the metadata when a scene lacks a requested category.
