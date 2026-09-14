# Temporal Evolution of Fire Scenarios

`scripts/capture_fire_temporal_evolution.py` exports twelve real Habitat /
FireWorld videos: three examples each for kitchen grease fire, bedroom textile
fire, living-room electrical fire and multi-origin fire.

The selected scenes are `4ok3usBNeis`, `6s7QHgap2fW` and `BAbdmeyTvMZ`.
All three have genuine source objects for the first three template views,
without semantic fallback. Their plans are the same merged medium plans used
by `outputs/fire_gallery_medium_36_stronger_20260912`. These are views of
shared merged plans, not isolated single-template propagation experiments.

## Time and camera semantics

- One dense timeline per scene, shared by its four camera views. The original
  plan, inventory, 0.15 m voxels, 0.5 s solver step and 600 s duration are kept.
- Snapshots are saved every 2 s: 301 actual simulation states from 0 to 600 s.
  Every common 10 s state is checked for exact flame/smoke/temperature equality
  with the previous gallery bake. No fire-field interpolation is used.
- Sources ignite at t=0 according to the existing solver. Each video begins
  with a labelled 2 s clean reference, which is not a simulated negative-time
  state; the final 600 s state also has a 2 s hold.
- The 301 evolution frames play at 5 FPS: 10x simulation time, 60.2 s of
  evolution, 64.2 s including both holds. Each video has 321 decoded frames.
- Each camera stays fixed at the existing gallery pose. Native RGB/depth
  cameras use 800x600, 79 degree HFOV, 1.3 m height, 12 m depth range. RGB fire
  and smoke come from the suite's actual VoxelSmokeSensor with 64 ray samples,
  0.5 render scale (400x300 volumetric integration over native 800x600
  scene texture), Torch/CUDA FP16, and the stronger gallery optical preset.
  The subtle gray presentation haze is retained. This does not change medium
  propagation. The job does not regenerate unused per-frame radar/LiDAR plots.
- The first three template videos show one source camera. Each multi-origin
  video synchronizes all four source cameras from the same plan, making
  simultaneous fires in different regions visible in one video.

The process is the actual model evolution. Later smoke may obscure some
objects, and a sustained source can remain burning at 600 s; the videos do
not invent an extinction stage or adjust times to manufacture extra spread.

## Reproduce

Run from the project root with the firenav environment and GPU access:

```bash
HABITAT_SIM_LOG=error MAGNUM_LOG=quiet OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python scripts/capture_fire_temporal_evolution.py \
  --stage all
```

Stages can also run separately: `prepare`, `bake`, `pilot`, `render`, `encode`,
`validate`. `--scene` restricts bake/render/encode, and `--workers` controls
CPU bake workers. `--template` restricts encoding to selected templates. Run full `validate` after all selected scenes are encoded.
Pilot captures representative times but does not mark a full render complete.

## Outputs and validation

### Fifteen-second revision

`outputs/fire_temporal_evolution_medium_15s_20260912` contains the shortened
version of all twelve clips. It reuses the original scene frames and dense
timelines through relative symlinks. All 301 actual evolution frames remain:
25 FPS gives 50x simulation speed, with 37 clean-reference frames and 37
final-hold frames. Thus each clip has 375 frames and is exactly 15 seconds.
Each hold lasts 1.48 s. Labels and per-video metadata use the revised speed.

With `inputs.json` and the `scenes` / `timelines` links prepared in that folder:

```bash
python scripts/capture_fire_temporal_evolution.py \
  --stage encode --output-dir outputs/fire_temporal_evolution_medium_15s_20260912 \
  --fps 25 --hold-frames 37
python scripts/capture_fire_temporal_evolution.py \
  --stage validate --output-dir outputs/fire_temporal_evolution_medium_15s_20260912
```

Validation reads each video's own FPS and duration from its manifest. The
original 64.2-second clips are retained in the original output directory.

Output root: `outputs/fire_temporal_evolution_medium_20260912`.

- `index.html`: videos grouped by template, with download and stage links.
- `<template>/<scene>/temporal_evolution.mp4`: H.264, yuv420p, faststart.
- `<template>/<scene>/temporal_stages.jpg`: 0, 10, 30, 60, 180, 600 s summary.
- `<template>/<scene>/manifest.json`: exact frame count, timing, source camera
  slots, plan hash, encoder arguments, and video hash.
- `scenes/<scene>/view_*/`: clean references and numbered rendered frames.
- `scenes/<scene>/render_manifest.json`: camera provenance and per-frame
  smoke/flame visibility diagnostics and image hashes.
- `timelines/`: three dense shared bakes and exact-reference-state validation.
- `sensor_config.json`, `inputs.json`, `validation.json`: configuration,
  selected inputs and complete video decode results.

Validation decodes every frame with OpenCV, checks frame count, FPS and
dimensions, verifies temporal changes, and performs a full FFmpeg error-mode
decode. These checks establish media integrity; plan and common-time field
comparisons establish the relationship to the original medium simulation.
