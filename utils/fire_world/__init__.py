"""Fire-world subsystem.

This package owns the offline pipeline that turns a Habitat scene into a
deterministic, time-evolving fire/smoke environment, and the runtime
surface that a navigation agent queries against. It is intentionally
decoupled from ``utils.fire_sensors``: that module remains the rendering
backend (smoke RGB, thermal, depth degradation), while this module is
responsible for *what is happening in the world* (where flames are, how
dense the smoke is, in 3D, at time t).

Layout (kept stable across PRs):

  scene_scan.py    -> inventory.json
  material_table   -> per-category flammability + smoke yield
  planner.py       -> plan.json (LLM-generated, hash-cached)
  voxel_world.py   -> voxel grid construction
  propagation.py   -> reaction-diffusion + buoyancy + ceiling jet
  runtime.py       -> FireWorld.load(scene_id, plan_id) for the agent loop
"""
