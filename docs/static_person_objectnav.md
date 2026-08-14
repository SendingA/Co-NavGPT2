# Static Person ObjectNav benchmark

This benchmark treats a fixed humanoid as the seventh HM3D ObjectNav category:
person has task category ID 6, after the six standard HM3D categories.
Person episodes use the same ObjectGoalSensor, detector matching, STOP action,
DistanceToGoal, Success, and SPL path as chair/bed/plant/toilet/tv/sofa.

## Generate the dataset

Activate the Habitat environment used for main.py, then generate the default
validation split used by ``configs/person_objectnav_hm3d.yaml``:

    python scripts/build_person_objectnav_dataset.py --split val

The generator reads the matching split below
data/datasets/objectnav_hm3d_v2, loads each scene navmesh, chooses a
deterministic navigable humanoid position, and samples surrounding navigable
view points. Source data is not modified; output is written to:

    data/datasets/objectnav_hm3d_person_v1/<split>/
      <split>.json.gz
      content/<scene>.json.gz

The generator explicitly supplies Habitat-Sim with the same navmesh settings
used by the active HM3D ObjectNav runtime: agent radius 0.18 m, height 0.88 m,
maximum climb 0.20 m, maximum slope 45 degrees, and no static objects baked
into the navmesh. These values can be overridden with the corresponding
`--agent-*` flags only when the runtime config is changed to match.

Generate the smaller smoke-test split or limit generation to selected scenes
with:

    python scripts/build_person_objectnav_dataset.py --split val_mini
    python scripts/build_person_objectnav_dataset.py \
      --split val_mini --scene TEEsavR23oF

Validate an existing generated split against its live navmesh:

    python scripts/build_person_objectnav_dataset.py \
      --split val_mini --validate-only

Use ``--structural-only`` only for a quick JSON/category check when scene
assets or Habitat-Sim are unavailable. Normal live validation recomputes both
close-stop coverage and every episode start-to-viewpoint geodesic. It rejects
non-finite paths and a stored geodesic that differs from the current runtime
navmesh. Therefore an older person dataset generated with Habitat-Sim's bare
precomputed navmesh must be regenerated, not merely resumed.

The default view-point set is a 5 cm Cartesian lattice from 0.20 m to 1.10 m
around the person's ground projection. Every retained point must be finite,
on the same navmesh island, within 8 cm of its unsnapped candidate and 25 cm
of the person's navmesh height, and visible by an unobstructed environment ray
from robot camera height to humanoid torso height. Points are deduplicated at
4 cm, producing hundreds to roughly 1,500 endpoints per person, comparable to
the density of native HM3D ObjectNav goals.

Generation also uses an independent, half-cell-offset 5 cm probe lattice from
0.20 m to 1.00 m. Every visible navigable probe must have a geodesic distance
below 0.15 m to a retained view point. This leaves a 5 cm margin below the
runtime ``success_distance: 0.2``. A person position that fails this coverage
contract is rejected and the next deterministic source start is tried; hidden
or non-navigable points are never inserted merely to make validation pass.

## Run

    python main.py \
      --task_config person_objectnav_hm3d.yaml \
      --num_agents 2

configs/person_objectnav_hm3d.yaml enables conav.static_person_goal and
provides one humanoid by default. At every episode reset, the runtime reads
the person ObjectGoals and calls the same
``KinematicHumanoid.reconfigure()`` path used by the original humanoid demo.
Habitat automatically resolves the sibling ``.ao_config.json`` and textured
``.glb`` asset through that standard path; this benchmark does not duplicate
the model loader or force a separate render mode.
The articulated object is created on first use, moved by assigning its
``base_pos`` to the current goal, and reused while it remains alive; if a scene
reset invalidates it, Habitat recreates it through ``reconfigure()``. Static
humanoids remain in their resting pose, so ``walker.step()`` does not move them.

The original demo renders the newly placed model with ``env.sim.step(None)``.
Here RGB and depth are refreshed at the same point in the lifecycle, but the
new visual frames are merged into the observation returned by ``env.reset()``.
This keeps ObjectNav task sensors such as ``objectgoal``, GPS, and compass and
does not consume an action. The Habitat 3 humanoid is therefore placed before
the first navigation observation is processed.

For this benchmark, person detection comes from the rendered RGB image just
like chair, bed, and the other goal categories. The optional FireWorld thermal
pipeline does not project a known static goal position into a synthetic person
mask: that oracle path could see through a wall and previously produced a
``person 0.95`` box over wall pixels. Dynamic pedestrian experiments retain
thermal projection, but when depth is available a failed body-silhouette carve
now yields no person mask instead of falling back to an ellipse.

After a person is detected, it follows exactly the same policy path as chair,
bed, plant, toilet, tv, and sofa. The detected RGB-D point cloud is projected
into ``goal_map``; the nearest detected surface point is used by the Habitat
shortest-path follower; and the fallback FMM planner uses the normal
``disk(5)`` goal dilation and its normal STOP decision. There is no
person-specific stop ring, planner branch, or STOP override. Dataset
``goal.position`` and ``view_points`` remain evaluator-only data and are never
exposed to the policy.

The authoritative metrics are Habitat's standard success and spl. Success
requires the agent to call STOP while its geodesic distance to one of the
generated visible view points is below the configured 0.2 m tolerance.
There is no separate human-distance score and no goal-category override.

Because the denser endpoints change both success geometry and the shortest
path used by SPL, results produced with the older sparse person dataset are not
comparable and must be rerun.

If multiple person goals are included in a scene, ObjectNav category semantics
apply: reaching a valid view point for any person instance is sufficient.
Targeting a particular individual would require an instance-goal benchmark
instead.
