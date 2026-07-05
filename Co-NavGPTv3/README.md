# Co-NavGPTv3

Habitat3.3 version of the Co-NavGPT multi-agent ObjectNav demo with random
humanoids and optional visible robot models.

**Installation**

This demo is tested with Python 3.9, Habitat-Lab 0.3.3, and Habitat-Sim
0.3.3 in the local `habitat` conda environment.

Create and activate the environment:

```bash
conda create -n habitat python=3.9 cmake=3.14.0
conda activate habitat
```

Install Habitat-Sim with Bullet physics:

```bash
conda install habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat
```


Install the local Habitat-Lab 3.3 checkout in editable mode:

```bash
cd /home/ybg-itx/Project/habitat33/habitat-lab
pip install -e habitat-lab
```


Download Habitat humanoid assets if they are missing:

```bash
cd /home/ybg-itx/Project/habitat33/habitat-lab
python -m habitat_sim.utils.datasets_download --uids habitat_humanoids --data-path data/
```

Optional robot assets:

```bash
python -m habitat_sim.utils.datasets_download --uids hab_stretch --data-path data/
```


**What This Demo Does**


- `env.reset()` returns `List[observations]` when multiple robot agents exist.
- `env.step([a0, a1, ...])` applies one discrete navigation action per robot.
- `main.py` follows the v2 shape: create `env`, create `agent`, reset each
  episode, then run `while not env.episode_over`.
- Each step builds `actions` with `for i in range(num_agents)` and
  `agent[i].act(...)`, then calls `env.step(actions)`.
- ObjectNav stays the active task; pedestrians are random kinematic humanoids,
  not SocialNav/Rearrange task agents.
- `data/datasets` and `data/scene_datasets` are symlinks to the existing v2
  HM3D/ObjectNav data.
- Pedestrians use Habitat3 humanoid controller root motion by default
  (`human_use_controller_root_motion: True`), so they may turn in place before
  walking instead of sliding immediately.
- Multiple pedestrians can use different humanoid meshes. Configure
  `human_urdfs` and `human_motion_data_paths`; the demo cycles through the
  lists when `num_humans` is larger than the number of profiles.


**Robot Models**

Habitat3.3 includes robot agent classes/configs for:

- `fetch`: `FetchRobot`
- `fetch_no_wheels`: `FetchRobotNoWheels`
- `fetch_suction`: `FetchSuctionRobot`
- `spot`: `SpotRobot`
- `stretch`: `StretchRobot`


**Run**

```bash
conda activate habitat
python Co-NavGPTv3/main.py
```

The default config is loaded as:

```bash
python Co-NavGPTv3/main.py --task_config objectnav_hm3d_multi_humans.yaml
```

Use `--config /path/to/file.yaml` only when passing a full config path.

**Keyboard Controls**

- `w`: move all robots forward
- `a`: turn all robots left
- `d`: turn all robots right
- `f` or space: stop
- `q` or escape: quit

**Useful Options**

```bash
python Co-NavGPTv3/main.py --num-robots 3 --num-humans 2
python Co-NavGPTv3/main.py --robot-profiles fetch,fetch_no_wheels
python Co-NavGPTv3/main.py --no-robot-models
python Co-NavGPTv3/main.py --no-display
python Co-NavGPTv3/main.py --config Co-NavGPTv3/configs/objectnav_hm3d_multi_humans.yaml
```

**Data**

`Co-NavGPTv3/data/datasets` and `Co-NavGPTv3/data/scene_datasets` are expected
to point to the existing v2 HM3D/ObjectNav data. If the symlinks are missing,
create them to the corresponding `Co-NavGPTv2/data` folders.

```bash
cd /home/ybg-itx/Project
mkdir -p Co-NavGPTv3/data
ln -s ../../Co-NavGPTv2/data/datasets Co-NavGPTv3/data/datasets
ln -s ../../Co-NavGPTv2/data/scene_datasets Co-NavGPTv3/data/scene_datasets
```
