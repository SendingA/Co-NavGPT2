
# Co-NavGPT: Multi-Robot Cooperative Visual Semantic Navigation Using Vision Language Models

[**ArXiv**](https://arxiv.org/abs/2310.07937v3) | [**Project Page**](https://sites.google.com/view/co-navgpt2) | [**Video**](https://youtu.be/vnOJDUoQ7A8)

We proposed a new framework to explore and search for the target in unknown environment based on Vision Language Model. Our work is based on [VLN-Game](https://sites.google.com/view/vln-game). You can find the code of this paper about simulation and real-world implementation in ros2 foxy.

**Author:** Bangguo Yu, Qihao Yuan, Kailai Li, Hamidreza Kasaei, and Ming Cao

**Affiliation:** University of Groningen

## Abstract

Visual target navigation is a critical capability for autonomous robots operating in unknown environments, particularly in human-robot interaction scenarios. While classical and  earning-based methods have shown promise, most existing approaches lack common-sense reasoning and are typically designed for single-robot settings, leading to reduced efficiency and robustness in complex environments. To address these limitations, we introduce Co-NavGPT, a novel framework that integrates Vision-Language Models (VLMs) as global planners to enable common-sense multi-robot visual target navigation. Co-NavGPT aggregates sub-maps from multiple robots with diverse viewpoints into a unified global map, encoding robot states and frontier regions. The VLM uses this information to assign frontiers across the robots, facilitating coordinated and efficient exploration. Experiments on the Habitat-Matterport 3D (HM3D) demonstrate that Co-NavGPT outperforms existing baselines in terms of success rate and navigation efficiency, without requiring task-specific training. Ablation studies further confirm the importance of semantic priors from the VLM. We also validate the framework in real-world scenarios using quadrupedal robots.

![image-20200706200822807](img/framework.png)

## Installation

For a complete, branch-specific setup covering the patched Habitat runtime,
datasets, model assets, FireWorld generation, risk benchmarks, teleoperation,
all configuration files, and all scripts, see
[`docs/vulcan_reproduction.md`](docs/vulcan_reproduction.md).

The project now targets **Habitat-Lab 0.3.3 + Habitat-Sim 0.3.3**.
Humanoid pedestrians are provided by Habitat 3. The classic `Sim-v0`
multi-agent navigation contract and direct camera tilt actions require the
small patch stored at `ref/habitat_lab_0.3.3_vulcan.patch`; stock Habitat-Lab
0.3.3 alone is not sufficient. Follow the reproduction guide above when
creating a new environment.

- Set up the conda environment (Python 3.9 + CUDA 11.8 + PyTorch 2.0.1):
    ```
    conda create -n co-nav3 python=3.9 cmake=3.14.0
    conda activate co-nav3
    conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```

- Install habitat-sim 0.3.3 with Bullet physics:
    ```
    conda install habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat
    ```

- Clone this repository and install the remaining Python requirements:
    ```
    git clone --branch vulcan https://github.com/SendingA/Co-NavGPT2.git
    cd Co-NavGPT2/
    pip install -r requirements.txt
    ```

- Install the patched Habitat-Lab 0.3.3 runtime from the repository root:
    ```
    PROJECT_ROOT="$PWD"
    HABITAT_LAB_ROOT="/path/to/habitat-lab-0.3.3"
    git clone https://github.com/facebookresearch/habitat-lab.git "$HABITAT_LAB_ROOT"
    git -C "$HABITAT_LAB_ROOT" checkout 094d6be2f9d057e4781a68ae792132895fd4d3d0
    git -C "$HABITAT_LAB_ROOT" apply "$PROJECT_ROOT/ref/habitat_lab_0.3.3_vulcan.patch"
    pip install -e "$HABITAT_LAB_ROOT/habitat-lab"
    ```

- (Optional) Download Habitat 3 humanoid assets so `--num_humans > 0`
  works. These sit under `data/humanoids/humanoid_data/...`:
    ```
    python -m habitat_sim.utils.datasets_download --uids habitat_humanoids --data-path data/
    ```

- (Optional) Download visible robot URDF assets so
  `--robot_models_enabled 1` works. Fetch is bundled with habitat-sim;
  Spot and Stretch are separate uids:
    ```
    python -m habitat_sim.utils.datasets_download --uids hab_spot_arm --data-path data/
    python -m habitat_sim.utils.datasets_download --uids hab_stretch --data-path data/
    ```

### Download HM3D_v0.2 datasets:

Download [HM3D](https://aihabitat.org/datasets/hm3d/) dataset using download utility and [instructions](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d):
```
python -m habitat_sim.utils.datasets_download --username <api-token-id> --password <api-token-secret> --uids hm3d_minival_v0.2
```

## Setup


### Setting up datasets
The code requires the datasets in a `data` folder in the following format (same as habitat-lab):
```
Co-NavGPT/
  data/
    scene_datasets/
    versioned_data
    datasets/
        objectnav_hm3d_v2/
            val/
```

### Setup Openao API
```
export OPENAI_API_KEY="your_api_key_here"
```

### For evaluation:

Run the multi-robot ObjectNav task with the default config
(`configs/multi_objectnav_hm3d.yaml`, two robots, no humanoids):

```
python main.py
```

Add pedestrians:

```
python main.py --num_humans 3
```

Show visible articulated robot models on top of the nav agents:

```
python main.py --robot_models_enabled 1 --robot_profiles spot,fetch
```

For multiprocessing evaluation:

```
python main_vec.py -n 2
```

You can also add `-v 1` to enable the Open3D visualization UI to check
the maps.

See [`docs/habitat3_migration.md`](docs/habitat3_migration.md) for a
detailed summary of what changed relative to the previous
Habitat-Lab 0.2.1 version.

## Real-world Implementation

![real experiemnt](img/real.png)

You can find the code that how to use two Unitree Go2 robots to run the multi-robot visual target navigation in the unknown real world using ROS2 foxy. Each robot equiped with the RealSense D455 camera and Livox MID 360 lidar.

1. Install [ROS2 foxy](https://docs.ros.org/en/foxy/Installation.html) environment in Python 3.8 (ROS2 humble should also work well).

2. Install related dependencies in the same conda environment as simulation (**conda activate co-nav3**):
    ```
    sudo apt-get install ros-<ros_distro>-tf-transformations
    sudo pip3 install transforms3d
    pip install numpy-quaternion
    ```
3. Config your real robots and sensors following this [instruction](https://github.com/ASIG-X/Go2Go). Each robot has its namespace with robot ID, such as `robot_0`, `robot_1`, ....
4. Enable the conda environemnt
   ```
   conda activate co-nav3
   ```
5. Start all your robots and sensros. Let two robots stand side by side, facing forword together, with an initial distance of approximately 1.5 meters between them. Then calculate the registration between two robots using G-ICP by running:
   ```
   python multi_lidar_icp.py
   ```
   You will find the static transform of the registration in the terminal, and run it. For example:
   ```
   ros2 run tf2_ros static_transform_publisher -0.3489087452591184 1.6667295859539184 0 0.007028384544181147 -0.005619125549335727 0.10024310851203151 0.9949222816052544 camera_init_2  camera_init_1
   ```
   This `static_transform_publisher` is used to connect the two robots' tf2, then all the sensors' data can be transformed in the same coordinate system. You can also change the icp_initial_transform in `multi_lidar_icp.py` based on the formation of your robots.

6. For multi-robot visual target navigation, run:
   ```
   python ros_multi_nav.py
   ```
   Each robot will rotate around firstly to initial the environment map, then start navigation following the assignment of the VLM. You can also test your setup for each single robot with frontier-based exploration to verify the configurations before running the multi-robot experiment:
   ```
   python ros_singel_nav.py 
   ```

   You can also add `-v 1` to enable the Open3D visualization UI to check the map.
