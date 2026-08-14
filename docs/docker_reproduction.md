# Docker reproduction

This workflow packages the complete Co-NavGPT2 software stack while keeping
licensed datasets, model weights, credentials and generated artifacts outside
the image.

The resulting image contains:

- Ubuntu 22.04 and CUDA 11.8 runtime libraries;
- Python 3.9.19 and pinned direct Python dependencies;
- PyTorch 2.0.1, torchvision 0.15.2 and torchaudio 2.0.2 CUDA 11.8 wheels;
- Habitat-Sim 0.3.3 with Bullet;
- Habitat-Lab commit `094d6be2f9d057e4781a68ae792132895fd4d3d0`;
- the required `ref/habitat_lab_0.3.3_vulcan.patch` applied during build;
- the current project source and committed FireWorld plans.

HM3D cannot be redistributed inside the image because it is license-controlled.
The two detector checkpoints are also mounted separately so image publication
does not silently redistribute third-party weights.

## 1. Host prerequisites

Use a Linux x86-64 host with:

- Docker Engine with Compose v2;
- an NVIDIA GPU and driver compatible with CUDA 11.8;
- NVIDIA Container Toolkit configured for Docker;
- enough disk space for the image, Conda packages and HM3D assets.

Verify GPU forwarding before building the project image:

```bash
docker run --rm --gpus all \
  nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

The full navigation agent is not CPU-compatible: YOLO-World, MobileSAM and
Habitat rendering require the GPU/EGL path. The `conav-test` service remains
useful on CPU-only hosts.

## 2. Prepare host assets

From the repository root, the minimum navigation layout is:

```text
Co-NavGPT2/
├── data/
│   ├── datasets/objectnav_hm3d_v2/val/val.json.gz
│   └── scene_datasets/hm3d_v0.2/
│       └── hm3d_annotated_basis.scene_dataset_config.json
├── mobile_sam.pt
├── yolov8l-world.pt
└── outputs/docker/
```

The HM3D scene files and ObjectNav content shards below those roots are also
required for the episodes being evaluated. Follow the official Habitat/HM3D
download flow described in [vulcan_reproduction.md](vulcan_reproduction.md).

Validate detector hashes:

```bash
sha256sum mobile_sam.pt yolov8l-world.pt
```

Expected values:

```text
6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f  mobile_sam.pt
8bdfaef999116760247d6fb0b0f8fca064b43e94598b3d4a807ebae9bcf0cdd5  yolov8l-world.pt
```

Optional assets remain under the same mounted `data/` root:

- `data/humanoids/` for pedestrians and static-person ObjectNav;
- `data/robots/` for visible robot URDFs;
- `data/ddppo-models/gibson-2plus-resnet50.pth` for PointNav local planning.

The static-person dataset is generated data and must match the runtime agent
navmesh. The normal Compose data mount is intentionally read-only, so create
or refresh this dataset with a temporary write-enabled mount before running a
person benchmark:

```bash
CONAV_PERSON_DATA_DIR="${CONAV_DATA_DIR:-$PWD/data}"
docker compose run --rm \
  --volume "$CONAV_PERSON_DATA_DIR:/workspace/data:rw" \
  conav python scripts/build_person_objectnav_dataset.py --split val
docker compose run --rm \
  --volume "$CONAV_PERSON_DATA_DIR:/workspace/data:rw" \
  conav python scripts/build_person_objectnav_dataset.py \
    --split val --validate-only
```

After generation, normal navigation commands return to the read-only mount.
Do not resume a person benchmark checkpoint made with an older generated
dataset: episode order and the total episode count can change when unreachable
starts are removed.

## 3. Configure Compose

Create an untracked root `.env` from the template:

```bash
cp docker/.env.example .env
mkdir -p outputs/docker
```

Set `CONAV_HOST_UID` and `CONAV_HOST_GID` to the user that owns the output
directory. On a normal Linux host:

```bash
sed -i "s/^CONAV_HOST_UID=.*/CONAV_HOST_UID=$(id -u)/" .env
sed -i "s/^CONAV_HOST_GID=.*/CONAV_HOST_GID=$(id -g)/" .env
```

Do not commit `.env`. Prefer exporting `OPENAI_API_KEY` only in the shell that
runs a GPT experiment instead of saving it in a file.

If assets live outside the repository, set the corresponding absolute paths:

```text
CONAV_DATA_DIR=/datasets/conav/data
CONAV_OUTPUT_DIR=/experiments/conav
CONAV_MOBILE_SAM_PATH=/models/mobile_sam.pt
CONAV_YOLO_WORLD_PATH=/models/yolov8l-world.pt
```

The data and model mounts are read-only. Only `CONAV_OUTPUT_DIR` is writable.

## 4. Build and verify

Build the pinned image and record the project revision in its OCI metadata:

```bash
CONAV_VCS_REF="$(git rev-parse HEAD)" docker compose build conav
```

Run the CPU-capable software-stack tests without datasets, weights or a GPU:

```bash
docker compose --profile test run --rm conav-test
```

This runs the Docker contract, global/risk planner, local-planner and benchmark
launcher suites. To request the entire repository suite after mounting any
test-specific assets, override the command with `python -m unittest discover`.

Run the strict navigation preflight with all runtime mounts and GPU forwarding:

```bash
docker compose run --rm conav
```

The preflight checks exact core versions, the Habitat VULCAN patch, CUDA,
checkpoint hashes, required dataset roots and output writability. It exits
nonzero on a missing required component. Optional humanoid and PointNav assets
are warnings.

To inspect the software layer only:

```bash
docker compose --profile test run --rm conav-test \
  preflight --mode base --strict
```

## 5. Run navigation

Clean two-robot `co_ut` baseline:

```bash
docker compose run --rm conav python main.py \
  --task_config multi_objectnav_hm3d.yaml \
  --num_agents 2 --num_humans 0 \
  --nav_mode co_ut --cost_utility_lambda 0.5 \
  --seed 1 --fire_world 0 \
  --dump_location /workspace/outputs/clean_co_ut
```

Risk-aware FireWorld run:

```bash
# First bake the generated timeline into the mounted output directory. This is
# required once per plan/solver configuration and can produce a large file.
docker compose run --rm conav python -m utils.fire_world.propagation \
  --scene Nfvxx8J5NCo \
  --plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
  --scenes_root /workspace/scenes \
  --out_root /workspace/outputs/fire_world \
  --voxel_m 0.15 --dt 0.5 --save_dt 1.0

docker compose run --rm conav python main.py \
  --task_config multi_objectnav_hm3d.yaml \
  --num_agents 2 --nav_mode co_ut --seed 1 \
  --fire_world 1 \
  --fire_world_plan_id Nfvxx8J5NCo_bedroom_textile_severe_83679a07b632 \
  --fire_clock_mode step \
  --risk_enabled 1 --risk_source sensed \
  --risk_run_id docker_sensed_seed1 \
  --risk_dump_dir /workspace/outputs/risk_assessment \
  --dump_location /workspace/outputs/navigation
```

GPT run:

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
docker compose run --rm conav \
  preflight --mode gpt --strict

docker compose run --rm conav python main.py \
  --task_config multi_objectnav_hm3d.yaml \
  --num_agents 2 --nav_mode gpt --seed 1 \
  --fire_world 0 \
  --dump_location /workspace/outputs/clean_gpt
```

All relative configuration paths resolve under `/workspace`. Results appear in
the host directory configured by `CONAV_OUTPUT_DIR`.

## 6. Interactive windows

The default Compose service is headless. For a trusted local X11 session, add
the display socket for a single run:

```bash
xhost +si:localuser:$(id -un)
docker compose run --rm \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  conav python scripts/keyboard_teleop_full.py --help
```

Revoke the temporary X11 permission afterwards. Remote servers should use EGL,
VirtualGL or an explicitly configured virtual display instead.

## 7. Publish or transfer the software image

The software image can be transferred independently of licensed assets:

```bash
docker save "${CONAV_IMAGE:-conavgpt2:vulcan}" | gzip \
  > conavgpt2-vulcan-image.tar.gz
```

Recipients load it with `docker load`, obtain HM3D through their own license,
place checkpoints at the configured host paths, and run the same preflight.
For paper results, publish the Git commit, image digest, `.env` paths without
secrets, command line, seed, dataset split and FireWorld plan ID.

## 8. Known limitations

- `supervision==0.19.0` declares `opencv-python-headless` as a dependency,
  while the keyboard/X11 tools need the full OpenCV build. The image pins both
  package metadata entries to `4.10.0.84` and reinstalls `opencv-python` last,
  so the GUI-capable `cv2` files are active.
- The current repository does not contain redistributable HM3D scenes.
- Docker build requires network access to Conda/PyPI, PyTorch wheels and the
  pinned Habitat-Lab Git commit.
- The dependency file pins all direct runtime packages, but package indexes can
  remove old artifacts; archive the built image by digest for long-term use.
- A passing software-only test does not prove host NVIDIA/EGL compatibility.
- `main_vec.py` still intentionally rejects risk-aware evaluation; use
  `main.py` for synchronized risk maps and exposure metrics.
