# syntax=docker/dockerfile:1.7

FROM condaforge/miniforge3:24.3.0-0 AS conda

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG HABITAT_LAB_COMMIT=094d6be2f9d057e4781a68ae792132895fd4d3d0
ARG VCS_REF=unknown

LABEL org.opencontainers.image.title="Co-NavGPT2 VULCAN"
LABEL org.opencontainers.image.description="Patched Habitat 0.3.3 multi-agent ObjectNav and FireWorld runtime"
LABEL org.opencontainers.image.source="https://github.com/SendingA/Co-NavGPT2"
LABEL org.opencontainers.image.revision="${VCS_REF}"

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/envs/conav/bin:/opt/conda/bin:${PATH} \
    CONDA_DEFAULT_ENV=conav \
    CONAV_CONTAINER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    HABITAT_SIM_LOG=quiet \
    MAGNUM_LOG=quiet

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglvnd0 \
        libgles2 \
        libgomp1 \
        libsm6 \
        libx11-6 \
        libxcursor1 \
        libxext6 \
        libxi6 \
        libxinerama1 \
        libxrandr2 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=conda /opt/conda /opt/conda
COPY docker/environment.yml /tmp/conav-environment.yml

RUN conda env create --file /tmp/conav-environment.yml \
    && conda clean --all --yes \
    && rm -f /tmp/conav-environment.yml

COPY docker/requirements.lock.txt /tmp/conav-requirements.lock.txt

RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1+cu118 \
        torchvision==0.15.2+cu118 \
        torchaudio==2.0.2+cu118 \
    && python -m pip install --no-cache-dir \
        --requirement /tmp/conav-requirements.lock.txt \
    && python -m pip install --no-cache-dir --force-reinstall --no-deps \
        opencv-python==4.10.0.84 \
    && rm -f /tmp/conav-requirements.lock.txt

RUN mkdir -p /opt/habitat-lab-src \
    && git -C /opt/habitat-lab-src init \
    && git -C /opt/habitat-lab-src remote add origin \
        https://github.com/facebookresearch/habitat-lab.git \
    && git -C /opt/habitat-lab-src fetch --depth 1 origin \
        "${HABITAT_LAB_COMMIT}" \
    && git -C /opt/habitat-lab-src checkout --detach FETCH_HEAD

COPY ref/habitat_lab_0.3.3_vulcan.patch /tmp/habitat-vulcan.patch

RUN git -C /opt/habitat-lab-src apply --check \
        /tmp/habitat-vulcan.patch \
    && git -C /opt/habitat-lab-src apply /tmp/habitat-vulcan.patch \
    && python -m pip install --no-cache-dir --no-deps \
        -e /opt/habitat-lab-src/habitat-lab \
        -e /opt/habitat-lab-src/habitat-baselines \
    && rm -f /tmp/habitat-vulcan.patch

WORKDIR /workspace
COPY . /workspace

RUN chmod 0755 \
        /workspace/scripts/docker_entrypoint.sh \
        /workspace/scripts/docker_preflight.py \
    && groupadd --gid 1000 conav \
    && useradd --create-home --uid 1000 --gid 1000 \
        --shell /bin/bash conav \
    && mkdir -p /workspace/outputs \
    && chown -R conav:conav /workspace/outputs

USER conav

ENTRYPOINT ["/workspace/scripts/docker_entrypoint.sh"]
CMD ["preflight", "--mode", "navigation", "--strict"]
