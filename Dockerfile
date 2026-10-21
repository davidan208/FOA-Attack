# CUDA 13.0 build: needed for Blackwell B300 (sm_103); host driver must be >= 580.
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Only what the attack path imports; torch/torchvision come with the base image.
RUN pip install --no-cache-dir \
    transformers==4.49.0 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    wandb==0.27.0 \
    PyYAML \
    tqdm \
    git+https://github.com/subhadarship/kmeans_pytorch.git

ARG BRANCH=ablation_study
# Changes whenever the branch gets a new commit, so the clone below is not served stale from cache.
ADD https://api.github.com/repos/davidan208/FOA-Attack/git/refs/heads/${BRANCH} /tmp/branch_ref.json
RUN git clone --depth 1 --branch ${BRANCH} https://github.com/davidan208/FOA-Attack.git /workspace/FOA-Attack

WORKDIR /workspace/FOA-Attack

CMD ["/bin/bash"]
