# CUDA 12.4 build: runs on the server driver (CUDA 12.6). Versions below follow requirements_old.txt,
# the environment this code originally ran in on the server.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# numba is a hard import of kmeans_pytorch; torch/torchvision come with the base image.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pillow==11.1.0 \
    transformers==4.49.0 \
    tokenizers==0.21.0 \
    safetensors==0.5.0 \
    huggingface-hub==0.28.1 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0 \
    wandb==0.19.1 \
    PyYAML==6.0.2 \
    tqdm==4.67.1 \
    numba==0.60.0 \
    llvmlite==0.43.0 \
    git+https://github.com/subhadarship/kmeans_pytorch.git

ARG BRANCH=ablation_study
# Changes whenever the branch gets a new commit, so the clone below is not served stale from cache.
ADD https://api.github.com/repos/davidan208/FOA-Attack/git/refs/heads/${BRANCH} /tmp/branch_ref.json
RUN git clone --depth 1 --branch ${BRANCH} https://github.com/davidan208/FOA-Attack.git /workspace/FOA-Attack

WORKDIR /workspace/FOA-Attack

CMD ["/bin/bash"]
