#!/usr/bin/env bash
# Base x adaptive ablation (goc, A, B, AB) for every attack, first 100 pairs, run inside Docker.
#
#   goc : base + adaptive        (SOTAttack.py)
#   A   : base + no adaptive     (SOTAttack_ablation.py)
#   B   : no base + adaptive     (SOTAttack_ablation.py)
#   AB  : no base + no adaptive  (SOTAttack_ablation.py)
# M = optim.num_crops = 5 crops, always averaged uniformly. 300 steps, batch 16, unseeded.
#
# Run on the GPU host:
#   bash docker_ablation.sh                      # all 4 ablations x fgsm, mifgsm, pgd on GPU 3
#   GPU=0 bash docker_ablation.sh                # another physical GPU
#   ATTACKS=fgsm bash docker_ablation.sh         # only FGSM
#   tail -f ablation_base_adaptive/runner.log
#
# The image clones the code from the 'ablation_study' branch (see Dockerfile); only the
# output folder and the Hugging Face cache are mounted, so results land on the host:
#   ablation_base_adaptive/<attack>/seeddefault/             goc
#   ablation_base_adaptive/<A|B|AB>/<attack>/seeddefault/    ablations
#   ablation_base_adaptive/runner.log                        full queue output
# Re-running the script resumes every variant from the images already saved.

set -euo pipefail
cd "$(dirname "$0")"

IMAGE=${IMAGE:-foa-attack}
NAME=${NAME:-foa-ablation}
GPU=${GPU:-3}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
ROOT=${ROOT:-ablation_base_adaptive}
ABLATIONS=${ABLATIONS:-goc,A,B,AB}
ATTACKS=${ATTACKS:-fgsm,mifgsm,pgd}

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "A container named '$NAME' already exists. Check it with 'docker logs $NAME',"
    echo "remove it with 'docker rm -f $NAME' if it is finished, or pick another NAME=..."
    exit 1
fi

# Always rebuild: cached layers make this quick, and a new commit on the branch re-clones the code.
docker build -t "$IMAGE" .

mkdir -p "$ROOT" "$HF_CACHE"
OUT_HOST="$(cd "$ROOT" && pwd)"

# Only physical GPU $GPU is exposed, so inside the container it is cuda:0 (the config default).
# --user keeps the output owned by you on the host instead of root.
docker run -d --rm \
    --name "$NAME" \
    --gpus "\"device=$GPU\"" \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp \
    -e HF_HOME=/hf_cache \
    -v "$OUT_HOST":/workspace/FOA-Attack/"$ROOT" \
    -v "$HF_CACHE":/hf_cache \
    "$IMAGE" \
    bash -c 'python run_variants.py "$@" >> "'"$ROOT"'/runner.log" 2>&1' _ \
        --ablations "$ABLATIONS" \
        --attacks "$ATTACKS" \
        --seeds default \
        --root "$ROOT" \
        data.num_samples=100 data.batch_size=16 optim.num_crops=5 optim.steps=300 \
        "$@"

echo "Started container '$NAME' on GPU $GPU."
echo "  progress : tail -f $ROOT/runner.log"
echo "  results  : $OUT_HOST"
echo "  stop     : docker stop $NAME   (re-run this script to resume)"
