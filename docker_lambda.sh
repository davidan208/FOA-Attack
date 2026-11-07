#!/usr/bin/env bash
# Lambda logging study of SOTAttack (original algorithm, measurement only) + Gemini benchmark.
# Settings of the main run: FGSM (iterative sign), M = 5, T = 300, eps 16/255, alpha 1/255,
# crops [0.5, 0.9], K = 10, batch 16, unseeded. The study runs on the first NUM pairs (default 20).
#
#   bash docker_lambda.sh check     # pair #0 alone, log + gradient diagnostic, prints the summary
#   bash docker_lambda.sh verify    # pair #0 with a fixed seed: plain / logged / plain -> images must match
#   bash docker_lambda.sh full      # pairs 0..NUM-1 with logging (background container 'foa-lambda')
#   bash docker_lambda.sh analyze   # tables + histogram from the full run's logs
#   GEMINI_API_KEY=... GEMINI_MODEL=<model id> bash docker_lambda.sh bench <name> [<adv image dir>]
#                                   # optional black-box check, not needed for the lambda study
#
# Env: GPU (default 7), NUM (default 20), ROOT (default lambda_study), DIAG_EVERY (default 10; 0 = no gradient
# diagnostic), VERIFY_STEPS (default 300), GEMINI_THINKING (default low), HF_CACHE.
# Everything is written under ./$ROOT on the host.

set -euo pipefail
cd "$(dirname "$0")"

CMD=${1:-}
IMAGE=${IMAGE:-foa-attack}
GPU=${GPU:-7}
NUM=${NUM:-20}
ROOT=${ROOT:-lambda_study}
DIAG_EVERY=${DIAG_EVERY:-10}
VERIFY_STEPS=${VERIFY_STEPS:-300}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
WS=/workspace/FOA-Attack

mkdir -p "$ROOT" "$HF_CACHE"
ROOT_HOST="$(cd "$ROOT" && pwd)"

ATTACK_ARGS=(
    --config-name=ensemble_3models attack=fgsm
    data.cle_data_path=resources/images/bigscale_100
    data.tgt_data_path=resources/images/target_images_100
    optim.num_crops=5 optim.steps=300 optim.alpha=1.0 optim.epsilon=16
)
if [ "$DIAG_EVERY" -gt 0 ]; then
    LOG_ARGS=(log.lambda_log=true log.diag_grad=true "log.diag_every=$DIAG_EVERY")
else
    LOG_ARGS=(log.lambda_log=true)
fi

build() { docker build -q -t "$IMAGE" . > /dev/null; }

# docker run with the GPU, the output folder and the HF cache mounted; extra docker flags first.
run_gpu() {
    docker run --rm --gpus "\"device=$GPU\"" --ipc=host --user "$(id -u):$(id -g)" \
        -e HOME=/tmp -e HF_HOME=/hf_cache \
        -v "$ROOT_HOST":"$WS/$ROOT" -v "$HF_CACHE":/hf_cache "$@"
}

attack() {  # attack <out subdir> <hydra overrides...>
    local out=$1; shift
    python_cmd=(python SOTAttack.py "${ATTACK_ARGS[@]}" "data.output=$ROOT/$out" "hydra.run.dir=$ROOT/$out/hydra" "$@")
    run_gpu "$IMAGE" "${python_cmd[@]}"
}

summary() {  # summary <out subdir> <steps>
    local out=$1 steps=$2
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
        -v "$ROOT_HOST":"$WS/$ROOT" "$IMAGE" \
        bash -c 'python summarize.py "$1/logs" --steps "$2" --out "$1/analysis"' \
        _ "$ROOT/$out" "$steps"
}

case "$CMD" in
check)
    build
    attack check data.num_samples=1 data.batch_size=1 seed=null "${LOG_ARGS[@]}" 2>&1 | tr '\r' '\n' | grep -v kmeans | tail -15
    summary check 300
    echo "Check output: $ROOT_HOST/check (images, logs/, analysis/)"
    ;;
verify)
    build
    for v in plain logged plain2; do
        extra=()
        [ "$v" = logged ] && extra=("${LOG_ARGS[@]}")
        echo "== verify: $v ($VERIFY_STEPS steps, seed 0)"
        attack "verify/$v" data.num_samples=1 data.batch_size=1 seed=0 "optim.steps=$VERIFY_STEPS" ${extra[@]+"${extra[@]}"} \
            2>&1 | tr '\r' '\n' | grep -E "Saved|Error|error" || true
    done
    docker run --rm -i -v "$ROOT_HOST":"$WS/$ROOT" "$IMAGE" python - "$ROOT/verify" <<'PY'
import glob, sys
import numpy as np
from PIL import Image
root = sys.argv[1]
def img(v):
    paths = glob.glob(f"{root}/{v}/img/*/*/*.png")
    assert len(paths) == 1, (v, paths)
    return np.asarray(Image.open(paths[0])).astype(int)
a, b, c = img("plain"), img("logged"), img("plain2")
d_log, d_rep = np.abs(a - b), np.abs(a - c)
print(f"plain vs logged : max |diff| = {d_log.max()}, pixels differing = {100 * (d_log > 0).mean():.3f}%")
print(f"plain vs plain2 : max |diff| = {d_rep.max()}, pixels differing = {100 * (d_rep > 0).mean():.3f}%  (GPU run-to-run noise)")
print("RESULT:", "identical" if d_log.max() == 0 else
      "differs, but no more than two plain runs differ" if d_log.max() <= d_rep.max() else
      "logging changed the result - investigate")
PY
    ;;
full)
    build
    mkdir -p "$ROOT/full"
    if docker ps -a --format '{{.Names}}' | grep -qx foa-lambda; then
        echo "Container 'foa-lambda' already exists; stop it with 'docker rm -f foa-lambda' first."
        exit 1
    fi
    run_gpu -d --name foa-lambda "$IMAGE" bash -c '"$@" >> "'"$ROOT"'/full/run.log" 2>&1' _ \
        python SOTAttack.py "${ATTACK_ARGS[@]}" "data.output=$ROOT/full" "hydra.run.dir=$ROOT/full/hydra" \
        "data.num_samples=$NUM" data.batch_size=16 seed=null "${LOG_ARGS[@]}" > /dev/null
    echo "Started 'foa-lambda' on GPU $GPU for pairs 0-$((NUM - 1)). Log: $ROOT_HOST/full/run.log"
    echo "Images so far: find $ROOT/full/img -name '*.png' | wc -l   (re-run 'full' after a stop to resume)"
    ;;
analyze)
    build
    summary full 300
    echo "Analysis: $ROOT_HOST/full/analysis"
    ;;
bench)
    NAME=${2:?usage: bash docker_lambda.sh bench <name> [<adv image dir>]}
    ADV=${3:-$ROOT/full/img}
    : "${GEMINI_API_KEY:?set GEMINI_API_KEY}"
    : "${GEMINI_MODEL:?set GEMINI_MODEL to the exact Gemini model id}"
    [ -d "$ADV" ] || { echo "No image folder '$ADV' (run 'full' first or pass a folder)."; exit 1; }
    build
    ADV_HOST="$(cd "$ADV" && pwd)"
    # -e VAR without a value passes it from this shell, so the key never shows up in 'ps'.
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -e GEMINI_API_KEY -e GEMINI_MODEL -e GEMINI_THINKING \
        -v "$ROOT_HOST":"$WS/$ROOT" -v "$ADV_HOST":/adv:ro "$IMAGE" \
        python gemini_benchmark.py --adv-dir /adv --out "$ROOT/bench/$NAME" \
            --target-cache "$ROOT/bench/target_desc.jsonl" --num "$NUM" "${@:4}"
    echo "Benchmark: $ROOT_HOST/bench/$NAME (results.jsonl, summary.json)"
    ;;
*)
    sed -n '2,19p' "$0"
    exit 1
    ;;
esac
