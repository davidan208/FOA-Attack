#!/usr/bin/env bash
# Ablation on the number of extra source views M (optim.num_crops).
# SOTAttack FGSM, 300 steps, first 100 pairs, batch 16, unseeded; M = 7, 3, 1 (M = 5 is the main run).
#
# Run from anywhere:
#   nohup bash run_ablation_crops.sh > ablation_crops.out 2>&1 &
#   tail -f ablation_crops/queue.log
# Extra Hydra overrides are passed through, e.g.:
#   bash run_ablation_crops.sh model.device=cuda:1
#
# Output, one folder per M:
#   ablation_crops/M7/img/...    adversarial images
#   ablation_crops/M7/run.log    full SOTAttack output
#   ablation_crops/queue.log     start/end of every run
# Re-running the script resumes each M from the images already saved.

set -u
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
ROOT=ablation_crops
QUEUE_LOG="$ROOT/queue.log"
mkdir -p "$ROOT"

# Abort if this Python cannot see a GPU: SOTAttack would silently fall back to CPU.
if ! "$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "$(date '+%F %T') | CUDA not available for $(command -v "$PYTHON") - aborting" | tee -a "$QUEUE_LOG"
    exit 1
fi
echo "$(date '+%F %T') | python: $(command -v "$PYTHON") | GPU: $("$PYTHON" -c 'import torch; print(torch.cuda.get_device_name(0))')" | tee -a "$QUEUE_LOG"

for M in 3 1; do
    OUT="$ROOT/M$M"
    mkdir -p "$OUT"
    echo "$(date '+%F %T') | start M=$M -> $OUT" | tee -a "$QUEUE_LOG"

    "$PYTHON" SOTAttack.py --config-name=ensemble_3models \
        attack=fgsm seed=null \
        optim.use_mca=true optim.num_crops="$M" optim.steps=300 \
        data.num_samples=100 data.batch_size=16 \
        data.output="$OUT" hydra.run.dir="$OUT/hydra" \
        "$@" >> "$OUT/run.log" 2>&1
    status=$?

    n=$(find "$OUT/img" -name '*.png' 2>/dev/null | wc -l | tr -d ' ')
    echo "$(date '+%F %T') | end   M=$M exit=$status images=$n/100" | tee -a "$QUEUE_LOG"

    sleep 10  # let the GPU memory be released before the next run
done

echo "$(date '+%F %T') | all done" | tee -a "$QUEUE_LOG"
