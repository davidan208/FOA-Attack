#!/usr/bin/env bash
# Ablation study end to end, no Docker: environment -> adversarial images -> benchmark -> tables.
#
#   bash run_all.sh                     # everything on GPU 0
#   GPUS=0,1,2,3 bash run_all.sh        # four GPUs share the 42 runs
#   STAGE=gen bash run_all.sh           # images only;  STAGE=eval: benchmark + tables only
#
# Needs a .env file here with:  GEMINI_API_KEY=...   DEEPSEEK_API_KEY=...
# Env: GPUS (0), SHARD (0/1, several machines only), SEEDS (0,1,2), ONLY (variants, default all 14), STAGE (all|gen|eval),
#      TORCH_CUDA (cu128 for RTX 50xx, cu124 for A6000/A100/H100), OUT (ablation_out).
# Everything resumes: re-run the same command after an interruption.

set -euo pipefail
cd "$(dirname "$0")"

STAGE=${STAGE:-all}
GPUS=${GPUS:-0}
SHARD=${SHARD:-0/1}
SEEDS=${SEEDS:-0,1,2}
OUT=${OUT:-ablation_out}

if [ ! -x .venv/bin/python ]; then
    bash setup_vast.sh
fi
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [ "$STAGE" != gen ]; then
    for key in GEMINI_API_KEY DEEPSEEK_API_KEY; do
        if [ -z "${!key:-}" ] && ! grep -q "^$key=." .env 2>/dev/null; then
            echo "Missing $key: add a line $key=... to $(pwd)/.env"
            exit 1
        fi
    done
fi

if [ "$STAGE" = all ] || [ "$STAGE" = gen ]; then
    python ablation_runs.py --out "$OUT" --gpus "$GPUS" --shard "$SHARD" --seeds "$SEEDS" ${ONLY:+--only "$ONLY"}
fi

if [ "$STAGE" = all ] || [ "$STAGE" = eval ]; then
    python evaluate.py --out "$OUT"
    python aggregate.py --out "$OUT"
    # Everything needed to re-check the results by hand: all images, descriptions, scores, tables.
    tar czf ablation_results.tgz -C "$OUT" images eval
    echo "Download: $(pwd)/ablation_results.tgz ($(du -h ablation_results.tgz | cut -f1))"
fi
