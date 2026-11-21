#!/usr/bin/env bash
# The runs still missing before the Sunday-noon cut, one queue on all four GPUs (one run per GPU at a time).
# Stop whatever is running first (Ctrl-C); interrupted runs resume from their last saved image.
#
#   bash run_rest.sh          # then, at any time:  python timing.py
#
# Order = priority:
#   fix05_M1_T700 s0,s1   lambda 0.5, M = 1, T = 700   (resumes)
#   fix05_M3_T420 s0      lambda 0.5, M = 3, T = 420
#   fix05_M7_T233 s0      lambda 0.5, M = 7, T = 233
#   fix05_bv s0,s1,s2     lambda 0.5, M = 5, T = 300 with the views of a step batched (wall-clock time)
set -euo pipefail
cd "$(dirname "$0")"

ONLY=fix05_M1_T700:0+1,fix05_M3_T420:0,fix05_M7_T233:0,fix05_bv:0+1+2 \
STAGE=gen GPUS=${GPUS:-0,1,2,3} bash run_all.sh
