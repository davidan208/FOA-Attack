#!/usr/bin/env bash
# One-shot setup on a GPU box (e.g. vast.ai), without conda: an isolated Python 3.11 in ./.venv,
# so packages already installed on the machine (conda, apt) can't get in the way.
#
#   bash setup_vast.sh                 # RTX 5090 / Blackwell: torch 2.8 + CUDA 12.8 (driver >= 570)
#   TORCH_CUDA=cu124 bash setup_vast.sh  # A6000 / A100 / H100: torch 2.6 + CUDA 12.4 (driver >= 550)
#   source .venv/bin/activate          # then, in every new shell / tmux pane

set -euo pipefail
cd "$(dirname "$0")"
TORCH_CUDA=${TORCH_CUDA:-cu128}

case "$TORCH_CUDA" in
    cu128) TORCH="torch==2.8.0 torchvision==0.23.0" ;;
    cu124) TORCH="torch==2.6.0 torchvision==0.21.0" ;;
    *) echo "TORCH_CUDA must be cu128 or cu124"; exit 1 ;;
esac

if ! command -v uv > /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# uv downloads its own Python 3.11 if the machine has none.
uv venv --python 3.11 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install $TORCH --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
uv pip install -r requirements.txt

python - <<'PY'
import torch, transformers, numba, kmeans_pytorch, hydra, matplotlib  # noqa: F401
print("torch", torch.__version__, "| transformers", transformers.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not usable: check the driver (nvidia-smi) and TORCH_CUDA.")
name = torch.cuda.get_device_name(0)
cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
print("GPU:", name, cap, "| supported by this torch:", cap in torch.cuda.get_arch_list())
PY
echo
echo "Done. Activate in every new shell:  source $(pwd)/.venv/bin/activate"
