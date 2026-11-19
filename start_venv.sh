#!/bin/bash

VENV_DIR="/mnt/c/aipipeline/animegan2-pytorch/venv"
PIPELINE_DIR="/mnt/c/aipipeline"

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[!] Virtual environment not found at: $VENV_DIR" >&2
    return 1 2>/dev/null || exit 1
fi

# If executed directly (not sourced) with no arguments, remind how to stay in the shell
if [[ "${BASH_SOURCE[0]}" == "${0}" ]] && [ "$#" -eq 0 ]; then
    echo "[!] Run 'source ./start_venv.sh' to activate in this shell, or pass a command:"
    echo "    ./start_venv.sh bash batch_pipeline.sh"
    exit 1
fi

cd "$PIPELINE_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

export COQUI_TOS_AGREED="1"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1"

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

echo "[+] Venv active: $(which python) ($(python --version))"
