#!/bin/bash
set -e

echo "[*] Recreating WSL Anime Pipeline Environment (Version 6)..."

# Ensure system dependencies (ffmpeg, build tools)
sudo apt update && sudo apt install -y ffmpeg git build-essential

# Recreate Virtual Environment with Python 3.14
python3 -m venv /mnt/d/aipipeline/animegan2-pytorch/venv

# Upgrade pip inside venv
/mnt/d/aipipeline/animegan2-pytorch/venv/bin/pip install --upgrade pip

# Install frozen dependencies
/mnt/d/aipipeline/animegan2-pytorch/venv/bin/pip install -r requirements_v6.txt

echo "[+] Environment setup complete! Ready to execute batch_pipeline.sh"
