#!/bin/bash
set -e

echo "[*] Initializing AnimeGAN2 Pipeline Recovery & Run..."

if [ ! -d "/home/admin1/miniconda/envs/animeenv" ]; then
    echo "[!] Conda environment 'animeenv' not found. Please verify Miniconda installation."
    exit 1
fi

echo "[*] Environment verified. Running pre-flight checks..."
/mnt/d/aipipeline/preflight_check.sh

echo "[*] Ready to execute pipeline!"
