#!/bin/bash
set -e

PROJECT_DIR="/mnt/c/aipipeline"
echo "[*] Cleaning up previous pipeline outputs and temporary caches..."

# Remove previous pipeline outputs (keeps input_videos intact!)
rm -rf "${PROJECT_DIR}/pipeline_output"/*

# Re-create root pipeline output directory
mkdir -p "${PROJECT_DIR}/pipeline_output"

echo "[+] Pipeline environment successfully cleaned and reset!"
