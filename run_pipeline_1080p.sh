#!/bin/bash
set -e

echo "=========================================="
echo "[*] Starting Optimized 1080p Widescreen Anime Pipeline"
echo "=========================================="

PIPELINE_DIR="/mnt/d/aipipeline"
SOURCE_VIDEO="$PIPELINE_DIR/stage1/final_dubbed_video.mp4"
RAW_FRAMES_DIR="$PIPELINE_DIR/pipeline_output/stage3_raw_frames"
ANIME_FRAMES_DIR="$PIPELINE_DIR/pipeline_output/stage5_anime_frames"
PYTHON_ENV="/home/admin1/miniconda/envs/animeenv/bin/python3"

# 1. Extract frames (skip if already extracted to save time)
if [ ! -d "$RAW_FRAMES_DIR" ] || [ -z "$(ls -A $RAW_FRAMES_DIR 2>/dev/null)" ]; then
    echo "[*] Step 1: Extracting 1920x1080 raw frames from source video..."
    mkdir -p "$RAW_FRAMES_DIR"
    ffmpeg -y -i "$SOURCE_VIDEO" -q:v 2 "$RAW_FRAMES_DIR/%04d.jpg"
else
    echo "[*] Step 1: Raw frames already extracted, skipping extraction."
fi

mkdir -p "$ANIME_FRAMES_DIR"

# 2. Run Python Inference Script
echo "[*] Step 2: Running optimized inference script with frame caching..."
$PYTHON_ENV "$PIPELINE_DIR/inference_1080p.py"

# 3. Run Pre-Flight Checks
echo "[*] Step 3: Running Pre-Flight Checks..."
"$PIPELINE_DIR/preflight_check.sh"

# 4. Mux Final Video with Audio
echo "[*] Step 4: Re-assembling final widescreen anime video..."
"$PIPELINE_DIR/mux_video.sh"

echo "=========================================="
echo "[*] Pipeline complete! Widescreen video ready."
echo "=========================================="
