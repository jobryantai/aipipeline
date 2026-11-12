#!/bin/bash
set -e

# --- Environment Setup ---
export INPUT_DIR="/mnt/d/aipipeline/input_videos"
export OUTPUT_DIR="/mnt/d/aipipeline/pipeline_output"

# Point directly to the virtual environment's python binary
VENV_PYTHON="/mnt/d/aipipeline/animegan2-pytorch/venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[!] Error: Virtual environment python not found at $VENV_PYTHON" >&2
    exit 1
fi

echo "[*] Using Python binary: $VENV_PYTHON"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "[*] AnimeGANv2 Batch Pipeline v5.0 (Production) - Scanning in $INPUT_DIR..."

# --- Multi-Video Batch Loop ---
for video in "$INPUT_DIR"/*.mp4; do
    # Skip if no matches found
    [ -e "$video" ] || continue

    filename=$(basename "$video")
    basename_no_ext="${filename%.*}"

    echo "=================================================="
    echo "[*] Processing Video: $basename_no_ext"
    echo "=================================================="

    # Create isolated staging directories per video
    VID_OUTPUT_DIR="$OUTPUT_DIR/$basename_no_ext"
    RAW_FRAMES="$VID_OUTPUT_DIR/stage3_raw_frames"
    ANIME_FRAMES="$VID_OUTPUT_DIR/stage5_anime_frames"

    mkdir -p "$RAW_FRAMES"
    mkdir -p "$ANIME_FRAMES"

    # Stage 1: Extract frames from source video
    echo "[*] Extracting raw frames..."
    ffmpeg -y -i "$video" -q:v 2 "$RAW_FRAMES/%04d.jpg"

    # Stage 2: Run AnimeGANv2 Inference using venv python
    echo "[*] Running AnimeGANv2 inference with cache & thresholding..."
    "$VENV_PYTHON" /mnt/d/aipipeline/inference_1080p.py \
        --input_dir "$RAW_FRAMES" \
        --output_dir "$ANIME_FRAMES" \
        --threshold 2.5

    # Stage 3: Dynamic Duration Detection & Flawless Muxing
    echo "[*] Calculating exact video duration..."
    INPUT_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video")
    echo "[*] Target Duration: ${INPUT_DURATION}s"

    echo "[*] Muxing final synchronized video..."
    ffmpeg -y -framerate 30000/1001 \
      -i "$ANIME_FRAMES/%04d.jpg" \
      -i "$video" \
      -c:v libx264 -pix_fmt yuv420p \
      -fps_mode cfr \
      -c:a aac -b:a 128k \
      -filter_complex "[1:a]apad=whole_dur=${INPUT_DURATION}[aout]" \
      -map 0:v:0 -map "[aout]" \
      -t "$INPUT_DURATION" \
      "$OUTPUT_DIR/${basename_no_ext}_anime_production.mp4"

    echo "[+] Successfully completed: ${basename_no_ext}_anime_production.mp4"
    echo ""
done

echo "[*] All batch processing tasks completed successfully!"
