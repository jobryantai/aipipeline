#!/bin/bash
set -e

echo "=========================================="
echo "[*] Running AnimeGAN2 Pipeline Pre-Flight Checks"
echo "=========================================="

# 1. Check Python & PyTorch CUDA
echo -n "[*] Checking PyTorch & GPU availability... "
GPU_CHECK=$(/home/admin1/miniconda/envs/animeenv/bin/python3 -c "import torch; print('CUDA:', torch.cuda.is_available())")
echo "$GPU_CHECK"

# 2. Check FFmpeg binary
echo -n "[*] Checking FFmpeg binary... "
if command -v ffmpeg &> /dev/null; then
    echo "FOUND ($(ffmpeg -version | head -n 1 | awk '{print $3}'))"
else
    echo "MISSING! Run: sudo apt install ffmpeg"
    exit 1
fi

# 3. Check Input Frames Directory & Count
FRAME_DIR="/mnt/d/aipipeline/pipeline_output/stage5_anime_frames"
echo -n "[*] Checking stylized frames in $FRAME_DIR... "
if [ -d "$FRAME_DIR" ]; then
    FRAME_COUNT=$(find "$FRAME_DIR" -type f \( -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
    echo "FOUND $FRAME_COUNT frames."
    if [ "$FRAME_COUNT" -eq 0 ]; then
        echo "Error: No image frames found!"
        exit 1
    fi
else
    echo "MISSING directory!"
    exit 1
fi

# 4. Check Audio Source
AUDIO_PATH="/mnt/d/aipipeline/stage1/final_dubbed_video.mp4"
echo -n "[*] Checking audio source ($AUDIO_PATH)... "
if [ -f "$AUDIO_PATH" ]; then
    echo "FOUND."
else
    echo "MISSING!"
    exit 1
fi

echo "=========================================="
echo "[*] All checks passed successfully!"
echo "=========================================="
