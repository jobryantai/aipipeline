#!/bin/bash
set -e

echo "[*] Re-assembling video with explicit stream mapping..."

INPUT_FRAMES="/mnt/d/aipipeline/pipeline_output/stage5_anime_frames"
AUDIO_SOURCE="/mnt/d/aipipeline/stage1/final_dubbed_video.mp4"
OUTPUT_VIDEO="/mnt/d/aipipeline/pipeline_output/final_anime_video.mp4"

ffmpeg -y \
       -framerate 30 -i "$INPUT_FRAMES/%04d.jpg" \
       -i "$AUDIO_SOURCE" \
       -map 0:v -map 1:a \
       -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \
       "$OUTPUT_VIDEO"

echo "[*] Success! Corrected anime video created at: $OUTPUT_VIDEO"
