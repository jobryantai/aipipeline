#!/bin/bash
set -e

echo "[*] Re-assembling video: scaling & padding 512x512 frames to 1920x1080..."

INPUT_FRAMES="/mnt/d/aipipeline/pipeline_output/stage5_anime_frames"
SOURCE_VIDEO="/mnt/d/aipipeline/stage1/final_dubbed_video.mp4"
OUTPUT_VIDEO="/mnt/d/aipipeline/pipeline_output/final_anime_video.mp4"

# FFmpeg filter explanation:
# 1. scale=ih*16/9:-1 -> scales height to match 1080 and adjusts width proportionally (or scale to fit height=1080)
# Let's scale height to 1080 (512 -> 1080 means height scales to 1080, width scales proportionally or we pad to 1920x1080)
# Better yet: scale to fit inside 1920x1080 while keeping aspect ratio, then pad the rest with black bars:
# scale=iw*min(1920/iw\,1080/ih):ih*min(1920/iw\,1080/ih) ... pad=1920:1080:(ow-iw)/2:(oh-ih)/2

ffmpeg -y \
       -framerate 30 -i "$INPUT_FRAMES/%04d.jpg" \
       -i "$SOURCE_VIDEO" \
       -filter_complex "[0:v]scale=1080:1080,pad=1920:1080:(1920-1080)/2:0:black[v_padded];[v_padded]fps=29.97,format=yuv420p[v_out]" \
       -map "[v_out]" -map 1:a \
       -c:v libx264 -crf 20 -c:a aac -shortest \
       "$OUTPUT_VIDEO"

echo "[*] Success! Corrected 1920x1080 anime video created at: $OUTPUT_VIDEO"
