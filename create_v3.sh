#!/bin/bash
set -e

echo "=========================================="
echo "[*] Setting up AnimeGAN2 Pipeline v3 Release"
echo "=========================================="

cd /mnt/d/aipipeline

# 1. Configure Git
echo "[*] Configuring Git identity for RT..."
git config user.name "RT"
git config user.email "job.ryantai@gmail.com"

# 2. Initialize Git if needed
if [ ! -d ".git" ]; then
    git init
    git branch -M main
fi

# 3. Create .gitignore
echo "[*] Writing .gitignore..."
cat << 'GITIGNORE_EOF' > .gitignore
# Python / Conda
__pycache__/
*.pyc
venv/
animeenv/

# Large binaries, weights, and output video files
pretrained_weights/
pipeline_output/
stage1/
*.mp4
*.pkl
GITIGNORE_EOF

# 4. Create preflight_check.sh
echo "[*] Writing preflight_check.sh..."
cat << 'PREFLIGHT_EOF' > preflight_check.sh
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
PREFLIGHT_EOF

# 5. Create mux_video.sh
echo "[*] Writing mux_video.sh..."
cat << 'MUX_EOF' > mux_video.sh
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
MUX_EOF

# 6. Create recover_and_run.sh
echo "[*] Writing recover_and_run.sh..."
cat << 'RECOVER_EOF' > recover_and_run.sh
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
RECOVER_EOF

# 7. Make scripts executable
chmod +x preflight_check.sh mux_video.sh recover_and_run.sh

# 8. Git commit and tag v3
echo "[*] Staging, committing, and tagging v3..."
git add preflight_check.sh mux_video.sh recover_and_run.sh .gitignore README.md 2>/dev/null || git add preflight_check.sh mux_video.sh recover_and_run.sh .gitignore
git commit -m "feat: release v3 automated build, pre-flight checks, and muxing pipeline" || echo "[*] No new changes to commit."

git tag -d v3 2>/dev/null || true
git tag -a v3 -m "AnimeGAN2 Pipeline v3 - Automated Build & Check"

echo "=========================================="
echo "[*] v3 Release Setup Complete Successfully!"
echo "=========================================="
