#!/bin/bash
set -e

cd /mnt/c/aipipeline

TARGET_BRANCH="${1:-stable}"
if ! git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
    if git show-ref --verify --quiet "refs/heads/stable_v7_5090"; then
        TARGET_BRANCH="stable_v7_5090"
    fi
fi

echo "[*] Checking out target branch: $TARGET_BRANCH"
git checkout "$TARGET_BRANCH"

echo "[*] Staging pipeline scripts..."
git add translate_dub_local.py batch_pipeline.sh inference_1080p.py

if git diff --cached --quiet; then
    echo "[*] No new changes to commit on $TARGET_BRANCH."
else
    git commit -m "Update dubbing I/O shim, NVENC 4K muxing, and batched AnimeGANv2 inference for RTX 5090"
    echo "[+] Committed changes to $TARGET_BRANCH."
fi

git status -s
git log -n 1 --oneline
