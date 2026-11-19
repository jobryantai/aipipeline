#!/bin/bash
set -e

cd /mnt/c/aipipeline

SOURCE_BRANCH="${1:-stable}"
if ! git show-ref --verify --quiet "refs/heads/$SOURCE_BRANCH"; then
    if git show-ref --verify --quiet "refs/heads/stable_v7_5090"; then
        SOURCE_BRANCH="stable_v7_5090"
    fi
fi

DEV_BRANCH="${2:-development}"

echo "[*] Source branch: $SOURCE_BRANCH"
echo "[*] Target branch: $DEV_BRANCH"

if git show-ref --verify --quiet "refs/heads/$DEV_BRANCH"; then
    git checkout "$DEV_BRANCH"
    git merge --no-edit "$SOURCE_BRANCH"
    echo "[+] Merged '$SOURCE_BRANCH' into '$DEV_BRANCH'."
else
    git checkout -b "$DEV_BRANCH" "$SOURCE_BRANCH"
    echo "[+] Created '$DEV_BRANCH' from '$SOURCE_BRANCH'."
fi

git log -n 2 --oneline --decorate
