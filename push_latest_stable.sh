#!/bin/bash
set -e

cd /mnt/c/aipipeline

TARGET_BRANCH="${1:-stable}"
if ! git show-ref --verify --quiet "refs/heads/$TARGET_BRANCH"; then
    if git show-ref --verify --quiet "refs/heads/stable_v7_5090"; then
        TARGET_BRANCH="stable_v7_5090"
    fi
fi

REMOTE_NAME="${2:-origin}"
if ! git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    REMOTE_NAME="$(git remote | head -n 1)"
fi

if [ -z "$REMOTE_NAME" ]; then
    echo "[!] No Git remote configured. Add one first with: git remote add origin <repo-url>" >&2
    exit 1
fi

echo "[*] Pushing '$TARGET_BRANCH' to '$REMOTE_NAME'..."
git push -u "$REMOTE_NAME" "$TARGET_BRANCH"
echo "[+] Push complete."
