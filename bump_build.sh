#!/bin/bash
set -e

VERSION_FILE="stage1/_version.py"
BRANCH="development"

# 1. Safely parse and increment build number using Python
python3 -c '
import re

with open("stage1/_version.py", "r") as f:
    content = f.read()

# Find current build number (expecting integer string like "1", "2")
match = re.search(r"__build__\s*=\s*\"(\d+)\"", content)
if match:
    old_build = int(match.group(1))
    new_build = old_build + 1
    content = content.replace(f"__build__ = \"{old_build}\"", f"__build__ = \"{new_build}\"")
else:
    new_build = 1

v_match = re.search(r"__version__\s*=\s*\"([^\"]+)\"", content)
version = v_match.group(1) if v_match else "2.0.0"

with open("stage1/_version.py", "w") as f:
    f.write(content)

print(f"VERSION={version}")
print(f"BUILD={new_build}")
' > .bump_vars.env

source .bump_vars.env
rm .bump_vars.env

BUILD_TAG="v${VERSION}-build.${BUILD}"
echo "Updated $VERSION_FILE to Build ${BUILD}"

# 2. Commit, Tag, and Push to GitHub
git add .
git commit -m "feat: add stage4 consistent avatar animation and auto-bump to ${BUILD_TAG}"
git tag -a "${BUILD_TAG}" -m "Automated Build Release ${BUILD_TAG}"
git push origin $BRANCH --tags

echo "Successfully created and pushed ${BUILD_TAG} to ${BRANCH}!"
