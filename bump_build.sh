#!/bin/bash
set -e

VERSION_FILE="stage1/_version.py"
BRANCH="development"

# 1. Parse current version and increment build number
VERSION=$(grep '__version__' $VERSION_FILE | cut -d'"' -f2)
CURRENT_BUILD=$(grep '__build__' $VERSION_FILE | cut -d'"' -f2)
NEW_BUILD=$((CURRENT_BUILD + 1))
BUILD_TAG="v${VERSION}-build.${NEW_BUILD}"

# 2. Update stage1/_version.py with the new build number
cat <<EOF_INNER > $VERSION_FILE
__version__ = "${VERSION}"
__build__ = "${NEW_BUILD}"

def get_version_info():
    return f"AI Pipeline v{__version__} (Build {__build__})"
EOF_INNER

echo "Updated $VERSION_FILE to Build ${NEW_BUILD}"

# 3. Commit, Tag, and Push to GitHub
git add .
git commit -m "feat: add stage4 consistent avatar animation and auto-bump to ${BUILD_TAG}"
git tag -a "${BUILD_TAG}" -m "Automated Build Release ${BUILD_TAG}"
git push origin $BRANCH --tags

echo "Successfully created and pushed ${BUILD_TAG} to ${BRANCH}!"
