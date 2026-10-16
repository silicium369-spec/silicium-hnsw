#!/usr/bin/env bash
# ==============================================================================
# Silicium ANN: Release Asset Downloader & Integrity Verifier
# ==============================================================================
set -euo pipefail

TAG="${1:-ann-v1.0.0}"
DEST_DIR="${2:-$(pwd)/lib}"
mkdir -p "$DEST_DIR"

SO_NAME="libsilicium_ann_x86_64.so"
TARGET_SO="$DEST_DIR/$SO_NAME"

echo "  ▶ Locating or downloading $SO_NAME (release: $TAG)..."

# 1. Check local candidates first (dist, repo, system paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDIDATE_PATHS=(
    "${SILICIUM_LOCAL_SO:-}"
    "$SCRIPT_DIR/../../dist/$SO_NAME"
    "$SCRIPT_DIR/../dist/$SO_NAME"
    "$TARGET_SO"
    "$SCRIPT_DIR/../lib/$SO_NAME"
    "/usr/local/lib/$SO_NAME"
    "/usr/lib/$SO_NAME"
)

FOUND=""
for p in "${CANDIDATE_PATHS[@]}"; do
    if [ -n "$p" ] && [ -f "$p" ] && [ -s "$p" ]; then
        FOUND="$p"
        break
    fi
done

if [ -n "$FOUND" ] && [ "$(realpath "$FOUND" 2>/dev/null || echo "$FOUND")" != "$(realpath "$TARGET_SO" 2>/dev/null || echo "$TARGET_SO")" ]; then
    echo "  ✅ Found local pre-compiled binary: $(basename "$FOUND")"
    cp -f "$FOUND" "$TARGET_SO"
elif [ -n "$FOUND" ]; then
    echo "  ✅ Binary already present at destination: $(basename "$TARGET_SO")"
else
    # 2. Try downloading via GitHub CLI if available and authenticated
    REPO_NAME="silicium369-spec/silicium-hnsw"
    if command -v gh &> /dev/null && gh auth status &> /dev/null; then
        echo "  ⬇️ Attempting download via gh release (authenticated)..."
        gh release download "$TAG" -p "$SO_NAME" --repo "$REPO_NAME" -D "$DEST_DIR" --clobber 2>/dev/null || true
    fi

    # 3. Fallback to direct public release download (with optional auth header)
    if [ ! -f "$TARGET_SO" ] || [ ! -s "$TARGET_SO" ]; then
        RELEASE_URL="https://github.com/$REPO_NAME/releases/download/$TAG/$SO_NAME"
        echo "  ⬇️ Downloading from $RELEASE_URL..."
        DOWNLOAD_TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
        if command -v curl &> /dev/null; then
            if [ -n "$DOWNLOAD_TOKEN" ]; then
                curl -fsSL -H "Authorization: token $DOWNLOAD_TOKEN" -o "$TARGET_SO" "$RELEASE_URL" || true
            else
                curl -fsSL -o "$TARGET_SO" "$RELEASE_URL" || true
            fi
        elif command -v wget &> /dev/null; then
            wget -qO "$TARGET_SO" "$RELEASE_URL" || true
        fi
    fi
fi

if [ ! -f "$TARGET_SO" ] || [ ! -s "$TARGET_SO" ]; then
    echo "  ❌ Failed to obtain $SO_NAME automatically." >&2
    echo "  💡 Note: Please ensure release '$TAG' is published on $REPO_NAME or place $SO_NAME into $DEST_DIR." >&2
    exit 1
fi

EXPECTED_SHA256="11bfdd30bfa78fbffc2b9d6ae88b3ea18310a274d684a167e410ca17b266ee58"
echo "$EXPECTED_SHA256  $TARGET_SO" | sha256sum -c - >/dev/null
chmod +x "$TARGET_SO"
echo "  ✅ Asset verified (SHA-256 OK) and ready: $TARGET_SO ($(stat -c%s "$TARGET_SO" 2>/dev/null || stat -f%z "$TARGET_SO") bytes)"
exit 0
