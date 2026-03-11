#!/usr/bin/env bash
# setup_storage.sh — Create /data/short/fjiang4/ layout and symlink into project
# Idempotent: safe to re-run; skips existing symlinks, only migrates real dirs.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STORAGE_ROOT="/data/short/fjiang4"
EXP_NAME="sd_watermark_comp"

echo "=== Storage Setup ==="
echo "Project root : $PROJECT_ROOT"
echo "Storage root : $STORAGE_ROOT"
echo ""

# --------------------------------------------------------------------------- #
# 1. Create directory tree under /data/short/fjiang4/
# --------------------------------------------------------------------------- #
echo "[1/4] Creating directory tree on $STORAGE_ROOT ..."

dirs=(
    "$STORAGE_ROOT/models/sd-v1-4"
    "$STORAGE_ROOT/models/sd-v1-4-lora"
    "$STORAGE_ROOT/models/sleepermark-unet"
    "$STORAGE_ROOT/data/coco2014"
    "$STORAGE_ROOT/data/splits"
    "$STORAGE_ROOT/data/lora_train_dir"
    "$STORAGE_ROOT/data/sleepermark_train_images"
    "$STORAGE_ROOT/experiments/$EXP_NAME/scores"
    "$STORAGE_ROOT/experiments/$EXP_NAME/figures"
    "$STORAGE_ROOT/experiments/$EXP_NAME/logs"
)

for d in "${dirs[@]}"; do
    mkdir -p "$d"
done
echo "  Done."

# --------------------------------------------------------------------------- #
# 2. Migrate existing real directories from project root (if any)
# --------------------------------------------------------------------------- #
echo "[2/4] Migrating existing data from project root ..."

migrate() {
    local src="$1" dst="$2"
    if [ -d "$src" ] && [ ! -L "$src" ]; then
        echo "  Migrating $src -> $dst"
        rsync -a --remove-source-files "$src/" "$dst/"
        # Remove empty source dir tree left by rsync
        find "$src" -type d -empty -delete 2>/dev/null || true
    fi
}

migrate "$PROJECT_ROOT/models"   "$STORAGE_ROOT/models"
migrate "$PROJECT_ROOT/data"     "$STORAGE_ROOT/data"

# Experiment-level large dirs
for sub in scores figures logs; do
    migrate "$PROJECT_ROOT/experiments/$EXP_NAME/$sub" \
            "$STORAGE_ROOT/experiments/$EXP_NAME/$sub"
done

echo "  Done."

# --------------------------------------------------------------------------- #
# 3. Create symlinks: project paths -> /data/short/fjiang4/
# --------------------------------------------------------------------------- #
echo "[3/4] Creating symlinks ..."

make_link() {
    local link="$1" target="$2"
    if [ -L "$link" ]; then
        echo "  SKIP (already symlink): $link -> $(readlink "$link")"
    elif [ -e "$link" ]; then
        echo "  WARNING: $link exists and is NOT a symlink — skipping to be safe"
    else
        ln -sfn "$target" "$link"
        echo "  CREATED: $link -> $target"
    fi
}

# Top-level symlinks
make_link "$PROJECT_ROOT/models" "$STORAGE_ROOT/models"
make_link "$PROJECT_ROOT/data"   "$STORAGE_ROOT/data"

# Experiment output symlinks
for sub in scores figures logs; do
    make_link "$PROJECT_ROOT/experiments/$EXP_NAME/$sub" \
              "$STORAGE_ROOT/experiments/$EXP_NAME/$sub"
done

echo "  Done."

# --------------------------------------------------------------------------- #
# 4. Summary
# --------------------------------------------------------------------------- #
echo ""
echo "=== Summary ==="
echo "Symlinks:"
for p in models data; do
    if [ -L "$PROJECT_ROOT/$p" ]; then
        printf "  %-40s -> %s\n" "$p/" "$(readlink "$PROJECT_ROOT/$p")"
    fi
done
for sub in scores figures logs; do
    link="$PROJECT_ROOT/experiments/$EXP_NAME/$sub"
    if [ -L "$link" ]; then
        printf "  %-40s -> %s\n" "experiments/$EXP_NAME/$sub/" "$(readlink "$link")"
    fi
done

echo ""
echo "Storage usage:"
du -sh "$STORAGE_ROOT/models" "$STORAGE_ROOT/data" \
       "$STORAGE_ROOT/experiments/$EXP_NAME" 2>/dev/null || true
echo ""
echo "=== Done ==="
