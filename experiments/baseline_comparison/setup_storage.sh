#!/usr/bin/env bash
# setup_storage.sh — Migrate baseline_comparison large dirs to /data/short/fjiang4/
# Idempotent: safe to re-run; skips existing symlinks.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STORAGE_ROOT="/data/short/fjiang4"
EXP_NAME="baseline_comparison"
EXP_SRC="$PROJECT_ROOT/experiments/$EXP_NAME"
EXP_DST="$STORAGE_ROOT/experiments/$EXP_NAME"

echo "=== Storage Setup: $EXP_NAME ==="
echo "Project root : $PROJECT_ROOT"
echo "Storage root : $STORAGE_ROOT"
echo ""

# --------------------------------------------------------------------------- #
# 1. Create directory tree under /data/short/fjiang4/
# --------------------------------------------------------------------------- #
echo "[1/4] Creating directory tree on $STORAGE_ROOT ..."

dirs=(
    # Top-level (may already exist from sd_watermark_comp setup)
    "$STORAGE_ROOT/models"
    "$STORAGE_ROOT/data"
    # Experiment output dirs
    "$EXP_DST/scores"
    "$EXP_DST/figures"
    "$EXP_DST/logs"
    "$EXP_DST/results"
    "$EXP_DST/robustness"
    "$EXP_DST/wdm"
    "$EXP_DST/wdm_repo"
    "$EXP_DST/watermarkdm_repo"
    "$EXP_DST/zhao"
)

for d in "${dirs[@]}"; do
    mkdir -p "$d"
done
echo "  Done."

# --------------------------------------------------------------------------- #
# 2. Migrate existing real directories to /data/short/
# --------------------------------------------------------------------------- #
echo "[2/4] Migrating real directories ..."

migrate() {
    local src="$1" dst="$2"
    if [ -d "$src" ] && [ ! -L "$src" ]; then
        local size
        size=$(du -sh "$src" 2>/dev/null | cut -f1)
        echo "  Migrating $src ($size) -> $dst"
        rsync -a "$src/" "$dst/"
        rm -rf "$src"
        echo "    Removed original."
    elif [ -L "$src" ]; then
        echo "  SKIP (already symlink): $src"
    fi
}

# Top-level dirs (likely already symlinks from sd_watermark_comp setup)
migrate "$PROJECT_ROOT/models" "$STORAGE_ROOT/models"
migrate "$PROJECT_ROOT/data"   "$STORAGE_ROOT/data"

# Experiment-level large dirs
for sub in scores figures logs results robustness wdm wdm_repo watermarkdm_repo zhao; do
    migrate "$EXP_SRC/$sub" "$EXP_DST/$sub"
done

echo "  Done."

# --------------------------------------------------------------------------- #
# 3. Create symlinks
# --------------------------------------------------------------------------- #
echo "[3/4] Creating symlinks ..."

make_link() {
    local link="$1" target="$2"
    if [ -L "$link" ]; then
        echo "  SKIP (already symlink): $link -> $(readlink "$link")"
    elif [ -e "$link" ]; then
        echo "  WARNING: $link exists and is NOT a symlink — skipping"
    else
        ln -sfn "$target" "$link"
        echo "  CREATED: $link -> $target"
    fi
}

# Top-level (likely already set)
make_link "$PROJECT_ROOT/models" "$STORAGE_ROOT/models"
make_link "$PROJECT_ROOT/data"   "$STORAGE_ROOT/data"

# Experiment output dirs
for sub in scores figures logs results robustness wdm wdm_repo watermarkdm_repo zhao; do
    make_link "$EXP_SRC/$sub" "$EXP_DST/$sub"
done

echo "  Done."

# --------------------------------------------------------------------------- #
# 4. Summary
# --------------------------------------------------------------------------- #
echo ""
echo "=== Summary ==="
echo "Top-level symlinks:"
for p in models data; do
    link="$PROJECT_ROOT/$p"
    [ -L "$link" ] && printf "  %-30s -> %s\n" "$p/" "$(readlink "$link")"
done

echo ""
echo "Experiment symlinks (baseline_comparison):"
for sub in scores figures logs results robustness wdm wdm_repo watermarkdm_repo zhao; do
    link="$EXP_SRC/$sub"
    [ -L "$link" ] && printf "  %-30s -> %s\n" "$sub/" "$(readlink "$link")"
done

echo ""
echo "Storage usage on /data/short:"
du -sh "$EXP_DST"/*/ 2>/dev/null | sort -rh
echo ""
du -sh "$EXP_DST" 2>/dev/null
echo ""
echo "=== Done ==="
