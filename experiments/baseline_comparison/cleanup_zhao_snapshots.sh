#!/bin/bash
# Keep only the last 3 Zhao EDM snapshots to save disk space.
# Run periodically: watch -n 3600 bash cleanup_zhao_snapshots.sh
SNAP_DIR="/home/fjiang4/Membership-is-Ownership/experiments/baseline_comparison/zhao/cifar10/edm/00000-images-uncond-ddpmpp-edm-gpus1-batch512-fp32"
KEEP=3

cd "$SNAP_DIR" 2>/dev/null || exit 0
snapshots=($(ls -1t network-snapshot-*.pkl 2>/dev/null))
total=${#snapshots[@]}
if [ "$total" -le "$KEEP" ]; then
    echo "Only $total snapshots, keeping all."
    exit 0
fi
delete_count=$((total - KEEP))
echo "Found $total snapshots, deleting $delete_count oldest..."
for f in "${snapshots[@]:$KEEP}"; do
    echo "Removing: $f ($(du -h "$f" | cut -f1))"
    rm "$f"
done
echo "Done. Kept latest $KEEP snapshots."
