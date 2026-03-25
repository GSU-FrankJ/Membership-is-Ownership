#!/usr/bin/env python3
"""Create data splits for Phase 11 (3-point verification)."""
import json
import hashlib
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    split_path = os.path.join(PROJECT_ROOT, "data/splits/split_seed42.json")
    sub_split_path = os.path.join(PROJECT_ROOT, "data/splits/sub_split_1000.json")

    with open(split_path) as f:
        full_split = json.load(f)
    with open(sub_split_path) as f:
        a6_split = json.load(f)

    # W = A6's 1000 training members
    w_members = a6_split["members"]
    w_ids = {m["image_id"] for m in w_members}

    # --- W-only eval split (for verification scoring) ---
    # Include 1000 non-members for robust statistical testing
    rng = random.Random(42)
    nonmembers_shuffled = list(full_split["nonmembers"])
    rng.shuffle(nonmembers_shuffled)
    sanity_nonmembers = nonmembers_shuffled[:1000]

    w_eval = {
        "members": w_members,
        "nonmembers": sanity_nonmembers,
        "meta": {
            "description": "W-only eval (1000 A6 members + 1000 non-members)",
            "member_count": len(w_members),
            "nonmember_count": len(sanity_nonmembers),
        }
    }
    out_w = os.path.join(PROJECT_ROOT, "data/splits/phase11_w_only.json")
    os.makedirs(os.path.dirname(out_w), exist_ok=True)
    with open(out_w, "w") as f:
        json.dump(w_eval, f, indent=2)
    print(f"W eval split: {len(w_members)} members, {len(sanity_nonmembers)} non-members -> {out_w}")

    # SHA-256 of sorted W image IDs (Algorithm 2 hash commitment)
    w_ids_sorted = sorted(str(m["image_id"]) for m in w_members)
    w_hash = hashlib.sha256("\n".join(w_ids_sorted).encode()).hexdigest()
    print(f"SHA-256(sort(W)): {w_hash}")

    # --- B1 training split (1000 disjoint COCO images from remaining 9000) ---
    remaining = [m for m in full_split["members"] if m["image_id"] not in w_ids]
    rng2 = random.Random(42)
    rng2.shuffle(remaining)
    b1_members = remaining[:1000]

    b1_train = {
        "members": b1_members,
        "nonmembers": [],
        "meta": {
            "description": "Phase 11: B1 adversary training (1000 COCO images disjoint from W)",
            "member_count": len(b1_members),
        }
    }
    out_b1 = os.path.join(PROJECT_ROOT, "data/splits/phase11_b1_train.json")
    with open(out_b1, "w") as f:
        json.dump(b1_train, f, indent=2)
    print(f"B1 train split: {len(b1_members)} images -> {out_b1}")

    # Verify disjoint
    b1_ids = {m["image_id"] for m in b1_members}
    assert len(w_ids & b1_ids) == 0, f"OVERLAP: {len(w_ids & b1_ids)} shared images!"
    print(f"Disjoint check: W ∩ B1 = empty set (verified)")

    # --- B1 training dir (symlinks + metadata.jsonl) ---
    b1_dir = os.path.join(PROJECT_ROOT, "data/ablation_train_dir_b1")
    os.makedirs(b1_dir, exist_ok=True)
    coco_train_dir = os.path.join(PROJECT_ROOT, "data/coco2014/train2014")
    metadata_lines = []
    created = 0
    for entry in b1_members:
        src = os.path.join(coco_train_dir, entry["file_name"])
        dst = os.path.join(b1_dir, entry["file_name"])
        if not os.path.exists(dst):
            os.symlink(src, dst)
            created += 1
        caption = entry.get("caption", "a photograph")
        metadata_lines.append(json.dumps({"file_name": entry["file_name"], "text": caption}))
    with open(os.path.join(b1_dir, "metadata.jsonl"), "w") as f:
        f.write("\n".join(metadata_lines) + "\n")
    print(f"B1 train dir: {created} new symlinks, {len(b1_members)} total -> {b1_dir}")

    # --- B2 training dir (500 SleeperMark synthetic images) ---
    sm_dir = os.path.join(PROJECT_ROOT, "data/sleepermark_train_images")
    b2_dir = os.path.join(PROJECT_ROOT, "data/ablation_train_dir_b2")
    os.makedirs(b2_dir, exist_ok=True)
    sm_files = sorted([f for f in os.listdir(sm_dir) if f.endswith(".png")])[:500]
    metadata_lines = []
    created = 0
    for fname in sm_files:
        src = os.path.join(sm_dir, fname)
        dst = os.path.join(b2_dir, fname)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            created += 1
        metadata_lines.append(json.dumps({"file_name": fname, "text": "a photograph"}))
    with open(os.path.join(b2_dir, "metadata.jsonl"), "w") as f:
        f.write("\n".join(metadata_lines) + "\n")
    print(f"B2 train dir: {created} new symlinks, {len(sm_files)} total -> {b2_dir}")

    print(f"\n=== Phase 11 splits complete ===")
    print(f"W hash commitment: {w_hash}")


if __name__ == "__main__":
    main()
