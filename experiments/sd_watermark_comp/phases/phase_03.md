# Phase 03 — Construct Member / Non-member Split

## Objective
Create a deterministic, reproducible split of COCO2014 into:
- **Members** (10,000 train images) — the "private data watermark"
- **Non-members** (10,000 val images) — guaranteed never seen during fine-tuning

## Steps

### 1. Build split file
```python
import json, random, hashlib, os

random.seed(42)

with open("data/coco2014/annotations/captions_train2014.json") as f:
    train_data = json.load(f)
with open("data/coco2014/annotations/captions_val2014.json") as f:
    val_data = json.load(f)

def dedup_captions(annotations):
    seen = {}
    for ann in annotations:
        if ann["image_id"] not in seen:
            seen[ann["image_id"]] = ann["caption"]
    return seen

train_images = dedup_captions(train_data["annotations"])
val_images = dedup_captions(val_data["annotations"])

all_train_ids = sorted(train_images.keys())
random.shuffle(all_train_ids)
member_ids = all_train_ids[:10000]

all_val_ids = sorted(val_images.keys())
random.shuffle(all_val_ids)
nonmember_ids = all_val_ids[:10000]

split = {
    "seed": 42,
    "member_count": len(member_ids),
    "nonmember_count": len(nonmember_ids),
    "members": [
        {"image_id": iid, "caption": train_images[iid],
         "file_name": f"COCO_train2014_{iid:012d}.jpg"}
        for iid in member_ids
    ],
    "nonmembers": [
        {"image_id": iid, "caption": val_images[iid],
         "file_name": f"COCO_val2014_{iid:012d}.jpg"}
        for iid in nonmember_ids
    ]
}

os.makedirs("data/splits", exist_ok=True)
with open("data/splits/split_seed42.json", "w") as f:
    json.dump(split, f, indent=2)

with open("data/splits/split_seed42.json", "rb") as f:
    md5 = hashlib.md5(f.read()).hexdigest()
print(f"Split saved. MD5: {md5}")
print(f"Members: {len(member_ids)}, Non-members: {len(nonmember_ids)}")
```

### 2. Verify all images exist on disk
```python
import os, json

with open("data/splits/split_seed42.json") as f:
    split = json.load(f)

missing = 0
for e in split["members"]:
    if not os.path.exists(f"data/coco2014/train2014/{e['file_name']}"):
        missing += 1
for e in split["nonmembers"]:
    if not os.path.exists(f"data/coco2014/val2014/{e['file_name']}"):
        missing += 1
print(f"Missing: {missing} (must be 0)")
assert missing == 0
```

### 3. Create symlink directory for diffusers training script
```python
import json, os

with open("data/splits/split_seed42.json") as f:
    split = json.load(f)

os.makedirs("data/lora_train_dir", exist_ok=True)
metadata = []
for entry in split["members"]:
    src = os.path.abspath(f"data/coco2014/train2014/{entry['file_name']}")
    dst = f"data/lora_train_dir/{entry['file_name']}"
    if not os.path.exists(dst):
        os.symlink(src, dst)
    metadata.append({"file_name": entry["file_name"], "text": entry["caption"]})

with open("data/lora_train_dir/metadata.jsonl", "w") as f:
    for item in metadata:
        f.write(json.dumps(item) + "\n")

print(f"Created {len(metadata)} symlinks + metadata.jsonl")
```

## Sanity Check
- [ ] split_seed42.json contains exactly 10,000 members and 10,000 non-members
- [ ] Zero missing image files
- [ ] No overlap between member and non-member image_ids
- [ ] `data/lora_train_dir/` has 10,000 symlinks + metadata.jsonl
- [ ] MD5 recorded

## Update STATE.md
Record split md5 and data counts. Set Phase 03 = ✅ DONE.