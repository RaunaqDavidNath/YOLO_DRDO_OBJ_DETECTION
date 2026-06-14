"""Build a Roboflow-ready zip from a 2000-image train subset plus full val/test. See README for usage."""

import os, random, shutil, zipfile
from pathlib import Path

DATASET   = Path("aerial dataset")
OUT_ZIP   = Path("roboflow_upload.zip")
TRAIN_N   = 2000   # images to sample from train split
SEED      = 42

random.seed(SEED)

# Roboflow expects 'valid' not 'val'
SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}

tmp = Path("/tmp/rf_subset")
if tmp.exists():
    shutil.rmtree(tmp)

total_added = 0

for src_split, dst_split in SPLIT_MAP.items():
    img_dir = DATASET / "images" / src_split
    lbl_dir = DATASET / "labels" / src_split

    all_imgs = sorted(img_dir.glob("*.jpg"))

    if src_split == "train":
        imgs = random.sample(all_imgs, min(TRAIN_N, len(all_imgs)))
        print(f"[train] Sampling {len(imgs)} of {len(all_imgs)} images")
    else:
        imgs = all_imgs
        print(f"[{src_split}] Using all {len(imgs)} images")

    dst_img_dir = tmp / dst_split / "images"
    dst_lbl_dir = tmp / dst_split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in imgs:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        shutil.copy(img_path, dst_img_dir / img_path.name)
        shutil.copy(lbl_path, dst_lbl_dir / lbl_path.name)
        total_added += 1

# Write data.yaml
yaml_content = (
    "train: train/images\n"
    "val: valid/images\n"
    "test: test/images\n"
    "nc: 3\n"
    "names: ['car', 'bus', 'truck']\n"
)
(tmp / "data.yaml").write_text(yaml_content)

print(f"\nZipping {total_added} images → {OUT_ZIP} ...")
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(tmp.rglob("*")):
        if f.is_file():
            zf.write(f, f.relative_to(tmp))

shutil.rmtree(tmp)
size_gb = OUT_ZIP.stat().st_size / 1e9
print(f"Done! {OUT_ZIP} — {size_gb:.2f} GB")
