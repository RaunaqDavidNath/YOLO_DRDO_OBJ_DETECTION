"""Upload the aerial dataset (images + YOLO labels) to Roboflow. See README for usage."""

import os
from pathlib import Path
from roboflow import Roboflow

# Configuration
API_KEY       = os.environ.get("ROBOFLOW_API_KEY")  # read from environment, never hardcode
WORKSPACE     = "drdo-pylxb"
PROJECT_NAME  = "uav-vehicle-detection-9t43s"
DATASET_ROOT  = Path("aerial dataset")

if not API_KEY:
    raise SystemExit('ROBOFLOW_API_KEY is not set. See README for setup.')

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_NAME)

SPLITS = ["train", "val", "test"]
total_uploaded = 0
total_skipped  = 0

for split in SPLITS:
    img_dir = DATASET_ROOT / "images" / split
    lbl_dir = DATASET_ROOT / "labels" / split

    images = sorted(img_dir.glob("*.jpg"))
    print(f"\n[{split}] {len(images)} images found")

    for i, img_path in enumerate(images, 1):
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        if not lbl_path.exists():
            total_skipped += 1
            continue

        print(f"  [{i}/{len(images)}] Uploading {img_path.name}...", flush=True)
        try:
            project.upload(
                image_path=str(img_path),
                annotation_path=str(lbl_path),
                annotation_overwrite=True,
                split=split,
                num_retry_uploads=3,
            )
            total_uploaded += 1
            print(f"  [{i}/{len(images)}] OK", flush=True)
        except Exception as e:
            print(f"  [{i}/{len(images)}] ERROR: {e}", flush=True)
            total_skipped += 1

print(f"\nDone. Uploaded: {total_uploaded}  Skipped: {total_skipped}")
