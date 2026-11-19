"""Converts YOLO format annotations to COCO format for RT-DETRv2

Output: coco.json"""

import json, sys
from pathlib import Path
from PIL import Image

def convert(images_dir, labels_dir, out_json, image_exts=(".jpg", ".jpeg")):
    images, annotations = [], []
    ann_id, img_id = 1, 1
    for img_path in sorted(Path(images_dir).iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            w, h = 512, 512
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                xc, yc, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                x = (xc - bw / 2) * w
                y = (yc - bh / 2) * h
                bw_abs, bh_abs = bw * w, bh * h
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "bbox": [round(x, 2), round(y, 2), round(bw_abs, 2), round(bh_abs, 2)],
                    "area": round(bw_abs * bh_abs, 2), "iscrowd": 0,
                })
                ann_id += 1
        img_id += 1
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "bird", "supercategory": "bird"}]}
    out = Path(out_json); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(out, "w"))
    print(f"{out}: {len(images)} imgs, {len(annotations)} boxes")

if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3])