"""Evaluate the robustness of Method 2 on distorted images."""

from pathlib import Path
import csv
import sys

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.metrics import evaluate

sys.path.insert(0, str(ROOT / "exp_analysis_demo"))
from eval_all import load_gt_yolo

DATASET = ROOT / "common/dataset"
DISTORTED = ROOT / "exp_robustness/distorted_images"
OUTPUT = ROOT / "exp_robustness/results"

CONDITIONS = [
    "original",
    "noise",
    "blur",
    "low_brightness",
    "occlusion",
    "denoised",
]

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    image_dir = DATASET / "images/test"
    gt = load_gt_yolo(
        str(DATASET / "labels/test"),
        str(DATASET / "dataset_manifest.csv"),
        str(image_dir),
    )

    names = sorted(
        p.name for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    model = YOLO(str(ROOT / "method2/weights/best.pt"))
    rows = []

    for condition in CONDITIONS:
        folder = image_dir if condition == "original" else DISTORTED / condition

        print(f"\nEvaluating Method 2: {condition}", flush=True)

        preds = {}
        for start in range(0, len(names), 8):
            chunk = names[start:start + 8]
            images = [
                str(folder / name)
                for name in chunk
            ]

            results = model.predict(
                images,
                conf=0.01,
                iou=0.7,
                imgsz=640,
                device="cpu",
                verbose=False,
            )

            for name, result in zip(chunk, results):
                preds[name] = {
                    "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
                    "scores": result.boxes.conf.cpu().numpy().tolist(),
                }

        result = evaluate(
            gt,
            preds,
            iou_thr=0.5,
            name=f"method2/{condition}",
        )

        row = {
            "model": "method2",
            "condition": condition,
            "map50": round(result["ap50"], 4),
            "precision": round(result["best_precision"], 4),
            "recall": round(result["best_recall"], 4),
        }

        rows.append(row)

        print(
            f"Method 2/{condition}: mAP@0.5={row['map50']:.4f}",
            flush=True,
        )

        with open(OUTPUT / "method2_unified_metrics.csv",
                  "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)

    print("\nMethod 2 evaluation completed!")

if __name__ == "__main__":
    main()
