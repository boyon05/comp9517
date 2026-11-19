"""Evaluate the robustness of Method 3 (both models) on distorted images."""

from pathlib import Path
import csv
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exp_analysis_demo"))
sys.path.insert(0, str(ROOT))

from robustness import run_method3
from eval_all import load_gt_yolo
from common.metrics import evaluate

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

MODELS = {
    "default": (
        ROOT / "method3/output/standard.pth",
        ROOT / "method3/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",
    ),
    "stride4": (
        ROOT / "method3/output/stride4.pth",
        ROOT / "method3/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco_stride4.yml",
    ),
}

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    image_dir = DATASET / "images/test"
    label_dir = DATASET / "labels/test"
    manifest = DATASET / "dataset_manifest.csv"

    gt = load_gt_yolo(str(label_dir), str(manifest), str(image_dir))
    names = sorted(
        p.name for p in image_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    print(f"Test images: {len(names)}", flush=True)

    rows = []

    for model_name, (weights, config) in MODELS.items():
        for condition in CONDITIONS:
            folder = image_dir if condition == "original" else DISTORTED / condition

            print(f"\nEvaluating {model_name}: {condition}", flush=True)

            images = {
                name: Image.open(folder / name).convert("RGB")
                for name in names
            }

            predictions = run_method3(
                images, str(weights), str(config)
            )

            result = evaluate(
                gt,
                predictions,
                iou_thr=0.5,
                name=f"{model_name}/{condition}",
            )

            row = {
                "model": model_name,
                "condition": condition,
                "map50": round(result["ap50"], 4),
                "precision": round(result["best_precision"], 4),
                "recall": round(result["best_recall"], 4),
            }
            rows.append(row)

            print(
                f"{model_name}/{condition}: "
                f"mAP@0.5={row['map50']:.4f}",
                flush=True,
            )

            with open(OUTPUT / "method3_robustness_metrics.csv",
                      "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)

    print("\nMethod 3 evaluation completed!")

if __name__ == "__main__":
    main()
