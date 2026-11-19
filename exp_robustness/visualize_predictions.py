
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "common/archive/Beach Bird Detection Dataset v1.0"
DISTORTED = ROOT / "exp_robustness/distorted_images"
MODEL_PATH = ROOT / "method2_single_stage/weights/best.pt"
OUTPUT_DIR = ROOT / "exp_robustness/results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose one test image
images = sorted((DATASET / "images/test").glob("*.jpg"))
image_name = images[1].name

print(f"Using image: {image_name}")

paths = {
    "Original": DATASET / "images/test" / image_name,
    "Gaussian Noise": DISTORTED / "noise" / image_name,
    "Denoised Noise": DISTORTED / "denoised" / image_name,
}

model = YOLO(str(MODEL_PATH))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (title, image_path) in zip(axes, paths.items()):
    result = model.predict(
        source=str(image_path),
        conf=0.25,
        imgsz=640,
        verbose=False,
    )[0]

    annotated = result.plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    ax.imshow(annotated)
    ax.set_title(
        f"{title}\nDetections: {len(result.boxes)}",
        fontsize=12,
    )
    ax.axis("off")

plt.tight_layout()

output_path = OUTPUT_DIR / "prediction_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")
