"""Generate distorted versions of test images for robustness evaluation.

Noise, blur, low brightness, and occlusion distortions are applied to the images in the test set."""

from pathlib import Path
import cv2
import numpy as np

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "common/dataset"
INPUT_DIR = DATASET / "images/test"
OUTPUT_DIR = ROOT / "exp_robustness/distorted_images"

# Keep results reproducible
rng = np.random.default_rng(42)


def gaussian_noise(image):
    noise = rng.normal(0, 25, image.shape)
    result = image.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_blur(image):
    return cv2.GaussianBlur(image, (9, 9), 2)


def low_brightness(image):
    result = image.astype(np.float32) * 0.5
    return np.clip(result, 0, 255).astype(np.uint8)


def occlusion(image):
    result = image.copy()
    h, w = result.shape[:2]

    # Cover a fixed central area
    x1, y1 = int(w * 0.4), int(h * 0.4)
    x2, y2 = int(w * 0.6), int(h * 0.6)

    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), -1)
    return result


def main():
    methods = {
        "noise": gaussian_noise,
        "blur": gaussian_blur,
        "low_brightness": low_brightness,
        "occlusion": occlusion,
    }

    images = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    print(f"Found {len(images)} test images")

    for method_name, transform in methods.items():
        output = OUTPUT_DIR / method_name
        output.mkdir(parents=True, exist_ok=True)

        for image_path in images:
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping {image_path.name}")
                continue

            distorted = transform(image)
            cv2.imwrite(str(output / image_path.name), distorted)

        print(f"Finished: {method_name}")

    print("All distortions generated successfully!")


if __name__ == "__main__":
    main()
