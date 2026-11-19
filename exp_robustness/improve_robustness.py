"""Denoise images using OpenCV's fastNlMeansDenoisingColored function."""

from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "exp_robustness/distorted_images/noise"
OUTPUT_DIR = ROOT / "exp_robustness/distorted_images/denoised"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def denoise_image(image):
    # Remove noise while preserving image edges
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        10,
        10,
        7,
        21,
    )


def main():
    image_paths = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    print(f"Found {len(image_paths)} noisy images")

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Skipping: {image_path.name}")
            continue

        improved = denoise_image(image)

        output_path = OUTPUT_DIR / image_path.name
        cv2.imwrite(str(output_path), improved)

        if index % 40 == 0:
            print(f"Processed {index}/{len(image_paths)} images")

    print("Denoising completed successfully!")


if __name__ == "__main__":
    main()
