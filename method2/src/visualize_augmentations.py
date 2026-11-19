"""Visualizes the effects of different data augmentation strategies on a sample image."""

import argparse
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def apply_basic_aug(img):
    aug_img = img.copy()
    if random.random() > 0.5:
        aug_img = cv2.flip(aug_img, 1)
    hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.7, 1.3)
    hsv[:, :, 2] *= random.uniform(0.7, 1.3)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return aug_img

def apply_simulated_mosaic(images, out_size=(640, 640)):
    h, w = out_size
    mosaic_img = np.full((h, w, 3), 114, dtype=np.uint8)
    xc, yc = (int(w * 0.5), int(h * 0.5))
    coords = [(0, 0, xc, yc), (xc, 0, w, yc), (0, yc, xc, h), (xc, yc, w, h)]
    for idx, (x1, y1, x2, y2) in enumerate(coords):
        sub_img = images[idx % len(images)]
        sub_resized = cv2.resize(sub_img, (x2 - x1, y2 - y1))
        mosaic_img[y1:y2, x1:x2] = sub_resized
    return mosaic_img

def visualize_augmentations(image_path, output_path='results/augmentation_demo.png'):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if image_path and os.path.exists(image_path):
        base_img = cv2.imread(image_path)
    else:
        base_img = np.full((480, 640, 3), (200, 220, 240), dtype=np.uint8)
        cv2.rectangle(base_img, (0, 240), (640, 480), (140, 200, 230), -1)
        for pos in [(150, 320), (320, 280), (450, 350), (220, 300)]:
            cv2.circle(base_img, pos, 12, (50, 50, 50), -1)
            cv2.ellipse(base_img, (pos[0] + 15, pos[1]), (18, 10), 0, 0, 360, (240, 240, 240), -1)
    img_baseline = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    img_basic = cv2.cvtColor(apply_basic_aug(base_img), cv2.COLOR_BGR2RGB)
    mosaic_raw = apply_simulated_mosaic([base_img, apply_basic_aug(base_img), base_img, apply_basic_aug(base_img)])
    img_mosaic = cv2.cvtColor(mosaic_raw, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_baseline)
    axes[0].set_title('(a) Baseline: Original Beach Scene', fontsize=11, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(img_basic)
    axes[1].set_title('(b) Strategy 1: Basic (Flip + HSV Jitter)', fontsize=11, fontweight='bold')
    axes[1].axis('off')
    axes[2].imshow(img_mosaic)
    axes[2].set_title('(c) Strategy 2: Advanced (4-Image Mosaic)', fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[Success] Generated data augmentation visualization: {output_path}')
MODULE_ROOT = Path(__file__).resolve().parent.parent

def main():
    default_out = str(MODULE_ROOT / 'results' / 'augmentation_demo.png')
    parser = argparse.ArgumentParser(description='Visualize Augmentation Transformations')
    parser.add_argument('--image', type=str, default=None, help='Sample bird image path')
    parser.add_argument('--output', type=str, default=default_out, help='Output figure path')
    args = parser.parse_args()
    visualize_augmentations(args.image, args.output)


if __name__ == '__main__':
    main()
