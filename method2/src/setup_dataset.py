"""Setup and verify the beach-bird dataset for YOLOv8 training"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path
MODULE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODULE_ROOT.parent
DATASET_DIR = REPO_ROOT / 'common' / 'dataset'
IMAGES_DIR = DATASET_DIR / 'images'
LABELS_DIR = DATASET_DIR / 'labels'

def check_dataset_status():
    print('=' * 70)
    print('CHECKING BEACH BIRD DATASET INTEGRITY')
    print(f'Target Directory: {DATASET_DIR}')
    print('=' * 70)
    if not DATASET_DIR.exists():
        print(f'[Error] Dataset directory does not exist: {DATASET_DIR}')
        return False
    has_labels = LABELS_DIR.exists()
    has_images = IMAGES_DIR.exists()
    print(f'[*] Labels directory found: {has_labels}')
    print(f'[*] Images directory found: {has_images}')
    if has_images:
        for split in ['train', 'val', 'test']:
            split_dir = IMAGES_DIR / split
            count = len(list(split_dir.glob('*.*'))) if split_dir.exists() else 0
            print(f'    - {split}: {count} image(s)')
        print('\n[Success] Dataset images are properly installed!')
        return True
    else:
        print('\n[Notice] Real image files are not yet detected.')
        print('  -> Kaggle Download Link: https://www.kaggle.com/datasets/liugalhigh/beach-bird')
        print(f"  -> Please extract the 'images' folder into: {DATASET_DIR}/images")
        return False

def generate_tiny_mock_dataset(num_samples=5):
    print('\n[Setup] Generating a tiny mock beach-bird sample set for pipeline verification...')
    for split in ['train', 'val', 'test']:
        split_lbl_dir = LABELS_DIR / split
        split_img_dir = IMAGES_DIR / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        if not split_lbl_dir.exists():
            continue
        lbl_files = list(split_lbl_dir.glob('*.txt'))[:num_samples]
        for lbl_file in lbl_files:
            img_file = split_img_dir / lbl_file.with_suffix('.jpg').name
            if not img_file.exists():
                mock_img = np.full((640, 640, 3), (210, 230, 245), dtype=np.uint8)
                cv2.rectangle(mock_img, (0, 300), (640, 640), (140, 200, 230), -1)
                cv2.imwrite(str(img_file), mock_img)
    print('[Success] Mock samples generated. You can now dry-run training or ablation anytime!')

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Beach Bird Dataset Verification Tool')
    parser.add_argument('--mock', action='store_true', help='Generate tiny mock images for dry-run pipeline test')
    args = parser.parse_args()
    status = check_dataset_status()
    if not status and args.mock:
        generate_tiny_mock_dataset()


if __name__ == '__main__':
    main()
