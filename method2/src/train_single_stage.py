"""Train Single-Stage Detector (YOLOv8)

Loads dataset configuration and augmentation parameters (optional)

Output best.pt"""

import argparse
import os
import sys
import time
import yaml
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    print('[Error] ultralytics is not installed. Run: pip install ultralytics')
    sys.exit(1)
from utils import MODULE_ROOT, get_resolved_data_yaml

def load_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def train_model(args):
    data_yaml = get_resolved_data_yaml(args.data)
    aug_yaml = None
    if args.aug_config:
        aug_yaml = Path(args.aug_config)
        if not aug_yaml.is_absolute():
            aug_yaml = (MODULE_ROOT / aug_yaml).resolve()
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (MODULE_ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 70)
    print('COMP9517 - Single-Stage Detection Training')
    print(f'Base Model:       {args.model}')
    print(f'Dataset Config:   {data_yaml}')
    print(f"Augmentation:     {(aug_yaml if aug_yaml else 'Default')}")
    print(f'Epochs:           {args.epochs}')
    print(f'Batch Size:       {args.batch}')
    print(f'Image Size:       {args.imgsz}')
    print(f'Device:           {args.device}')
    print('=' * 70)
    aug_params = {}
    if aug_yaml and aug_yaml.exists():
        aug_params = load_yaml(aug_yaml) or {}
        print(f'[Info] Loaded {len(aug_params)} augmentation hyperparameters from {aug_yaml.name}')
    model = YOLO(args.model)
    train_kwargs = {'data': str(data_yaml), 'epochs': args.epochs, 'batch': args.batch, 'imgsz': args.imgsz, 'device': args.device, 'project': str(save_dir), 'name': args.exp_name, 'exist_ok': True, 'save': True, 'plots': True, 'verbose': True, 'workers': args.workers}
    train_kwargs.update(aug_params)
    start_time = time.time()
    results = model.train(**train_kwargs)
    total_train_time = time.time() - start_time
    print('\n' + '=' * 70)
    print(f'Training completed in: {total_train_time:.2f} seconds ({total_train_time / 60:.2f} mins)')
    best_weight = save_dir / args.exp_name / 'weights' / 'best.pt'
    print(f'Best weight saved at: {best_weight}')
    print('=' * 70)
    return (results, total_train_time)

def main():
    default_data = str(MODULE_ROOT / 'configs' / 'data.yaml')
    default_save = str(MODULE_ROOT / 'results' / 'train_runs')
    parser = argparse.ArgumentParser(description='Train Single-Stage Detector (YOLOv8) with Augmentation Config')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Base model (e.g. yolov8n.pt, yolov8s.pt)')
    parser.add_argument('--data', type=str, default=default_data, help='Path to data.yaml')
    parser.add_argument('--aug_config', type=str, default=None, help='Path to augmentation yaml config')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution')
    parser.add_argument('--device', type=str, default='0', help='Device (0 or cpu)')
    parser.add_argument('--workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--save_dir', type=str, default=default_save, help='Directory to save runs')
    parser.add_argument('--exp_name', type=str, default='single_stage_baseline', help='Experiment name')
    args = parser.parse_args()
    train_model(args)


if __name__ == '__main__':
    main()
