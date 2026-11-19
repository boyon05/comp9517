"""Compares the performance of different data augmentation strategies

For each strategy, it trains a YOLOv8s model on the same dataset and evaluates its performance.

The results are saved in a CSV file and a summary chart is generated."""

import argparse
import os
import sys
import time
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    print('[Error] ultralytics is required. Install via: pip install ultralytics')
    sys.exit(1)
from utils import MODULE_ROOT, get_resolved_data_yaml

def load_yaml(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def benchmark_inference_speed(model, test_img_path=None, imgsz=640, device='0', iterations=100):
    import numpy as np
    import torch
    if test_img_path and os.path.exists(test_img_path):
        dummy_input = test_img_path
    else:
        dummy_input = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(10):
        _ = model.predict(dummy_input, imgsz=imgsz, device=device, verbose=False)
    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = model.predict(dummy_input, imgsz=imgsz, device=device, verbose=False)
        if torch.cuda.is_available() and device != 'cpu':
            torch.cuda.synchronize()
    total_time = time.time() - start
    avg_ms = total_time / iterations * 1000.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
    return (avg_ms, fps)

def run_ablation_pipeline(args):
    data_yaml = Path(get_resolved_data_yaml(args.data))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (MODULE_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = [{'name': 'Strategy_0_Baseline', 'desc': 'Baseline (No/Minimal Aug)', 'config': MODULE_ROOT / 'configs' / 'aug_baseline.yaml'}, {'name': 'Strategy_1_Basic', 'desc': 'Basic (Flip + HSV Jitter + Scale)', 'config': MODULE_ROOT / 'configs' / 'aug_basic.yaml'}, {'name': 'Strategy_2_Advanced', 'desc': 'Advanced (Mosaic + Mixup + Scale)', 'config': MODULE_ROOT / 'configs' / 'aug_advanced.yaml'}]
    records = []
    print('\n' + '#' * 70)
    print('STARTING DATA AUGMENTATION ABLATION STUDY FOR BEACH BIRD DETECTION')
    print(f'Base Model:       {args.model}')
    print(f'Dataset Config:   {data_yaml}')
    print(f'Epochs per Exp:   {args.epochs}')
    print(f'Output Directory: {output_dir}')
    print('#' * 70 + '\n')
    for strat in strategies:
        print('\n' + '=' * 60)
        print(f"Running Experiment: {strat['name']} [{strat['desc']}]")
        print('=' * 60)
        aug_cfg_path = Path(strat['config'])
        aug_params = load_yaml(aug_cfg_path) if aug_cfg_path.exists() else {}
        model = YOLO(args.model)
        train_args = {'data': str(data_yaml), 'epochs': args.epochs, 'batch': args.batch, 'imgsz': args.imgsz, 'device': args.device, 'project': str(output_dir / 'runs'), 'name': strat['name'], 'exist_ok': True, 'save': True, 'plots': True, 'verbose': False}
        train_args.update(aug_params)
        train_start = time.time()
        model.train(**train_args)
        train_time = time.time() - train_start
        best_weight = output_dir / 'runs' / strat['name'] / 'weights' / 'best.pt'
        if not best_weight.exists():
            raise FileNotFoundError(f"Training failed: {best_weight} was not generated.")
        eval_model = YOLO(str(best_weight))
        val_metrics = eval_model.val(data=str(data_yaml), split='val', imgsz=args.imgsz, device=args.device, verbose=False)
        precision = float(val_metrics.box.mp)
        recall = float(val_metrics.box.mr)
        map50 = float(val_metrics.box.map50)
        map50_95 = float(val_metrics.box.map)
        lat_ms, fps = benchmark_inference_speed(eval_model, imgsz=args.imgsz, device=args.device)
        records.append({'Strategy': strat['name'], 'Description': strat['desc'], 'mAP@0.5': round(map50, 4), 'mAP@0.5:0.95': round(map50_95, 4), 'Precision': round(precision, 4), 'Recall': round(recall, 4), 'Train_Time_min': round(train_time / 60.0, 2), 'Latency_ms': round(lat_ms, 2), 'FPS': round(fps, 1)})
    df = pd.DataFrame(records)
    csv_path = output_dir / 'ablation_results.csv'
    df.to_csv(csv_path, index=False)
    md_path = output_dir / 'ablation_results.md'
    try:
        df.to_markdown(md_path, index=False)
    except Exception:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(df.to_string(index=False))
    print('\n' + '=' * 70)
    print('ABLATION STUDY COMPLETED! Summary Results:')
    print('=' * 70)
    print(df.to_string(index=False))
    print(f'\nSaved CSV: {csv_path}')
    plot_ablation_results(df, output_dir)

def plot_ablation_results(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(df))
    width = 0.35
    axes[0].bar([i - width / 2 for i in x], df['mAP@0.5'], width=width, label='mAP@0.5', color='#1f77b4')
    axes[0].bar([i + width / 2 for i in x], df['mAP@0.5:0.95'], width=width, label='mAP@0.5:0.95', color='#aec7e8')
    axes[0].set_ylabel('Detection Precision Score', fontsize=11)
    axes[0].set_title('Detection Accuracy Across Augmentation Strategies', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([d.split(' (')[0] for d in df['Description']], rotation=15, ha='right')
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend()
    axes[1].bar([i - width / 2 for i in x], df['Precision'], width=width, label='Precision', color='#2ca02c')
    axes[1].bar([i + width / 2 for i in x], df['Recall'], width=width, label='Recall', color='#ff7f0e')
    axes[1].set_ylabel('Score', fontsize=11)
    axes[1].set_title('Precision vs Recall Trade-off', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d.split(' (')[0] for d in df['Description']], rotation=15, ha='right')
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].legend()
    plt.tight_layout()
    chart_path = output_dir / 'ablation_comparison_chart.png'
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f'Saved ablation comparison figure: {chart_path}')

def main():
    default_data = str(MODULE_ROOT / 'configs' / 'data.yaml')
    default_out = str(MODULE_ROOT / 'results' / 'ablation_study')
    parser = argparse.ArgumentParser(description='Run Full Data Augmentation Ablation Pipeline')
    parser.add_argument('--model', type=str, default='yolov8s.pt', help='Single-stage base model')
    parser.add_argument('--data', type=str, default=default_data, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per experiment (default 30 for ablation)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input resolution')
    parser.add_argument('--device', type=str, default='0', help='Device (0 or cpu)')
    parser.add_argument('--output_dir', type=str, default=default_out, help='Output results dir')
    args = parser.parse_args()
    run_ablation_pipeline(args)


if __name__ == '__main__':
    main()
