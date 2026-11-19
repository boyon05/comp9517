"""Evaluate a trained model on the test set and extract examples

Ultralytics validation metrics: mAP@0.5, mAP@0.5:0.95, Precision, Recall, Inference Speed

Output: Annotated images, summary metrics text file

Used for augmentation ablation study across YOLOv8s models ONLY"""

import argparse
import sys
import os
import cv2
import numpy as np
import yaml
from pathlib import Path
try:
    from ultralytics import YOLO
except ImportError:
    print('[Error] ultralytics is required. Install via: pip install ultralytics')
    sys.exit(1)
from utils import MODULE_ROOT, get_resolved_data_yaml

def compute_iou_matrix(boxes1, boxes2):
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, a_min=1e-06, a_max=None)

def load_gt_boxes(label_path, img_w, img_h):
    if not label_path.exists():
        return np.zeros((0, 4))
    boxes = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, cx, cy, w, h = map(float, parts[:5])
                x1 = max(0, (cx - w / 2) * img_w)
                y1 = max(0, (cy - h / 2) * img_h)
                x2 = min(img_w, (cx + w / 2) * img_w)
                y2 = min(img_h, (cy + h / 2) * img_h)
                boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4))

def analyze_and_draw_cases(image_path, label_path, preds, iou_thresh=0.45):
    img = cv2.imread(str(image_path))
    if img is None:
        return (None, 0, 0, 0)
    h, w = img.shape[:2]
    gt_boxes = load_gt_boxes(label_path, w, h)
    pred_boxes = preds.boxes.xyxy.cpu().numpy() if len(preds.boxes) > 0 else np.zeros((0, 4))
    pred_confs = preds.boxes.conf.cpu().numpy() if len(preds.boxes) > 0 else np.zeros(0)
    iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
    matched_gt = set()
    tp_count, fp_count, fn_count = (0, 0, 0)
    annotated = img.copy()
    for i, pbox in enumerate(pred_boxes):
        conf = pred_confs[i]
        matched = False
        if len(gt_boxes) > 0:
            best_gt_idx = np.argmax(iou_mat[i])
            if iou_mat[i, best_gt_idx] >= iou_thresh and best_gt_idx not in matched_gt:
                matched = True
                matched_gt.add(best_gt_idx)
        x1, y1, x2, y2 = map(int, pbox)
        if matched:
            tp_count += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(annotated, f'Bird {conf:.2f}', (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
        else:
            fp_count += 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f'FP {conf:.2f}', (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    for j, gbox in enumerate(gt_boxes):
        if j not in matched_gt:
            fn_count += 1
            gx1, gy1, gx2, gy2 = map(int, gbox)
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 140, 255), 2)
            cv2.putText(annotated, 'FN (Missed)', (gx1, max(gy1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 140, 255), 1)
    return (annotated, tp_count, fp_count, fn_count)

def evaluate_and_extract_cases(model_path, data_yaml, split='test', conf_thresh=0.25, output_dir=None):
    data_yaml = get_resolved_data_yaml(data_yaml)
    if output_dir is None:
        out_dir = MODULE_ROOT / 'results' / 'evaluation'
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    success_dir = out_dir / 'successful_cases'
    failure_dir = out_dir / 'failure_cases'
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)
    print('=' * 75)
    print('COMP9517: Evaluation & Failure Case Analysis')
    print(f'Model Path:         {model_path}')
    print(f'Dataset Config:     {data_yaml}')
    print(f'Evaluation Split:   {split.upper()} (Strict independent benchmark)')
    print(f'Confidence Thresh:  {conf_thresh}')
    print('=' * 75)
    model = YOLO(model_path)
    metrics = model.val(data=str(data_yaml), split=split, conf=conf_thresh, verbose=True)
    print('\n' + '=' * 45)
    print(f'QUANTITATIVE RESULTS ON {split.upper()} SET:')
    print(f'mAP@0.5:       {metrics.box.map50:.4f}  (Core assessment metric)')
    print(f'mAP@0.5:0.95:  {metrics.box.map:.4f}')
    print(f'Precision:     {metrics.box.mp:.4f}')
    print(f'Recall:        {metrics.box.mr:.4f}')
    print(f"Inference Lat: {metrics.speed['inference']:.2f} ms")
    print('=' * 45)
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    data_root = Path(data_yaml).parent / data_cfg.get('path', '.')
    img_split_dir = data_root / data_cfg.get(split, f'images/{split}')
    lbl_split_dir = data_root / 'labels' / split
    if img_split_dir.exists():
        img_files = sorted(list(img_split_dir.glob('*.jpg')) + list(img_split_dir.glob('*.png')))
        saved_succ, saved_fail = (0, 0)
        print(f'\n[Analyzing {len(img_files)} images from {split} set for failure extraction...]')
        for img_p in img_files:
            lbl_p = lbl_split_dir / img_p.with_suffix('.txt').name
            preds = model.predict(str(img_p), conf=conf_thresh, verbose=False)[0]
            vis_img, tp, fp, fn = analyze_and_draw_cases(img_p, lbl_p, preds)
            if vis_img is None:
                continue
            if (fp > 0 or fn > 0) and saved_fail < 10:
                save_p = failure_dir / f'fail_{img_p.stem}_fp{fp}_fn{fn}.jpg'
                cv2.imwrite(str(save_p), vis_img)
                saved_fail += 1
            elif tp >= 5 and fp == 0 and (fn == 0) and (saved_succ < 10):
                save_p = success_dir / f'success_{img_p.stem}_birds{tp}.jpg'
                cv2.imwrite(str(save_p), vis_img)
                saved_succ += 1
        print(f'[*] Saved {saved_succ} successful case images to: {success_dir}')
        print(f'[*] Saved {saved_fail} failure case images (FP/FN) to: {failure_dir}')
    summary_file = out_dir / f'metrics_{split}_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f'COMP9517 Group Project - Method 2 Evaluation on {split.upper()} Set\n')
        f.write('=' * 65 + '\n')
        f.write(f'mAP@0.5:      {metrics.box.map50:.4f}\n')
        f.write(f'mAP@0.5:0.95: {metrics.box.map:.4f}\n')
        f.write(f'Precision:    {metrics.box.mp:.4f}\n')
        f.write(f'Recall:       {metrics.box.mr:.4f}\n')
        f.write(f"Speed:        {metrics.speed['inference']:.2f} ms/image\n")
    print(f'\n[Completed] Summary metrics saved to: {summary_file}')

def main():
    default_data = str(MODULE_ROOT / 'configs' / 'data.yaml')
    parser = argparse.ArgumentParser(description='Evaluate on Test Set and Extract Failure Cases')
    parser.add_argument('--model', type=str, required=True, help='Path to trained weights (best.pt)')
    parser.add_argument('--data', type=str, default=default_data, help='Path to data.yaml')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'], help="Evaluation split (default: 'test' for independent final evaluation)")
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    evaluate_and_extract_cases(args.model, args.data, args.split, args.conf, args.output_dir)


if __name__ == '__main__':
    main()
