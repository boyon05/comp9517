"""Method 1"""
import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
from common.data_loader import DEFAULT_DATASET, load_dataset


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1)*max(0, y2-y1)
    aa = max(0, a[2]-a[0])*max(0, a[3]-a[1])
    bb = max(0, b[2]-b[0])*max(0, b[3]-b[1])
    return inter/(aa+bb-inter+1e-9)


def nms(items, cutoff=0.4):
    items = sorted(items, key=lambda x: x[1], reverse=True)
    out = []
    for candidate in items:
        if all(iou(candidate[0], b[0]) < cutoff for b in out):
            out.append(candidate)
    return out


def detect(img, area_min=6, area_max_fraction=0.12, threshold=21):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.GaussianBlur(enhanced, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    responses = [cv2.morphologyEx(smooth, op, kernel)
                 for op in (cv2.MORPH_BLACKHAT, cv2.MORPH_TOPHAT)]
    h, w = gray.shape
    detections = []
    for response in responses:
        _, mask = cv2.threshold(response, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            if not (area_min <= area <= area_max_fraction*h*w):
                continue
            if min(bw, bh) < 2 or max(bw, bh)/max(1, min(bw, bh)) > 7:
                continue
            patch = response[y:y+bh, x:x+bw]
            score = float(np.clip(patch.mean()/90, 0.01, 0.99))
            detections.append(([x, y, x+bw, y+bh], score))
    return nms(detections)


def ap50(records, total_gt):
    """Single-class AP at IoU 0.5"""
    if total_gt == 0:
        return None
    det = sorted(((score, img, box) for img, gt, pred in records
                  for box, score in pred), reverse=True)
    if not det:
        return 0.0
    gtmap = {img: gt for img, gt, pred in records}
    used = {img: set() for img, _, _ in records}
    tp, fp = [], []
    for score, img, box in det:
        matches = [iou(box, g) for g in gtmap[img]]
        idx = int(np.argmax(matches)) if matches else -1
        hit = idx >= 0 and matches[idx] >= 0.5 and idx not in used[img]
        if hit:
            used[img].add(idx)
        tp.append(int(hit))
        fp.append(int(not hit))
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = tp / total_gt
    precision = tp / np.maximum(tp+fp, 1)
    r, p = np.r_[0, recall, 1], np.r_[0, precision, 0]
    for j in range(len(p)-2, -1, -1):
        p[j] = max(p[j], p[j+1])
    indices = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[indices+1]-r[indices])*p[indices+1]))


def evaluate(dataset, split, output, threshold, min_area, visualize):
    out = output / split
    out.mkdir(parents=True, exist_ok=True)
    records, rows, times = [], [], []
    gt_count = 0
    for index, (path, img, gt) in enumerate(load_dataset(dataset, split)):
        start = time.perf_counter()
        pred = detect(img, min_area, threshold=threshold)
        elapsed = time.perf_counter()-start
        records.append((str(path), gt, pred))
        gt_count += len(gt)
        times.append(elapsed)
        rows.append({'image': path.name, 'ground_truth': len(gt),
                     'predicted': len(pred), 'inference_ms': round(elapsed*1000, 3)})
        if index < visualize:
            vis = img.copy()
            for box in gt:
                cv2.rectangle(vis, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 255, 0), 1)
            for box, score in pred:
                cv2.rectangle(vis, (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])), (0, 0, 255), 1)
            cv2.imwrite(str(out / f'example_{index:03d}.jpg'), vis)
    metrics = {'split': split, 'images': len(records), 'gt_boxes': gt_count,
               'mAP50': ap50(records, gt_count),
               'total_inference_seconds': sum(times),
               'mean_inference_ms': float(np.mean(times)*1000),
               'method': 'classical_no_training', 'training_seconds': 0,
               'threshold': threshold, 'min_area': min_area}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with (out / 'per_image.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Method 1 bird detection')
    parser.add_argument('--dataset', type=Path, default=DEFAULT_DATASET)
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='all')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent.parent / 'results')
    parser.add_argument('--threshold', type=int, default=21)
    parser.add_argument('--min-area', type=int, default=6)
    parser.add_argument('--visualize', type=int, default=12)
    args = parser.parse_args()
    splits = ('val', 'test') if args.split == 'all' else (args.split,)
    for split in splits:
        print(f'\n========== Running {split} ==========')
        evaluate(args.dataset, split, args.output, args.threshold, args.min_area, args.visualize)


if __name__ == '__main__':
    main()
