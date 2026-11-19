#!/usr/bin/env python3
"""Unified evaluation across all methods.

Reads every method's test-set predictions, computes mAP@0.5 with COCO-style
matching (pooled over the whole split), merges timing numbers, and writes:

    <out>/summary.md               markdown table -> paste straight into the report
    <out>/summary.csv
    <out>/per_image_<method>.csv   per-image n_gt / n_pred / n_tp / n_fn
                                   (this is the input for the failure analysis)

Usage (run from the repository root):

    python exp_analysis_demo/eval_all.py \
        --gt common/dataset/annotations/test.json \
        --preds-dir results/preds \
        --out results/eval

Expected layout (see the prediction-format section in
`exp_analysis_demo/README.md`):

    results/preds/method1_ml/test.json
    results/preds/method2_yolo/test.json
    results/preds/method3_rtdetr/test.json
    results/preds/<method>/timing.json      (optional:
                                             {"train_time_s": 1234.5,
                                              "inference_ms_per_image": 41.2,
                                              "fps": 24.3})

Accepted input formats
----------------------
ground truth: COCO instance JSON (``images`` + ``annotations`` with ``bbox`` in
    xywh) or a flat ``{"img.jpg": [[x1, y1, x2, y2], ...]}`` file.
predictions: per-image dict ``{"img.jpg": {"boxes": [[x1,y1,x2,y2]], "scores": [s]}}``,
    a list ``[{"image_id": ..., "boxes": [...], "scores": [...]}, ...]``,
    or a COCO-results list ``[{"image_id": 1, "bbox": [x,y,w,h], "score": s}]``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.metrics import evaluate, summary_table  # noqa: E402


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _xywh_to_xyxy(box) -> List[float]:
    x, y, w, h = [float(v) for v in box[:4]]
    return [x, y, x + w, y + h]


def load_gt(path: str) -> Dict[str, List[List[float]]]:
    """Load ground truth as {image_key: [[x1, y1, x2, y2], ...]}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotations" in data and "images" in data:
        id2name = {}
        for img in data["images"]:
            key = img.get("file_name", str(img["id"]))
            id2name[img["id"]] = key
            id2name[str(img["id"])] = key
        gts: Dict[str, List[List[float]]] = {name: [] for name in id2name.values()}
        for ann in data["annotations"]:
            key = id2name.get(ann["image_id"])
            if key is None:
                continue
            gts.setdefault(key, []).append(_xywh_to_xyxy(ann["bbox"]))
        return gts

    if isinstance(data, dict):
        # flat {image: [boxes]} or {"images": {image: {"boxes": ...}}}
        if set(data.keys()) <= {"images", "splits"} and isinstance(data.get("images"), dict):
            data = data["images"]
        out: Dict[str, List[List[float]]] = {}
        for key, value in data.items():
            if isinstance(value, dict):
                out[key] = [list(map(float, b)) for b in value.get("boxes", [])]
            else:
                out[key] = [list(map(float, b)) for b in value]
        return out

    raise ValueError(f"unrecognised ground-truth format in {path}")


def load_manifest(path: str) -> Dict[str, dict]:
    """Read dataset_manifest.csv -> {image_file_name: {w, h, count, group}}.

    The Beach Bird dataset ships a manifest with one row per image
    (image_path, image_width_px, image_height_px, object_count,
    source_group_id), which is all we need to turn the YOLO-normalised
    labels back into pixel boxes without touching the images.
    """
    rows: Dict[str, dict] = {}
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            name = os.path.basename(row.get("image_path", "").replace("\\", "/"))
            if not name:
                continue
            try:
                w = int(row["image_width_px"])
                h = int(row["image_height_px"])
            except (KeyError, TypeError, ValueError):
                continue
            rows[name] = {
                "w": w,
                "h": h,
                "count": int(row.get("object_count") or 0),
                "group": row.get("source_group_id") or "",
                "split": row.get("split") or "",
            }
    return rows


def load_gt_yolo(labels_dir: str, manifest: str = None, images_dir: str = None,
                 default_size=(512, 512)) -> Dict[str, List[List[float]]]:
    """Load YOLO-normalised ``labels/<split>/*.txt`` as pixel xyxy.

    Frame size comes from the manifest (preferred), then from the images on
    disk (needs Pillow), then from ``default_size``.
    """
    names = load_manifest(manifest) if manifest else {}
    stem2name = {os.path.splitext(k)[0]: k for k in names}

    sizes = {k: (v["w"], v["h"]) for k, v in names.items()}
    if images_dir:
        try:
            from PIL import Image  # noqa: F401  (only needed for exact sizes)
            for fn in sorted(os.listdir(images_dir)):
                if fn in sizes:
                    continue
                try:
                    with Image.open(os.path.join(images_dir, fn)) as im:
                        sizes[fn] = im.size
                except Exception:
                    continue
        except ImportError:
            pass

    gts: Dict[str, List[List[float]]] = {}
    missing_size = 0
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.lower().endswith(".txt"):
            continue
        stem = os.path.splitext(fn)[0]
        image_name = stem2name.get(stem, stem + ".jpg")
        w, h = sizes.get(image_name, default_size)
        if image_name not in sizes:
            missing_size += 1
        boxes: List[List[float]] = []
        with open(os.path.join(labels_dir, fn), "r", encoding="utf-8-sig") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                              (cx + bw / 2) * w, (cy + bh / 2) * h])
        gts[image_name] = boxes

    if missing_size:
        print("[eval_all] WARNING: %d label file(s) had no image size (used %sx%s); "
              "pass --manifest or --images-dir for exact boxes"
              % (missing_size, default_size[0], default_size[1]), file=sys.stderr)
    return gts


def load_preds(path: str) -> Dict[str, Dict[str, list]]:
    """Load one method's predictions as {image_key: {"boxes": ..., "scores": ...}}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # unwrap {"predictions": ...} / {"detections": ...}
    if isinstance(data, dict) and len(data) == 1:
        only_key = next(iter(data))
        if only_key in ("predictions", "detections") and isinstance(data[only_key], (list, dict)):
            data = data[only_key]

    out: Dict[str, Dict[str, list]] = {}

    if isinstance(data, list):
        # list of per-image records, or a COCO results list (one dict per box)
        for item in data:
            if "boxes" in item:
                out[str(item["image_id"])] = {
                    "boxes": [list(map(float, b)) for b in item.get("boxes", [])],
                    "scores": [float(s) for s in item.get("scores", [])],
                }
            elif "bbox" in item:  # COCO results: one row per detection
                key = str(item["image_id"])
                entry = out.setdefault(key, {"boxes": [], "scores": []})
                entry["boxes"].append(_xywh_to_xyxy(item["bbox"]))
                entry["scores"].append(float(item.get("score", 1.0)))
        return out

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                boxes = value.get("boxes", value.get("bbox", []))
                scores = value.get("scores", value.get("score", []))
            else:  # {image: [[x1,y1,x2,y2], ...]} without scores -> assume 1.0
                boxes, scores = value, [1.0] * len(value)
            out[str(key)] = {
                "boxes": [list(map(float, b)) for b in boxes],
                "scores": [float(s) for s in np.reshape(scores, -1)],
            }
        return out

    raise ValueError(f"unrecognised prediction format in {path}")


def load_timing(path: str) -> Dict[str, float]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Unified mAP@0.5 evaluation over all methods")
    ap.add_argument("--gt", default=None, help="ground-truth json (COCO or flat)")
    ap.add_argument("--gt-yolo", default=None,
                    help="YOLO label dir, e.g. 'dataset/labels/test' (labels/*.txt)")
    ap.add_argument("--manifest", default=None,
                    help="dataset_manifest.csv, gives exact image sizes + density")
    ap.add_argument("--images-dir", default=None,
                    help="optional image dir; only used to read sizes when no manifest")
    ap.add_argument("--preds-dir", default="results/preds", help="results/preds/<method>/test.json")
    ap.add_argument("--out", default="results/eval", help="output directory")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold (default 0.5)")
    ap.add_argument("--conf", type=float, default=None, help="optional confidence cut-off")
    ap.add_argument("--methods", default=None, help="comma separated subset, default: all")
    args = ap.parse_args(argv)

    if not args.gt and not args.gt_yolo:
        ap.error("give either --gt (json) or --gt-yolo (label directory)")

    if args.gt_yolo:
        gts = load_gt_yolo(args.gt_yolo, manifest=args.manifest, images_dir=args.images_dir)
        gt_desc = f"{os.path.basename(os.path.normpath(args.gt_yolo))}/*.txt"
    else:
        gts = load_gt(args.gt)
        gt_desc = os.path.basename(args.gt)
    n_pos = sum(1 for v in gts.values() if len(v) > 0)
    print(f"[eval_all] GT: {len(gts)} images ({n_pos} positive, {len(gts) - n_pos} negative), "
          f"{sum(len(v) for v in gts.values())} boxes  <- {gt_desc}")
    wanted = set(m.strip() for m in args.methods.split(",")) if args.methods else None

    pred_files = []
    for entry in sorted(os.listdir(args.preds_dir)):
        mdir = os.path.join(args.preds_dir, entry)
        if not os.path.isdir(mdir):
            continue
        if wanted and entry not in wanted:
            continue
        for cand in ("test.json", "preds.json", f"{entry}.json"):
            p = os.path.join(mdir, cand)
            if os.path.isfile(p):
                pred_files.append((entry, p))
                break

    if not pred_files:
        print(f"[eval_all] no predictions found under {args.preds_dir}/<method>/test.json")
        return 1

    results, timing, per_image_all = [], {}, {}
    for name, path in pred_files:
        preds = load_preds(path)
        unknown = sorted(set(preds) - set(gts))
        missing = sorted(set(gts) - set(preds))
        if unknown:
            print(f"[eval_all] {name}: {len(unknown)} predicted image keys not in GT, ignored "
                  f"(e.g. {unknown[:3]})")
        if missing:
            print(f"[eval_all] {name}: {len(missing)} GT images without predictions "
                  f"(counted as all-missed, e.g. {missing[:3]})")

        res = evaluate(gts, preds, iou_thr=args.iou, name=name, score_thr=args.conf)
        per_image_all[name] = res.pop("_per_image")
        results.append(res)
        timing[name] = load_timing(os.path.join(os.path.dirname(path), "timing.json"))
        print(f"[eval_all] {name}: mAP@{args.iou} = {res['ap50']:.4f}  "
              f"(TP {res['tp']}, FP {res['fp']}, FN {res['fn']})")

    os.makedirs(args.out, exist_ok=True)

    # ---- summary.md / summary.csv ---------------------------------------- #
    table = summary_table(results, timing)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(f"# Unified evaluation (test split, IoU = {args.iou})\n\n")
        f.write(f"Ground truth: `{gt_desc}`, "
                f"{len(gts)} images, {sum(len(v) for v in gts.values())} boxes.\n\n")
        f.write(table + "\n")

    fields = ["method", "ap50", "n_gt", "n_pred", "tp", "fp", "fn",
              "best_precision", "best_recall", "best_f1", "best_threshold",
              "train_time_s", "inference_ms_per_image", "fps"]
    with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            row = dict(r)
            row.update(timing.get(r["method"], {}))
            w.writerow(row)

    # ---- per-image dump for the failure analysis -------------------------- #
    for name, stats in per_image_all.items():
        path = os.path.join(args.out, f"per_image_{name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "n_gt", "n_pred", "n_tp", "n_fn", "miss_rate", "fp_rate"])
            for img, s in sorted(stats.items()):
                miss = s["n_fn"] / s["n_gt"] if s["n_gt"] else 0.0
                fp_rate = (s["n_pred"] - s["n_tp"]) / s["n_pred"] if s["n_pred"] else 0.0
                w.writerow([img, s["n_gt"], s["n_pred"], s["n_tp"], s["n_fn"],
                            f"{miss:.4f}", f"{fp_rate:.4f}"])

    print("\n" + table)
    print(f"\n[eval_all] wrote {args.out}/summary.md, summary.csv, "
          f"per_image_<method>.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
