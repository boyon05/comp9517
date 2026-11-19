#!/usr/bin/env python3
"""Post-processing sweep: does NMS IoU / confidence explain the dense-scene
precision collapse, and can a cheap, retrain-free setting fix it?

Motivation: the per-density breakdown of Method 2 showed
recall is flat across density bins (0.89-0.96) while false positives per image
grow 5.3 -> 25.5 -> 76 -> 160 from 1-5 to 51+ birds.  So the mAP loss in dense
scenes is a *precision* loss: many boxes per bird, not missing birds.  Two
inference-time knobs control exactly that, and neither needs retraining:

    conf   confidence kept after NMS  (cuts the low-score tail that floods the
           PR curve in crowded images)
    iou    NMS overlap threshold     (decides whether two overlapping boxes on
           neighbouring birds are duplicates or two detections)

Protocol: sweep both on the **validation** split, pick the setting with the best
val mAP@0.5, then evaluate that single setting on **test** once.  Nothing is
tuned on test.

    python exp_analysis_demo/sweep_postprocess.py                 # val sweep + test confirm
    python exp_analysis_demo/sweep_postprocess.py --no-test       # val only

Outputs: results/sweep/sweep.csv, results/sweep/summary.md,
results/preds/<method>_tuned/{test,timing}.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "common", "dataset", "dataset_manifest.csv")
BIN_ORDER = ["0", "1-5", "6-20", "21-50", "51+"]


# --------------------------------------------------------------------------- nms
def nms(boxes, scores, iou_thr):
    """Plain greedy NMS for a single class; boxes are xyxy."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    b = np.asarray(boxes, dtype=float)
    s = np.asarray(scores, dtype=float)
    order = s.argsort()[::-1]
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=int)


def apply_settings(raw, iou_thr, conf_thr):
    out = {}
    for name, rec in raw.items():
        boxes = rec["boxes"]; scores = rec["scores"]
        if not boxes:
            continue
        keep = nms(boxes, scores, iou_thr)
        boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]
        pairs = [(b, s) for b, s in zip(boxes, scores) if s >= conf_thr]
        if pairs:
            out[name] = {"boxes": [p[0] for p in pairs], "scores": [p[1] for p in pairs]}
    return out


# ----------------------------------------------------------------- raw export
def export_raw(weights, images_dir, cache, conf=0.01, iou=0.85, imgsz=640, batch=16):
    """Run the model once, keeping near-duplicates (permissive NMS) so the sweep
    can re-apply NMS itself without another GPU pass."""
    if os.path.isfile(cache):
        with open(cache, "r", encoding="utf-8") as f:
            return json.load(f)

    import torch
    from ultralytics import YOLO

    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    model = YOLO(weights)
    device = 0 if torch.cuda.is_available() else "cpu"
    raw = {}
    t0 = time.perf_counter()
    for start in range(0, len(files), batch):
        chunk = [os.path.join(images_dir, f) for f in files[start:start + batch]]
        results = model.predict(chunk, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
        for f, r in zip(files[start:start + batch], results):
            raw[f] = {"boxes": [[round(float(v), 2) for v in b] for b in r.boxes.xyxy.cpu().numpy()],
                      "scores": [round(float(s), 4) for s in r.boxes.conf.cpu().numpy()]}
    dt = time.perf_counter() - t0
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    print("[sweep] raw export %d images in %.1fs (iou=%.2f, conf=%.2f) -> %s"
          % (len(files), dt, iou, conf, cache))
    return raw


# ------------------------------------------------------------------ reporting
def bin_table(gts, preds):
    from common.metrics import evaluate

    if not os.path.isfile(MANIFEST):
        return []
    manifest = {}
    with open(MANIFEST, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest[row["annotation_preview_file"]] = row
    rows = {}
    for name in gts:
        b = manifest.get(name, {}).get("density_bin", "?")
        agg = rows.setdefault(b, {"n": 0, "gt": 0, "pred": 0, "tp": 0})
        res = evaluate({name: gts[name]}, {name: preds.get(name, {"boxes": [], "scores": []})})
        agg["n"] += 1; agg["gt"] += res["n_gt"]; agg["pred"] += res["n_pred"]; agg["tp"] += res["tp"]
    out = []
    for b in BIN_ORDER + sorted(k for k in rows if k not in BIN_ORDER):
        a = rows.get(b)
        if not a:
            continue
        out.append((b, a["n"], a["gt"], a["tp"],
                    a["tp"] / a["gt"] if a["gt"] else float("nan"),
                    (a["pred"] - a["tp"]) / a["n"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.path.join(
        REPO_ROOT, "method2", "results", "train_runs", "single_stage_baseline",
        "weights", "best.pt"))
    ap.add_argument("--dataset", default=os.path.join(REPO_ROOT, "common", "dataset"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "sweep"))
    ap.add_argument("--ious", default="0.5,0.6,0.7,0.8,0.85")
    ap.add_argument("--confs", default="0.01,0.15,0.25,0.35,0.5")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--baseline-iou", type=float, default=0.7)
    ap.add_argument("--baseline-conf", type=float, default=0.01)
    ap.add_argument("--no-test", action="store_true")
    args = ap.parse_args(argv)

    sys.path.insert(0, REPO_ROOT)
    from common.metrics import evaluate
    from exp_analysis_demo.eval_all import load_gt_yolo

    os.makedirs(args.out, exist_ok=True)
    labels_val = os.path.join(args.dataset, "labels", "val")
    labels_test = os.path.join(args.dataset, "labels", "test")
    manifest = MANIFEST if os.path.isfile(MANIFEST) else None
    gts_val = load_gt_yolo(labels_val, manifest,
                           os.path.join(args.dataset, "images", "val"))
    gts_test = load_gt_yolo(labels_test, manifest,
                            os.path.join(args.dataset, "images", "test"))

    raw_val = export_raw(args.weights, os.path.join(args.dataset, "images", "val"),
                         os.path.join(args.out, "raw_val.json"), imgsz=args.imgsz)
    ious = [float(v) for v in args.ious.split(",")]
    confs = [float(v) for v in args.confs.split(",")]

    rows = []
    for iou_thr in ious:
        for conf_thr in confs:
            preds = apply_settings(raw_val, iou_thr, conf_thr)
            res = evaluate(gts_val, preds, 0.5)
            rows.append((iou_thr, conf_thr, res["ap50"], res["best_f1"],
                         res["best_precision"], res["best_recall"], res["n_pred"] / len(gts_val)))
            print("[val] iou %.2f conf %.2f -> mAP@0.5 %.4f  F1 %.3f  boxes/img %.1f"
                  % (iou_thr, conf_thr, res["ap50"], res["best_f1"], res["n_pred"] / len(gts_val)))

    base = next(r for r in rows if abs(r[0] - args.baseline_iou) < 1e-9
                and abs(r[1] - args.baseline_conf) < 1e-9) if any(
        abs(r[0] - args.baseline_iou) < 1e-9 and abs(r[1] - args.baseline_conf) < 1e-9 for r in rows) else None
    best = max(rows, key=lambda r: r[2])
    print("[val] best setting: iou %.2f conf %.2f -> mAP@0.5 %.4f" % (best[0], best[1], best[2]))

    with open(os.path.join(args.out, "sweep.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["iou", "conf", "val_map50", "val_best_f1", "val_best_precision",
                    "val_best_recall", "val_boxes_per_image"])
        w.writerows(rows)

    lines = ["# Post-processing sweep (method 2, validation split)", "",
             "Same weights, no retraining. Rows are the grid; the selected setting is",
             "chosen by val mAP@0.5 and applied to test exactly once.", "",
             "| NMS IoU | conf | val mAP@0.5 | val F1 | boxes/img |",
             "|---|---|---|---|---|"]
    for iou_thr, conf_thr, ap50, f1, p, r, bpi in rows:
        mark = " **<- selected**" if (iou_thr, conf_thr) == (best[0], best[1]) else ""
        lines.append("| %.2f | %.2f | %.4f | %.3f | %.1f |%s" % (iou_thr, conf_thr, ap50, f1, bpi, mark))
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    if args.no_test:
        return 0

    raw_test = export_raw(args.weights, os.path.join(args.dataset, "images", "test"),
                          os.path.join(args.out, "raw_test.json"), imgsz=args.imgsz)
    before = evaluate(gts_test, apply_settings(raw_test, args.baseline_iou, args.baseline_conf), 0.5)
    tuned_preds = apply_settings(raw_test, best[0], best[1])
    after = evaluate(gts_test, tuned_preds, 0.5)
    print("[test] baseline iou %.2f conf %.2f -> mAP@0.5 %.4f"
          % (args.baseline_iou, args.baseline_conf, before["ap50"]))
    print("[test] tuned    iou %.2f conf %.2f -> mAP@0.5 %.4f"
          % (best[0], best[1], after["ap50"]))

    out_dir = os.path.join(REPO_ROOT, "results", "preds", "method2_yolo_tuned")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump(tuned_preds, f)
    with open(os.path.join(out_dir, "timing.json"), "w") as f:
        json.dump({"train_time_s": 472, "train_images": 1369, "inference_ms_per_image": 19.98,
                   "device": "NVIDIA GeForce RTX 3060 Laptop GPU",
                   "note": "same weights as method2_yolo, NMS iou %.2f conf %.2f "
                           "(selected on val)" % (best[0], best[1])}, f, indent=1)

    lines += ["", "## Test (touch once)", "",
              "| Setting | mAP@0.5 | TP | FP | FN |", "|---|---|---|---|---|",
              "| baseline (iou %.2f / conf %.2f) | %.4f | %d | %d | %d |"
              % (args.baseline_iou, args.baseline_conf, before["ap50"], before["tp"],
                 before["fp"], before["fn"]),
              "| tuned (iou %.2f / conf %.2f) | %.4f | %d | %d | %d |"
              % (best[0], best[1], after["ap50"], after["tp"], after["fp"], after["fn"]),
              "", "### Per density bin", "",
              "| bin | imgs | GT | TP | recall | FP/img |", "|---|---|---|---|---|---|"]
    for label, preds in (("baseline", apply_settings(raw_test, args.baseline_iou, args.baseline_conf)),
                         ("tuned", tuned_preds)):
        lines.append("**%s**" % label)
        for b, n, gt, tp, rec, fpi in bin_table(gts_test, preds):
            lines.append("| %s | %d | %d | %d | %.3f | %.1f |" % (b, n, gt, tp, rec, fpi))
        lines.append("")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[sweep] wrote %s" % os.path.join(args.out, "summary.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
