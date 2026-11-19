#!/usr/bin/env python3
"""Post-processing variants tried against the dense-scene precision collapse.

This experiment compares lightweight post-processing options and records their
effects. Everything here uses the cached candidate dumps from
``sweep_postprocess.py`` (raw boxes at
NMS IoU 0.85, conf 0.01), so it runs in seconds and needs no GPU except for the
optional input-resolution pass.

Variants
--------
nms      greedy NMS with a range of IoU thresholds and confidence cut-offs
wbf      weighted box fusion (merge, don't delete) with and without a
         centre-distance gate that refuses to fuse two spatially distinct birds
imgsz    re-running the network at 800/1024/1280 instead of the trained 640

Verdict (see results/sweep/summary.md): none of them recovers mAP@0.5.  Dense
scenes lose *precision at a fixed operating point*, not mAP: each bird attracts
2.5-3.5 boxes that overlap at IoU<0.7, so NMS cannot remove them without also
merging adjacent birds, and mAP integrates over the confidence ranking, which
keeps the duplicate mass nearly free.

    python exp_analysis_demo/sweep_postprocess.py --no-test    # builds raw_val.json
    python exp_analysis_demo/postprocess_variants.py           # nms + wbf table
    python exp_analysis_demo/postprocess_variants.py --imgsz   # + resolution sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "common", "dataset", "dataset_manifest.csv")


def iou_pair(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def greedy_nms(boxes, scores, iou_thr):
    order = np.argsort(-np.asarray(scores)) if len(boxes) else []
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = rest[[iou_pair(boxes[i], boxes[j]) <= iou_thr for j in rest]]
    return keep


def weighted_fusion(boxes, scores, thr, conf=0.01, center_gate=None):
    """Merge overlapping boxes instead of deleting them.  ``center_gate`` scales
    the required centre distance by the box size, so two adjacent birds are not
    fused just because they overlap."""
    order = np.argsort(-np.asarray(scores)) if len(boxes) else []
    clusters = []
    for i in order:
        b = [float(v) for v in boxes[i]]
        s = float(scores[i])
        if s < conf:
            continue
        placed = False
        for c in clusters:
            fb = c[0]
            if iou_pair(b, fb) < thr:
                continue
            if center_gate is not None:
                d = (((b[0] + b[2]) / 2 - (fb[0] + fb[2]) / 2) ** 2 +
                     ((b[1] + b[3]) / 2 - (fb[1] + fb[3]) / 2) ** 2) ** 0.5
                if d > center_gate * ((fb[2] - fb[0] + b[2] - b[0]) / 2 +
                                      (fb[3] - fb[1] + b[3] - b[1]) / 2) / 2:
                    continue
            w1, w2 = sum(c[1]), s
            c[0] = [(fb[k] * w1 + b[k] * w2) / (w1 + w2) for k in range(4)]
            c[1].append(s)
            placed = True
            break
        if not placed:
            clusters.append([b, [s]])
    out = {"boxes": [], "scores": []}
    for c in clusters:
        out["boxes"].append([round(v, 2) for v in c[0]])
        out["scores"].append(round(float(np.mean(c[1])), 4))
    return out


def rebuild(raw, mode, **kw):
    preds = {}
    for name, rec in raw.items():
        if not rec.get("boxes"):
            continue
        if mode == "nms":
            keep = greedy_nms(rec["boxes"], rec["scores"], kw["iou"])
            boxes = [rec["boxes"][i] for i in keep]
            scores = [rec["scores"][i] for i in keep]
            pairs = [(b, s) for b, s in zip(boxes, scores) if s >= kw["conf"]]
            if pairs:
                preds[name] = {"boxes": [p[0] for p in pairs], "scores": [p[1] for p in pairs]}
        else:
            fused = weighted_fusion(rec["boxes"], rec["scores"], kw["thr"],
                                    center_gate=kw.get("gate"))
            if fused["boxes"]:
                preds[name] = fused
    return preds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", default=os.path.join(REPO_ROOT, "results", "sweep"))
    ap.add_argument("--weights", default=os.path.join(
        REPO_ROOT, "method2", "results", "train_runs", "single_stage_baseline",
        "weights", "best.pt"))
    ap.add_argument("--dataset", default=os.path.join(REPO_ROOT, "common", "dataset"))
    ap.add_argument("--imgsz", action="store_true", help="also re-run at 800/1024/1280")
    args = ap.parse_args(argv)

    sys.path.insert(0, REPO_ROOT)
    from common.metrics import evaluate
    from exp_analysis_demo.eval_all import load_gt_yolo

    raw_path = os.path.join(args.sweep_dir, "raw_val.json")
    if not os.path.isfile(raw_path):
        print("missing %s — run sweep_postprocess.py --no-test first" % raw_path)
        return 2
    raw = json.load(open(raw_path, encoding="utf-8"))
    manifest = MANIFEST if os.path.isfile(MANIFEST) else None
    gts = load_gt_yolo(
        os.path.join(args.dataset, "labels", "val"),
        manifest,
        os.path.join(args.dataset, "images", "val"),
    )
    n_img = len(gts)

    rows = []

    def record(tag, preds):
        res = evaluate(gts, preds, 0.5)
        rows.append((tag, res["ap50"], res["tp"], res["fp"], res["fn"],
                     round(res["n_pred"] / n_img, 1)))
        print("%-26s mAP@0.5 %.4f  TP %4d FP %5d FN %4d  boxes/img %.1f"
              % (tag, res["ap50"], res["tp"], res["fp"], res["fn"], res["n_pred"] / n_img))

    record("NMS 0.70 / conf 0.01 (baseline)", rebuild(raw, "nms", iou=0.7, conf=0.01))
    for iou in (0.5, 0.6, 0.8):
        record("NMS %.2f / conf 0.01" % iou, rebuild(raw, "nms", iou=iou, conf=0.01))
    for conf in (0.15, 0.25, 0.35, 0.5):
        record("NMS 0.70 / conf %.2f" % conf, rebuild(raw, "nms", iou=0.7, conf=conf))
    for thr in (0.4, 0.5, 0.55, 0.6, 0.7):
        record("WBF iou>=%.2f" % thr, rebuild(raw, "wbf", thr=thr))
    for gate in (0.3, 0.5):
        record("WBF iou>=0.60 gate %.1f" % gate, rebuild(raw, "wbf", thr=0.6, gate=gate))

    csv_path = os.path.join(args.sweep_dir, "variants.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "val_map50", "tp", "fp", "fn", "boxes_per_image"])
        w.writerows(rows)

    if args.imgsz:
        import torch
        from ultralytics import YOLO

        d = os.path.join(args.dataset, "images", "val")
        files = sorted(f for f in os.listdir(d) if f.lower().endswith(".jpg"))
        model = YOLO(args.weights)
        for size in (800, 1024, 1280):
            preds = {}
            for s in range(0, len(files), 16):
                chunk = [os.path.join(d, f) for f in files[s:s + 16]]
                for f, r in zip(files[s:s + 16],
                                model.predict(chunk, conf=0.01, iou=0.7, imgsz=size,
                                              device=0, verbose=False)):
                    preds[f] = {"boxes": [[float(v) for v in b] for b in r.boxes.xyxy.cpu().numpy()],
                                "scores": [float(x) for x in r.boxes.conf.cpu().numpy()]}
            record("network at imgsz %d" % size, preds)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerows(rows[-3:])

    best = max(rows, key=lambda r: r[1])
    with open(os.path.join(args.sweep_dir, "variants.md"), "w", encoding="utf-8") as f:
        f.write("# Post-processing variants on the validation split\n\n")
        f.write("Candidate pool: raw boxes from `sweep_postprocess.py` (NMS IoU 0.85,\n"
                "conf 0.01). Best row by val mAP@0.5: **%s**.\n\n" % best[0])
        f.write("| variant | val mAP@0.5 | TP | FP | FN | boxes/img |\n|---|---|---|---|---|---|\n")
        for tag, ap50, tp, fp, fn, bpi in rows:
            f.write("| %s | %.4f | %d | %d | %d | %.1f |\n" % (tag, ap50, tp, fp, fn, bpi))
        f.write("\nNo variant beats plain NMS at IoU 0.70 with conf 0.01. Raising conf\n"
                "cuts boxes/img by 4x but costs 1.5-15 mAP points, because mAP integrates\n"
                "over the confidence ranking: the duplicate mass is already ranked below\n"
                "the true detections, so deleting it removes true positives too.\n")
    print("\nwrote %s and %.0f rows to %s" % (os.path.join(args.sweep_dir, "variants.md"),
                                             len(rows), csv_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
