#!/usr/bin/env python3
"""Density-bin comparison of method 2 vs method 3 on the test split.

This analysis compares crowded-scene false positives and recall for Method 2
and RT-DETRv2, a set predictor without NMS. Both methods are compared at
the same visualisation confidence (0.25) and the same greedy matcher (IoU 0.5),
binned by the dataset's own ``density_bin``.

Inputs
  results/preds/method2_yolo/test.json   raw boxes (score >= 0.01) for method 2
  <method3-output>/method3-*/per_image.csv  per-image GT/pred/TP/FP/FN
  common/dataset/dataset_manifest.csv  official density bins

    python exp_analysis_demo/density_compare.py \
        --team-output "results/c_output"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "common", "dataset", "dataset_manifest.csv")
BINS = ["0", "1-5", "6-20", "21-50", "51+"]
CONF = 0.25


def load_bins(manifest_path):
    bins = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row.get("split") or "").strip() != "test":
                continue
            name = os.path.basename(row.get("image_path") or row.get("image")
                                    or row.get("file_name") or "")
            if name:
                bins[name] = (row.get("density_bin") or "").strip()
    return bins


def match_one(gt_boxes, pred, conf):
    sys.path.insert(0, REPO_ROOT)
    from common.metrics import match_image

    boxes = [b for b, s in zip(pred.get("boxes", []), pred.get("scores", [])) if s >= conf]
    scores = [s for s in pred.get("scores", []) if s >= conf]
    if not boxes:
        return 0, 0, len(gt_boxes)
    _s, labels, _n_gt, n_fn = match_image(boxes, scores, gt_boxes, 0.5)
    tp = int(sum(labels))
    return tp, len(boxes) - tp, n_fn


def bins_from_preds(preds_path, gts, bins, conf):
    preds = json.load(open(preds_path, encoding="utf-8"))
    out = {b: {"images": 0, "gt": 0, "tp": 0, "fp": 0, "fn": 0} for b in BINS}
    for name, gt_boxes in gts.items():
        b = bins.get(name)
        if b not in out:
            continue
        pred = preds.get(name) or {}
        tp, fp, fn = match_one(gt_boxes, pred, conf)
        row = out[b]
        row["images"] += 1
        row["gt"] += len(gt_boxes)
        row["tp"] += tp
        row["fp"] += fp
        row["fn"] += fn
    return out


def bins_from_csv(path, bins):
    out = {b: {"images": 0, "gt": 0, "tp": 0, "fp": 0, "fn": 0} for b in BINS}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_tp = "true_positives" in (reader.fieldnames or [])
        for row in reader:
            name = os.path.basename(row["image"])
            b = bins.get(name)
            if b not in out:
                continue
            r = out[b]
            r["images"] += 1
            r["gt"] += int(row.get("ground_truth", row.get("n_gt")) or 0)
            if has_tp:
                r["tp"] += int(row.get("true_positives") or 0)
                r["fp"] += int(row.get("false_positives") or 0)
                r["fn"] += int(row.get("false_negatives") or 0)
            else:
                r["tp"] += int(row.get("n_tp") or 0)
                r["fn"] += int(row.get("n_fn") or 0)
                r["fp"] += int(row.get("n_pred") or 0) - int(row.get("n_tp") or 0)
    return out


def show(label, table):
    print("\n%s" % label)
    print("| density | images | GT | recall | FP/image | FP/total-pred |")
    print("|---|---|---|---|---|---|")
    for b in BINS:
        r = table.get(b)
        if not r or not r["images"]:
            continue
        rec = r["tp"] / r["gt"] if r["gt"] else float("nan")
        pred = r["tp"] + r["fp"]
        share = r["fp"] / pred if pred else float("nan")
        print("| %s | %d | %d | %.3f | %.1f | %.2f |"
              % (b, r["images"], r["gt"], rec, r["fp"] / r["images"], share))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team-output", required=True,
                    help="directory holding method3-default/ and method3-stride4/")
    ap.add_argument("--preds", default=os.path.join(REPO_ROOT, "results", "preds",
                                                    "method2_yolo", "test.json"))
    ap.add_argument("--dataset", default=os.path.join(REPO_ROOT, "common", "dataset"))
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    if not os.path.isfile(args.manifest):
        raise SystemExit("density analysis requires dataset_manifest.csv: %s" % args.manifest)

    sys.path.insert(0, REPO_ROOT)
    from exp_analysis_demo.eval_all import load_gt_yolo
    gts = load_gt_yolo(
        os.path.join(args.dataset, "labels", "test"),
        args.manifest,
        os.path.join(args.dataset, "images", "test"),
    )
    bins = load_bins(args.manifest)
    result = {}

    m2 = bins_from_preds(args.preds, gts, bins, args.conf)
    result["method2_yolo"] = m2
    show("method 2 (YOLOv8s, conf >= %.2f)" % args.conf, m2)

    for variant in ("method3-default", "method3-stride4"):
        path = os.path.join(args.team_output, variant, "per_image.csv")
        if not os.path.exists(path):  # flattened copy committed under results/c_output
            path = os.path.join(args.team_output, "%s__per_image.csv" % variant)
        if not os.path.isfile(path):
            print("\n[skip] %s" % path)
            continue
        t = bins_from_csv(path, bins)
        result[variant] = t
        show("%s (conf >= %.2f)" % (variant, args.conf), t)

    if args.json_out:
        json.dump(result, open(args.json_out, "w", encoding="utf-8"), indent=2)
        print("\nwrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
