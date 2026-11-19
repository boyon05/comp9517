#!/usr/bin/env python3
"""Bootstrap confidence intervals for the test-split mAP@0.5 numbers.

    python exp_analysis_demo/bootstrap_ci.py --reps 1000

The test split has 240 images, so a difference of one point between methods is
only meaningful if it is larger than the sampling noise.  This script resamples
whole images with replacement (the same resample for every method, so pairwise
gaps are paired) and reports the 2.5/97.5 percentiles.

Outputs results/eval/bootstrap.{md,json}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

from common.metrics import average_precision, match_image  # noqa: E402
from exp_analysis_demo.eval_all import load_gt_yolo, load_manifest, load_preds  # noqa: E402

IMG_DIR = os.path.join(REPO, "common", "dataset", "images", "test")
LBL_DIR = os.path.join(REPO, "common", "dataset", "labels", "test")
MANIFEST = os.path.join(REPO, "common", "dataset", "dataset_manifest.csv")
PREDS = os.path.join(REPO, "results", "preds")


def per_image(gts: dict, preds: dict) -> dict:
    out = {}
    for name, gt_boxes in gts.items():
        pred = preds.get(name) or {}
        boxes = np.asarray(pred.get("boxes") or [], dtype=float).reshape(-1, 4)
        scores = np.asarray(pred.get("scores") or [], dtype=float).reshape(-1)
        gt = np.asarray(gt_boxes, dtype=float).reshape(-1, 4)
        s, lab, n_gt, n_fn = match_image(boxes, scores, gt, iou_thr=0.5)
        out[name] = (s, lab, n_gt)
    return out


def pooled_ap(images: list, idx: np.ndarray) -> float:
    s = np.concatenate([images[i][0] for i in idx]) if len(idx) else np.zeros(0)
    lab = np.concatenate([images[i][1] for i in idx]) if len(idx) else np.zeros(0)
    n_gt = int(sum(images[i][2] for i in idx))
    if n_gt == 0:
        return 0.0
    return average_precision(s, lab, n_gt)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="method1_ml,method2_yolo,method3_rtdetr")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "eval"))
    args = ap.parse_args(argv)

    gts = load_gt_yolo(LBL_DIR, MANIFEST if os.path.isfile(MANIFEST) else None, IMG_DIR)
    names = sorted(gts)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    data = {}
    for m in methods:
        path = os.path.join(PREDS, m, "test.json")
        if not os.path.exists(path):
            print("[boot] skip %s (no %s)" % (m, path))
            continue
        data[m] = per_image(gts, load_preds(path))
    if not data:
        raise SystemExit("[boot] nothing to do")

    rng = np.random.default_rng(args.seed)
    point = {m: pooled_ap([data[m][n] for n in names], np.arange(len(names))) for m in data}

    reps = {m: np.empty(args.reps) for m in data}
    pairs = [(a, b) for i, a in enumerate(data) for b in list(data)[i + 1:]]
    pair_vals = {p: np.empty(args.reps) for p in pairs}
    for b in range(args.reps):
        idx = rng.integers(0, len(names), len(names))
        for m in data:
            reps[m][b] = pooled_ap([data[m][n] for n in names], idx)
        for a, c in pairs:
            pair_vals[(a, c)][b] = reps[a][b] - reps[c][b]

    rows = []
    for m in data:
        lo, hi = np.percentile(reps[m], [2.5, 97.5])
        rows.append({"method": m, "ap50": round(point[m], 4),
                     "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                     "sd": round(float(reps[m].std()), 4)})
    gap_rows = []
    for (a, c), vals in pair_vals.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        gap_rows.append({"a": a, "b": c, "delta_pp": round((point[a] - point[c]) * 100, 2),
                         "ci_lo_pp": round(float(lo) * 100, 2),
                         "ci_hi_pp": round(float(hi) * 100, 2),
                         "p_a_worse": round(float((vals < 0).mean()), 3)})

    lines = ["# Bootstrap intervals for test mAP@0.5", "",
             "%d images, %d resamples, whole-image resampling (paired across methods)."
             % (len(names), args.reps), "",
             "| method | mAP@0.5 | 95% CI | bootstrap sd |", "|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %.4f | %.4f - %.4f | %.4f |"
                     % (r["method"], r["ap50"], r["ci_lo"], r["ci_hi"], r["sd"]))
    lines += ["", "Paired differences (positive = first method better):", "",
              "| pair | delta (pp) | 95% CI (pp) | P(first worse) |", "|---|---|---|---|"]
    for r in gap_rows:
        lines.append("| %s - %s | %+.2f | %+.2f ... %+.2f | %.3f |"
                     % (r["a"], r["b"], r["delta_pp"], r["ci_lo_pp"], r["ci_hi_pp"],
                        r["p_a_worse"]))
    lines += ["", "Read: a gap whose CI excludes 0 is larger than the sampling noise of this "
                  "240-image test split; a CI that straddles 0 means the ranking is not "
                  "resolved by this split alone."]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "bootstrap.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "bootstrap.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "gaps": gap_rows, "reps": args.reps, "n_images": len(names)}, f,
                  indent=1)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
