#!/usr/bin/env python3
"""Export method 1's predictions in the shared format

Method 1 (classical morphology, method1/method1.py) has no weights: it is a
function of the image, so it can always be re-run locally.  This wrapper only
*observes* it -- it imports the same ``detect`` function, never edits it -- and
writes the two files every other method also writes:

    results/preds/method1_ml/test.json     boxes (pixel xyxy) + scores
    results/preds/method1_ml/timing.json   per-image latency, no training time

    python exp_analysis_demo/export_method1.py --dataset common/dataset --split test
    python exp_analysis_demo/export_method1.py --dataset common/dataset --split val --threshold 24

Then, to compare with the other methods:

    python exp_analysis_demo/eval_all.py --gt-yolo dataset/labels/test \\
        --manifest common/dataset/dataset_manifest.csv --preds-dir results/preds --out results/eval
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from common.data_loader import load_dataset  # noqa: E402
from method1.method1 import detect          # noqa: E402  (unchanged, read-only use)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.path.join(REPO_ROOT, "common", "dataset"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "preds", "method1_ml"))
    ap.add_argument("--threshold", type=int, default=21, help="same default as method1.py")
    ap.add_argument("--min-area", type=int, default=6, help="same default as method1.py")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N images")
    args = ap.parse_args(argv)

    try:
        import cv2  # noqa: F401
    except ImportError:
        raise SystemExit("[export] opencv is required (pip install opencv-python)")

    os.makedirs(args.out, exist_ok=True)
    preds, latencies = {}, []
    n_boxes = 0
    t0 = time.perf_counter()
    for i, (path, img, gt) in enumerate(load_dataset(Path(args.dataset), args.split)):
        if args.limit and i >= args.limit:
            break
        start = time.perf_counter()
        det = detect(img, args.min_area, threshold=args.threshold)
        latencies.append((time.perf_counter() - start) * 1000.0)
        boxes = [[round(float(v), 2) for v in b[:4]] for b, _ in det]
        scores = [round(float(s), 4) for _, s in det]
        preds[path.name] = {"boxes": boxes, "scores": scores}
        n_boxes += len(boxes)
        if (i + 1) % 50 == 0:
            print("\r[export] %d images, %d boxes" % (i + 1, n_boxes), end="", file=sys.stderr)
    wall = time.perf_counter() - t0
    print("\r[export] %d images, %d boxes, wall %.1fs        " % (len(preds), n_boxes, wall))

    with open(os.path.join(args.out, "test.json" if args.split == "test" else
                           "%s.json" % args.split), "w", encoding="utf-8") as f:
        json.dump(preds, f)

    timing = {
        "train_time_s": 0.0,
        "train_images": 0,
        "inference_ms_per_image": round(float(np.median(latencies)), 3) if latencies else 0.0,
        "inference_ms_total": round(float(sum(latencies)), 2),
        "fps": round(1000.0 / max(float(np.median(latencies)), 1e-6), 1) if latencies else 0.0,
        "device": "CPU (%d threads, path=%s)" % (cv2.getNumThreads(), cv2.__file__[:40]),
        "note": "no training; threshold=%d min_area=%d; single-thread latency on %d %s images"
                % (args.threshold, args.min_area, len(preds), args.split),
    }
    with open(os.path.join(args.out, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=1)
    print("[export] wrote %s + timing.json (median %.2f ms/image)"
          % (args.out, timing["inference_ms_per_image"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
