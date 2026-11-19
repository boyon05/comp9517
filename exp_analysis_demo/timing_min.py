#!/usr/bin/env python3
"""Re-measure per-image latency for every exported few-shot weight, robust to a
busy GPU.

The export pass writes whatever latency it saw, and this laptop shares its 3060
with other processes, so a single pass can read 20-45 ms for the identical model.
Contention can only add time, so the fair estimate is the minimum of several
full passes (each pass reporting its own median).

    python exp_analysis_demo/timing_min.py --passes 3

Rewrites ``inference_ms_per_image`` / ``fps`` / ``note`` in each
``results/preds/fewshot_*/timing.json`` it touches and prints an old-vs-new table.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def weights_for(preds_dir):
    """Map an exported predictions dir to the weight file that produced it."""
    name = os.path.basename(preds_dir)              # fewshot_100 / fewshot_75_s1
    stem = name[len("fewshot_"):]                   # 100 / 75_s1
    ratio, _, seed = stem.partition("_s")
    seed = seed or "0"
    base = os.path.join(REPO_ROOT, "results",
                        "fewshot" if seed == "0" else "fewshot_seed%s" % seed)
    run = ratio if seed == "0" else "%s_seed%s" % (ratio, seed)
    return os.path.join(base, run, "weights", "best.pt")


def time_one(model, files, imgsz, passes):
    import torch

    device = 0 if torch.cuda.is_available() else "cpu"
    model.predict(files[0], conf=0.01, imgsz=imgsz, device=device, verbose=False)  # warmup
    best = None
    for _ in range(passes):
        lat = []
        for path in files:
            if device == 0:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.predict(path, conf=0.01, iou=0.7, imgsz=imgsz, device=device, verbose=False)
            if device == 0:
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(lat)
        best = med if best is None else min(best, med)
    return best, device


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-dir", default=os.path.join(REPO_ROOT, "results", "preds"))
    ap.add_argument("--dataset", default=os.path.join(REPO_ROOT, "common", "dataset"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from ultralytics import YOLO

    test_dir = os.path.join(args.dataset, "images", "test")
    files = sorted(os.path.join(test_dir, f) for f in os.listdir(test_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print("%d test images, %d passes each\n" % (len(files), args.passes))

    for preds_dir in sorted(glob.glob(os.path.join(args.preds_dir, "fewshot_*"))):
        tpath = os.path.join(preds_dir, "timing.json")
        weights = weights_for(preds_dir)
        if not (os.path.isfile(tpath) and os.path.isfile(weights)):
            print("skip %s (no timing.json or no %s)" % (os.path.basename(preds_dir),
                                                         weights))
            continue
        timing = json.load(open(tpath, encoding="utf-8"))
        old = timing.get("inference_ms_per_image")
        ms, device = time_one(YOLO(weights), files, args.imgsz, args.passes)
        print("%-18s old %6s -> min-of-%d %.2f ms  (%.1f FPS)"
              % (os.path.basename(preds_dir),
                 "%.2f" % old if old is not None else "n/a", args.passes, ms, 1000.0 / ms))
        if args.dry_run:
            continue
        timing["inference_ms_per_image"] = round(ms, 3)
        timing["fps"] = round(1000.0 / ms, 2)
        timing["device"] = "cuda:0" if device == 0 else "cpu"
        timing["note"] = ("min of %d full-pass medians (contention-robust); "
                          "score floor 0.01" % args.passes)
        with open(tpath, "w", encoding="utf-8") as f:
            json.dump(timing, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
