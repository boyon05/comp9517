#!/usr/bin/env python3
"""Export method 2 (YOLOv8) predictions in the shared format.

Same contract as every other method (see the prediction-format section in
`exp_analysis_demo/README.md`):

    results/preds/method2_yolo/test.json    {image_name: {boxes, scores}}
    results/preds/method2_yolo/timing.json  per-image latency (+ training time)

Important: export EVERY box with score >= 0.01. A previous evaluation used
``model.val(conf=0.25)``, which silently drops low-confidence boxes and makes
the number incomparable with methods 3/1.  mAP@0.5 is an integral over
confidence; the threshold has to be applied once, at evaluation time.

    python exp_analysis_demo/export_method2.py \\
        --weights weights/yolov8s_advanced_best.pt \\
        --images dataset/images/test --out results/preds/method2_yolo
    python exp_analysis_demo/export_method2.py --weights ... --images ... --split val

Training time is not measured here -- pass it in with --train-time-s (it comes
from the training log / results.csv, e.g. 7.87 min = 472 s).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="best.pt from training")
    ap.add_argument("--images", required=True, help="folder of images (one split)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "preds", "method2_yolo"))
    ap.add_argument("--split", default="test", help="label used in the output filename")
    ap.add_argument("--conf", type=float, default=0.01, help="export floor (keep at 0.01)")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU (ultralytics default)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--train-time-s", type=float, default=None)
    ap.add_argument("--train-images", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N images only")
    args = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO

    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMG_EXT))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("[export] no images in %s" % args.images)
    paths = [os.path.join(args.images, f) for f in files]

    device = 0 if torch.cuda.is_available() else "cpu"
    print("[export] %d images, weights=%s, device=%s"
          % (len(files), os.path.basename(args.weights), device))
    model = YOLO(args.weights)

    # warm up so the first image does not pay the cudnn/cpuid setup cost
    model.predict(paths[:1], conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                  device=device, verbose=False)

    preds, latencies = {}, []
    for i, (f, p) in enumerate(zip(files, paths)):
        if device == 0:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(p, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                          device=device, verbose=False)[0]
        if device == 0:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        xyxy = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
        preds[f] = {"boxes": [[round(float(v), 2) for v in b] for b in xyxy],
                    "scores": [round(float(s), 4) for s in conf]}
        if (i + 1) % 50 == 0:
            print("\r[export] %d/%d" % (i + 1, len(files)), end="", file=sys.stderr)

    n_boxes = sum(len(v["boxes"]) for v in preds.values())
    print("\r[export] %d images, %d boxes" % (len(preds), n_boxes))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "%s.json" % args.split), "w", encoding="utf-8") as f:
        json.dump(preds, f)
    timing = {
        "train_time_s": args.train_time_s,
        "train_images": args.train_images,
        "inference_ms_per_image": round(float(statistics.median(latencies)), 3),
        "inference_ms_total": round(float(sum(latencies)), 2),
        "fps": round(1000.0 / max(float(statistics.median(latencies)), 1e-6), 1),
        "device": str(torch.cuda.get_device_name(0)) if device == 0 else "cpu",
        "note": "ultralytics export, conf>=%.2f, imgsz=%d, batch=1, includes pre/post-process"
                % (args.conf, args.imgsz),
    }
    with open(os.path.join(args.out, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=1)
    print("[export] wrote %s + timing.json (median %.2f ms/image)"
          % (args.out, timing["inference_ms_per_image"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
