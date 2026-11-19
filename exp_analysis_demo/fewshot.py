#!/usr/bin/env python3
"""Few-shot controlled scaling experiment

Spec: "you are free to use only a subset of the data, for example 75% or even
just 50% ... as long as you clearly report your approach".  Instead of treating
that as a compute excuse we run it as a controlled experiment: the *same*
recipe (YOLOv8s, 30 epochs, advanced augmentation, seed 0) trained on 100%, 75%
and 50% of the training split, always validating on the full val split, and
touching test exactly once at the end.  The 75%/50% subsets come from
``splits/train_{75,50}.json`` which are stratified over the official density
bins with whole source-groups kept together (no near-duplicate leakage between
the kept and dropped halves).

    python exp_analysis_demo/fewshot.py                        # 100, 75, 50
    python exp_analysis_demo/fewshot.py --ratios 50 --epochs 30
    python exp_analysis_demo/fewshot.py --export-only          # reuse weights

Outputs, per ratio ``r``:

    results/fewshot/<r>/            ultralytics run (weights/, results.csv)
    results/preds/fewshot_<r>/{test,timing}.json
    results/fewshot/summary.md      ratio | mAP@0.5 | train min | test ms/img
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUG_YAML = os.path.join(REPO_ROOT, "method2", "configs", "aug_advanced.yaml")
SPLIT_JSON = {75: os.path.join(REPO_ROOT, "exp_analysis_demo", "splits", "train_75.json"),
              50: os.path.join(REPO_ROOT, "exp_analysis_demo", "splits", "train_50.json")}


def write_image_list(ratio, dataset_dir, out_dir):
    """Return the path of a txt file listing images ultralytics should train on."""
    img_dir = os.path.join(dataset_dir, "images", "train")
    all_names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if ratio >= 100:
        keep = all_names
    else:
        with open(SPLIT_JSON[ratio], "r", encoding="utf-8") as f:
            meta = json.load(f)
        wanted = {os.path.basename(n) for n in meta["file_names"]}
        keep = [n for n in all_names if n in wanted]
    os.makedirs(out_dir, exist_ok=True)
    txt = os.path.join(out_dir, "train_%d.txt" % ratio)
    with open(txt, "w", encoding="utf-8") as f:
        for n in keep:
            f.write(os.path.join(img_dir, n).replace("\\", "/") + "\n")
    print("[fewshot] %3d%% -> %d images (%s)" % (ratio, len(keep), txt))
    return txt, len(keep)


def write_data_yaml(ratio, dataset_dir, txt_path, out_dir):
    """Ultralytics resolves relative entries against `path:`, so everything here
    is absolute - otherwise a relative --out is read as dataset/<out>/..."""
    yml = os.path.join(out_dir, "data_%d.yaml" % ratio)
    root = os.path.abspath(dataset_dir).replace("\\", "/")
    with open(yml, "w", encoding="utf-8") as f:
        f.write("path: %s\ntrain: %s\nval: %s\nnames:\n  0: bird\n"
                % (root, os.path.abspath(txt_path).replace("\\", "/"),
                   os.path.join(root, "images", "val")))
    return yml


def run_name(ratio, seed):
    """seed 0 keeps the historical directory names so old artifacts stay valid."""
    return str(ratio) if seed == 0 else "%d_seed%d" % (ratio, seed)


def preds_name(ratio, seed):
    return "fewshot_%d" % ratio if seed == 0 else "fewshot_%d_s%d" % (ratio, seed)


def train_one(ratio, n_images, args):
    from ultralytics import YOLO

    out_dir = os.path.abspath(args.out)
    weights = os.path.join(out_dir, run_name(ratio, args.seed), "weights", "best.pt")
    if os.path.isfile(weights) and not args.force:
        print("[fewshot] %3d%% already trained, reusing %s" % (ratio, weights))
        return weights, None, None

    txt, _ = write_image_list(ratio, args.dataset, os.path.join(out_dir, "lists"))
    data_yml = write_data_yaml(ratio, args.dataset, txt, os.path.join(out_dir, "lists"))
    params = {}
    if os.path.isfile(AUG_YAML):
        import yaml
        params = yaml.safe_load(open(AUG_YAML, encoding="utf-8")) or {}

    model = YOLO(args.model)
    t0 = time.perf_counter()
    model.train(data=data_yml, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                device=args.device, seed=args.seed, workers=args.workers,
                project=out_dir, name=run_name(ratio, args.seed), exist_ok=True, plots=True,
                verbose=True, **params)
    train_s = time.perf_counter() - t0
    print("[fewshot] %3d%% trained in %.1f min" % (ratio, train_s / 60))
    return weights, train_s, n_images


def export_test(ratio, weights, args):
    """Run the trained model over the test split with the shared export rules."""
    import statistics

    import torch
    from ultralytics import YOLO

    test_dir = os.path.join(args.dataset, "images", "test")
    files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    model = YOLO(weights)
    device = 0 if torch.cuda.is_available() else "cpu"
    model.predict([os.path.join(test_dir, files[0])], conf=args.conf, imgsz=args.imgsz,
                  device=device, verbose=False)  # warmup
    preds, lat = {}, []
    for f in files:
        p = os.path.join(test_dir, f)
        if device == 0:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = model.predict(p, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                          device=device, verbose=False)[0]
        if device == 0:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000.0)
        preds[f] = {"boxes": [[round(float(v), 2) for v in b] for b in r.boxes.xyxy.cpu().numpy()],
                    "scores": [round(float(s), 4) for s in r.boxes.conf.cpu().numpy()]}

    out = os.path.join(REPO_ROOT, "results", "preds", preds_name(ratio, args.seed))
    os.makedirs(out, exist_ok=True)
    json.dump(preds, open(os.path.join(out, "test.json"), "w", encoding="utf-8"))
    timing = {"train_time_s": None, "train_images": None,
              "inference_ms_per_image": round(float(statistics.median(lat)), 3),
              "fps": round(1000.0 / max(float(statistics.median(lat)), 1e-6), 1),
              "device": str(torch.cuda.get_device_name(0)) if device == 0 else "cpu",
              "note": "few-shot %d%% export, conf>=%.2f" % (ratio, args.conf)}
    json.dump(timing, open(os.path.join(out, "timing.json"), "w"), indent=1)
    return out, timing["inference_ms_per_image"], sum(len(v["boxes"]) for v in preds.values())


def evaluate(ratio, seed):
    sys.path.insert(0, REPO_ROOT)
    from common.metrics import evaluate as ev
    from exp_analysis_demo.eval_all import load_gt_yolo, load_preds

    labels = os.path.join(REPO_ROOT, "common", "dataset", "labels", "test")
    manifest = os.path.join(REPO_ROOT, "common", "dataset", "dataset_manifest.csv")
    preds = os.path.join(REPO_ROOT, "results", "preds", preds_name(ratio, seed), "test.json")
    if not os.path.isfile(preds):
        return None
    return ev(load_gt_yolo(
        labels,
        manifest if os.path.isfile(manifest) else None,
        os.path.join(REPO_ROOT, "common", "dataset", "images", "test"),
    ), load_preds(preds), 0.5)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ratios", default="100,75,50")
    ap.add_argument("--dataset", default=os.path.join(REPO_ROOT, "common", "dataset"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "fewshot"))
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0,
                    help="training seed; non-zero seeds get a _seedN suffix on "
                         "the run dir and on the exported predictions")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--force", action="store_true", help="retrain even if weights exist")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    ratios = [int(r) for r in args.ratios.split(",") if r.strip()]
    rows = []
    for ratio in ratios:
        out_dir = os.path.abspath(args.out)
        txt, n_images = write_image_list(ratio, args.dataset, os.path.join(out_dir, "lists"))
        weights = os.path.join(out_dir, run_name(ratio, args.seed), "weights", "best.pt")
        train_s, n_used = None, n_images
        if not args.export_only:
            w, train_s, n_used = train_one(ratio, n_images, args)
            weights = w
            if n_used is None:  # reused weights: the list length is still known
                n_used = n_images
        if not os.path.isfile(weights):
            print("[fewshot] %d%%: no weights at %s, skipping export" % (ratio, weights))
            continue
        out, ms, n_boxes = export_test(ratio, weights, args)
        res = evaluate(ratio, args.seed)
        ap50 = res["ap50"] if res else float("nan")
        print("[fewshot] %3d%%  mAP@0.5 %.4f  train %s  %.1f ms/img  %d boxes"
              % (ratio, ap50, "%.1f min" % (train_s / 60) if train_s else "reused",
                 ms, n_boxes))
        rows.append((ratio, n_used, ap50, train_s, ms, n_boxes))

    md = ["# Few-shot scaling experiment (test split, mAP@0.5)", "",
          "Same recipe for every row: YOLOv8s, 30 epochs, advanced augmentation, seed %d;"
          % args.seed,
          "val is always the full val split; test is touched once.", "",
          "| Train data | Images | mAP@0.5 | Train (min) | Test (ms/img) | Boxes on test |",
          "|---|---|---|---|---|---|"]
    for ratio, n_used, ap50, train_s, ms, n_boxes in rows:
        md.append("| %d%% | %d | %.4f | %s | %.1f | %d |"
                  % (ratio, n_used, ap50,
                     "%.1f" % (train_s / 60) if train_s else "reused", ms, n_boxes))
    path = os.path.join(args.out, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("[fewshot] wrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
