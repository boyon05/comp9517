"""Re-measure method 2 and method 3 latency on an idle GPU, same protocol as timing_min."""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, "exp_analysis_demo")
sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image

IMG = "common/dataset/images/test"
N_PASSES = 3
names = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".jpg", ".jpeg", ".png")))
pils = [Image.open(os.path.join(IMG, n)).convert("RGB") for n in names]


def min_of_pass_medians(run_once):
    medians = []
    for _ in range(N_PASSES):
        ts = run_once()
        medians.append(statistics.median(ts))
    return min(medians), medians


# ---- method 2
from ultralytics import YOLO

y = YOLO("method2/results/train_runs/single_stage_baseline/weights/best.pt")


def run2():
    ts = []
    for im in pils:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y.predict(im, conf=0.01, iou=0.7, imgsz=640, verbose=False, device=0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return ts


best2, all2 = min_of_pass_medians(run2)
print("[lat] method2 min-of-3 = %.2f ms  (passes %s)" % (best2, ["%.1f" % m for m in all2]))

# ---- method 3
import torchvision.transforms as T
from exp_analysis_demo.export_method3 import build_model

m3, cfg = build_model("method3/output/default_params.pth",
                  "method3/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",
                      "cuda")
size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
tf = T.Compose([T.Resize(tuple(size)), T.ToTensor()])


def run3():
    ts = []
    with torch.no_grad():
        for im in pils:
            w, h = im.size
            data = tf(im)[None].cuda()
            orig = torch.tensor([w, h])[None].cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            m3(data, orig)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
    return ts


best3, all3 = min_of_pass_medians(run3)
print("[lat] method3 min-of-3 = %.2f ms  (passes %s)" % (best3, ["%.1f" % m for m in all3]))

for path, val, note in (
    ("results/preds/method2_yolo/timing.json", best2,
     "min of 3 full-pass medians on an idle GPU, conf>=0.01, imgsz 640, includes pre/post"),
    ("results/preds/method3_rtdetr/timing.json", best3,
     "min of 3 full-pass medians on an idle GPU, eval size 640, batch 1"),
):
    d = json.load(open(path))
    d["inference_ms_per_image"] = round(val, 3)
    d["fps"] = round(1000.0 / max(val, 1e-6), 1)
    d["note"] = note
    if "train_note" in d:
        d["note"] += "; train time author-reported (see train_note)"
    json.dump(d, open(path, "w"), indent=1)
    print("[lat] wrote", path)
