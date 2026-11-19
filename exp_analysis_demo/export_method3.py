#!/usr/bin/env python3
"""Export method 3 (RT-DETRv2) predictions in the shared format.

Method 3 lives in ``method3/`` (a vendored RT-DETRv2 fork).  This
wrapper builds the *deploy* model exactly like ``references/deploy/rtdetrv2_torch.py``
does -- load the checkpoint's ema/module state, call ``cfg.model.deploy()`` and
``cfg.postprocessor.deploy()`` -- but loads it once and walks a whole folder,
writing

    results/preds/method3_rtdetr/test.json
    results/preds/method3_rtdetr/timing.json

    set PYTHONPATH=method3          (Windows)  /  export PYTHONPATH=...
    python exp_analysis_demo/export_method3.py \
        --weights method3/output/best.pth \
        --images dataset/images/test \
        --config method3/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco_stride4.yml

The evaluation size is read from the selected configuration's
``eval_spatial_size`` setting.
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


def build_model(weights, config, device):
    import torch
    import torch.nn as nn

    sys.path.insert(0, os.path.join(REPO_ROOT, "method3"))

    from src.core import YAMLConfig

    cfg = YAMLConfig(config, resume=weights)
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]

    model_state = cfg.model.state_dict()
    state = dict(state)
    for key in ("decoder.anchors", "decoder.valid_mask"):
        if key in state and key in model_state and state[key].shape != model_state[key].shape:
            state.pop(key)
    cfg.model.load_state_dict(state, strict=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    return Model().to(device).eval(), cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--config", default=os.path.join(
        REPO_ROOT, "method3", "configs", "rtdetrv2",
        "rtdetrv2_r50vd_m_7x_coco_stride4.yml"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "preds", "method3_rtdetr"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--train-time-s", type=float, default=None)
    ap.add_argument("--train-images", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    import torchvision.transforms as T
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMG_EXT))
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise SystemExit("[export] no images in %s" % args.images)

    model, cfg = build_model(args.weights, args.config, device)
    size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    tf = T.Compose([T.Resize(tuple(size)), T.ToTensor()])
    print("[export] %d images, eval size %s, device %s" % (len(files), size, device))

    preds, latencies = {}, []
    for i, f in enumerate(files):
        im = Image.open(os.path.join(args.images, f)).convert("RGB")
        w, h = im.size
        data = tf(im)[None].to(device)
        orig = torch.tensor([w, h])[None].to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            labels, boxes, scores = model(data, orig)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        preds[f] = {"boxes": [[round(float(v), 2) for v in b] for b in boxes[0]],
                    "scores": [round(float(s), 4) for s in scores[0]]}
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
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "note": "RT-DETRv2 deploy mode, eval size %s, batch=1" % (size,),
    }
    with open(os.path.join(args.out, "timing.json"), "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=1)
    print("[export] wrote %s + timing.json (median %.2f ms/image)"
          % (args.out, timing["inference_ms_per_image"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
