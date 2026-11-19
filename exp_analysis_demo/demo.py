#!/usr/bin/env python3
"""Recordable demo: one test image at a time, through all three methods.

Four panels per frame:

    [ ground truth ] [ method 1 classic CV ] [ method 2 YOLOv8s ] [ method 3 RT-DETRv2 ]

Green  = ground-truth box
Blue   = prediction matched to a ground-truth box (IoU >= 0.5)
Orange = unmatched prediction (false positive)
Red    = ground-truth box nobody found (missed)

Two ways to use it:

    # 1) interactive window (record your screen, press space to advance)
    python exp_analysis_demo/demo.py --show

    # 2) render a clip directly (no window, no clicking)
    python exp_analysis_demo/demo.py --video demo.mp4 --seconds 3

Both show the same curated images, then a summary frame with the group numbers
read from results/eval/summary.csv.

Everything runs offline from local checkpoints + common/dataset/.  First call loads ~650 MB of
checkpoints (10-20 s), so the load happens before the first frame.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "exp_analysis_demo"))
sys.path.insert(0, REPO)

from common.metrics import match_image  # noqa: E402
from exp_analysis_demo.eval_all import load_gt_yolo, load_manifest  # noqa: E402

PANEL, MARGIN, HEADER, CAPTION, FOOTER = 384, 14, 78, 44, 46
W = 4 * PANEL + 5 * MARGIN
H = HEADER + PANEL + CAPTION + FOOTER

# Hand-picked so the clip tells a story instead of showing average frames.
CURATED = [
    "beach_bird_1790.jpg",  # 66 birds: the dense-scene failure
    "beach_bird_1758.jpg",  # 26 birds on rocks: same story, smaller crowd
    "beach_bird_1661.jpg",  # 9 birds, water behind: method 1 explodes, method 2 is clean
    "beach_bird_1677.jpg",  # 0 birds: false positives on an empty beach
]
CURATED_FALLBACK_COUNT = 3  # if a curated name is missing, add one typical frame

GREEN, BLUE, ORANGE, RED = (0, 214, 90), (255, 170, 40), (30, 140, 255), (60, 60, 240)


def font(size: int, bold: bool = False):
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for n in names:
        p = os.path.join("C:\\Windows\\Fonts", n)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_BIG, F_MID, F_SMALL = font(26, True), font(17, True), font(15)


def pick_images(split_dir: str, manifest: dict, wanted: list[str]) -> list[str]:
    have = set(os.listdir(split_dir))
    chosen = [n for n in wanted if n in have]
    # one typical frame (3 birds), deterministic: first by name
    for name in sorted(have):
        if name in chosen:
            continue
        if manifest.get(name, {}).get("count") == CURATED_FALLBACK_COUNT:
            chosen.append(name)
            break
    return chosen or sorted(have)[:4]


REQUIRED = ["numpy", "PIL", "cv2", "torch", "torchvision", "yaml", "scipy"]


def preflight():
    """Fail with one readable line instead of a 20-line import traceback."""
    import importlib
    missing = []
    for mod in REQUIRED:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    if not missing:
        return
    names = {"PIL": "Pillow", "cv2": "opencv-python", "yaml": "PyYAML", "torch": "torch",
             "torchvision": "torchvision", "numpy": "numpy", "scipy": "scipy", "rich": "rich"}
    pkgs = " ".join(names.get(m, m) for m in missing)
    print("")
    print("[x] missing Python packages: %s" % ", ".join(missing))
    print("")
    print("    install them with:")
    print("      setup.bat            (does it for you, GPU or CPU)")
    print("    or, into the interpreter you are using:")
    print("      python -m pip install %s" % pkgs)
    print("")
    raise SystemExit(3)



def load_models(args):
    from ultralytics import YOLO

    print("[demo] loading YOLOv8s ...", flush=True)
    yolo = YOLO(args.yolo_weights)

    print("[demo] loading RT-DETRv2 ...", flush=True)
    from exp_analysis_demo.export_method3 import build_model

    m3, cfg = build_model(args.m3_weights, args.m3_config, "cuda" if _cuda() else "cpu")
    size = tuple(cfg.yaml_cfg.get("eval_spatial_size", [640, 640]))
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(size), T.ToTensor()])

    print("[demo] loading method 1 (no weights) ...", flush=True)
    from method1.method1 import detect as m1_detect

    def run_m1(bgr):
        t0 = time.perf_counter()
        dets = m1_detect(bgr, area_min=6, threshold=21)
        ms = (time.perf_counter() - t0) * 1000
        boxes = np.array([d[0] for d in dets], dtype=float).reshape(-1, 4)
        scores = np.array([d[1] for d in dets], dtype=float)
        return boxes, scores, ms

    import torch

    def run_m2(bgr):
        if _cuda():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = yolo.predict(bgr, imgsz=640, conf=args.conf, iou=0.7, verbose=False)[0]
        if _cuda():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        b = r.boxes.xyxy.cpu().numpy().astype(float)
        s = r.boxes.conf.cpu().numpy().astype(float)
        return b, s, ms

    def run_m3(rgb):
        w, h = rgb.size
        data = tf(rgb)[None].to("cuda" if _cuda() else "cpu")
        orig = torch.tensor([w, h])[None].to(data.device)
        if _cuda():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _, boxes, scores = m3(data, orig)
        if _cuda():
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        b = boxes[0].float().cpu().numpy().astype(float)
        s = scores[0].float().cpu().numpy().astype(float)
        keep = s >= args.conf
        return b[keep], s[keep], ms

    return {"method1_ml": run_m1, "method2_yolo": run_m2, "method3_rtdetr": run_m3}


def _cuda():
    import torch
    return torch.cuda.is_available()


def draw_panel(base: Image.Image, gt: np.ndarray, pred: np.ndarray, labels: np.ndarray,
               n_fn: int, title: str, stats: str, cap: str) -> Image.Image:
    img = Image.new("RGB", (PANEL, PANEL + CAPTION), (18, 18, 20))
    img.paste(base.resize((PANEL, PANEL), Image.BICUBIC), (0, 0))
    d = ImageDraw.Draw(img)
    for g in gt:
        d.rectangle(list(map(float, g)), outline=GREEN, width=2)
    for b, hit in zip(pred, labels):
        d.rectangle(list(map(float, b)), outline=BLUE if hit else ORANGE, width=2)
    if n_fn:
        txt = "%d missed" % n_fn
        tw = d.textlength(txt, font=F_SMALL)
        d.text((PANEL - tw - 10, 10), txt, font=F_SMALL, fill=RED,
               stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((8, 8), title, font=F_MID, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((8, PANEL - 26), stats, font=F_SMALL, fill=(240, 240, 240),
           stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((8, PANEL + 10), cap, font=F_SMALL, fill=(200, 200, 200))
    return img


def frame_no_gt(path: str, runners: dict, conf: float) -> Image.Image:
    """Three panels for an image that has no ground truth (a photo from anywhere)."""
    pil = Image.open(path).convert("RGB")
    if max(pil.size) > 1280:
        pil.thumbnail((1280, 1280), Image.LANCZOS)
    bgr = np.array(pil)[:, :, ::-1].copy()

    canvas = Image.new("RGB", (3 * PANEL + 4 * MARGIN, HEADER + PANEL + CAPTION), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    d.text((MARGIN, 16), "Your image - %d x %d px" % pil.size, font=F_BIG, fill=(255, 255, 255))
    d.text((MARGIN, 50), "no ground truth for this picture, so only predictions are drawn",
           font=F_SMALL, fill=(170, 170, 175))

    pretty = {"method1_ml": "Method 1 - classic CV", "method2_yolo": "Method 2 - YOLOv8s",
              "method3_rtdetr": "Method 3 - RT-DETRv2"}
    thumbs = {"method1_ml": "creates boxes from contrast",
              "method2_yolo": "one box per confident bird",
              "method3_rtdetr": "one query per bird, no NMS"}
    sx = PANEL / float(pil.size[0])
    for i, key in enumerate(("method1_ml", "method2_yolo", "method3_rtdetr")):
        pred, scores, ms = runners[key](bgr if key == "method1_ml" else pil)
        n = len(pred)
        panel = Image.new("RGB", (PANEL, PANEL))
        panel.paste(pil.resize((PANEL, PANEL), Image.BICUBIC), (0, 0))
        canvas.paste(draw_panel(panel, np.zeros((0, 4)), pred * sx,
                                np.zeros((n,), bool), 0, pretty[key],
                                "%d boxes" % n, thumbs[key]),
                     (MARGIN + i * (PANEL + MARGIN), HEADER))
    return canvas


def frame_for(name: str, img_dir: str, gt_all: dict, runners: dict, conf: float,
              hint: bool = True) -> Image.Image:
    pil = Image.open(os.path.join(img_dir, name)).convert("RGB")
    bgr = np.array(pil)[:, :, ::-1].copy()
    gt = np.array(gt_all.get(name, []), dtype=float).reshape(-1, 4)

    canvas = Image.new("RGB", (W, H), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    d.text((MARGIN, 16), "Beach bird detection - same test image, four views", font=F_BIG, fill=(255, 255, 255))
    d.text((MARGIN, 50), "%s   |   %d ground-truth birds   |   predictions shown at conf %.2f"
           % (name, len(gt), conf), font=F_SMALL, fill=(170, 170, 175))
    if hint:
        d.text((W - 220, 50), "space = next   q = quit", font=F_SMALL, fill=(120, 120, 125))

    panel = Image.new("RGB", (PANEL, PANEL))
    panel.paste(pil.resize((PANEL, PANEL), Image.BICUBIC), (0, 0))
    sx = PANEL / 512.0
    canvas.paste(draw_panel(panel, gt * sx, np.zeros((0, 4)), np.zeros((0,), np.int64), 0,
                            "Ground truth",
                            "%d annotated birds" % len(gt) if len(gt) else "no birds in this frame",
                            "every green box is a label we score against"), (MARGIN, HEADER))

    names = ["method1_ml", "method2_yolo", "method3_rtdetr"]
    pretty = {"method1_ml": "Method 1 - classic CV", "method2_yolo": "Method 2 - YOLOv8s",
              "method3_rtdetr": "Method 3 - RT-DETRv2"}
    for i, key in enumerate(names):
        if key == "method1_ml":
            pred, scores, ms = runners[key](bgr)
        else:
            pred, scores, ms = runners[key](pil)
        x = MARGIN + (i + 1) * (PANEL + MARGIN)
        base = Image.new("RGB", (PANEL, PANEL))
        base.paste(pil.resize((PANEL, PANEL), Image.BICUBIC), (0, 0))
        sx = PANEL / 512.0
        pred_s = pred * sx
        _, labels, _, n_fn = match_image(pred_s, scores, gt * sx)
        tp, n_pred = int(np.sum(labels)) if len(labels) else 0, len(pred_s)
        canvas.paste(draw_panel(base, gt * sx, pred_s, labels, n_fn, pretty[key],
                                "tp %d   fp %d   fn %d" % (tp, n_pred - tp, n_fn),
                                "blue = matched, orange = false positive" if key != "method3_rtdetr"
                                else "one query per bird, no NMS"), (x, HEADER))
    d.text((MARGIN, H - 32), "Test split, 240 images never used for training.  Boxes are raw model output; "
                             "mAP is computed separately by the shared evaluator.",
           font=F_SMALL, fill=(150, 150, 155))
    return canvas


def summary_frame() -> Image.Image:
    rows = []
    path = os.path.join(REPO, "results", "eval", "summary.csv")
    want = {"method1_ml", "method2_yolo", "method3_rtdetr"}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("method") in want:
                    rows.append(r)
    order = {"method1_ml": 0, "method2_yolo": 1, "method3_rtdetr": 2}
    rows.sort(key=lambda r: order.get(r["method"], 9))

    canvas = Image.new("RGB", (W, 620), (18, 18, 20))
    d = ImageDraw.Draw(canvas)
    f_title, f_row, f_note = font(30, True), font(20, True), font(18)
    d.text((MARGIN, 24), "Same 240 test images, one evaluator: what the report shows", font=f_title,
           fill=(255, 255, 255))
    d.text((MARGIN, 66), "single class, IoU 0.5, all-point interpolation, raw boxes (no confidence cut)",
           font=f_note, fill=(160, 160, 168))
    y = 116
    cols = [("method", 430), ("mAP@0.5", 200), ("training time", 260), ("inference", 200)]
    x = MARGIN
    for name, w in cols:
        d.text((x, y), name, font=f_note, fill=(150, 150, 158))
        x += w
    y += 34
    pretty = {"method1_ml": "Method 1 - classic CV", "method2_yolo": "Method 2 - YOLOv8s",
              "method3_rtdetr": "Method 3 - RT-DETRv2"}
    for r in rows:
        ap = float(r["ap50"]) if r.get("ap50") else 0.0
        tt = r.get("train_time_s") or ""
        if tt not in ("", "0", "0.0"):
            tt = "%.1f min" % (float(tt) / 60)
        elif r["method"] == "method1_ml":
            tt = "none (no learning)"
        else:
            tt = "not measured here"
        ms = r.get("inference_ms_per_image") or ""
        ms = "%.1f ms" % float(ms) if ms else "n/a"
        vals = [pretty.get(r["method"], r["method"]), "%.4f" % ap, tt, ms]
        x = MARGIN
        for (name, w), v in zip(cols, vals):
            d.text((x, y), v, font=f_row if name == "method" else f_note,
                   fill=GREEN if name == "mAP@0.5" else (235, 235, 235))
            x += w
        y += 46
    y += 18
    for line in [
        "Method 1 is a documented negative result: an edge/texture detector, not a bird detector -",
        "305,288 boxes over 240 images and almost no IoU >= 0.5 hit.",
        "",
        "Method 3 wins because one query per bird keeps confidence concentrated in a crowd;",
        "Method 2 loses dense-scene birds to its own confidence threshold, not to a lack of boxes.",
    ]:
        d.text((MARGIN, y), line, font=f_note, fill=(190, 190, 195))
        y += 27
    return canvas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="single image name or path (skips the curated set)")
    ap.add_argument("--video", help="write an mp4 here instead of opening a window")
    ap.add_argument("--seconds", type=float, default=3.0, help="hold per image in the video")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show", action="store_true", help="open an interactive window")
    ap.add_argument("--yolo-weights", default="method2/results/train_runs/single_stage_baseline/weights/best.pt")
    ap.add_argument("--m3-weights", default="method3/output/default_params.pth")
    ap.add_argument("--m3-config", default="method3/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml")
    ap.add_argument("--split", default="test")
    args = ap.parse_args(argv)

    preflight()

    img_dir = os.path.join("common", "dataset", "images", args.split)
    labels_dir = os.path.join("common", "dataset", "labels", args.split)
    manifest_path = os.path.join("common", "dataset", "dataset_manifest.csv")
    manifest = load_manifest(manifest_path) if os.path.exists(manifest_path) else {}
    gt_all = load_gt_yolo(labels_dir, manifest_path if os.path.exists(manifest_path) else None, img_dir)

    if args.image:
        if not os.path.exists(args.image):
            print("[demo] no such file: %s" % args.image, flush=True)
            return 2
        names = [os.path.basename(args.image)]
        if not os.path.exists(os.path.join(img_dir, names[0])):
            img_dir = os.path.dirname(args.image)
    else:
        names = pick_images(img_dir, manifest, CURATED)

    print("[demo] %d images: %s" % (len(names), ", ".join(names)), flush=True)
    runners = load_models(args)

    # one untimed pass per model: the first call pays CUDA context + kernel init
    warm_pil = Image.open(os.path.join(img_dir, names[0])).convert("RGB")
    warm_bgr = np.array(warm_pil)[:, :, ::-1].copy()
    for key, fn in runners.items():
        fn(warm_bgr if key == "method1_ml" else warm_pil)
    print("[demo] warm-up done", flush=True)

    single_new = bool(args.image) and names[0] not in gt_all
    if single_new:
        print("[demo] no ground truth for %s - drawing predictions only" % names[0], flush=True)
        frames = [frame_no_gt(args.image, runners, args.conf)]
    else:
        frames = [frame_for(n, img_dir, gt_all, runners, args.conf, hint=not args.video)
                  for n in names]
        frames.append(summary_frame())   # the 240-image summary is meaningless for a new photo
    print("[demo] rendered %d frames" % len(frames), flush=True)

    if args.video:
        import cv2
        big = (max(f.width for f in frames), max(f.height for f in frames))
        out = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, big)
        hold = max(1, int(args.seconds * args.fps))
        for i, f in enumerate(frames):
            canvas = Image.new("RGB", big, (18, 18, 20))
            canvas.paste(f, (0, 0))
            arr = np.array(canvas)[:, :, ::-1]
            n_hold = hold if i < len(frames) - 1 else max(hold, args.fps * 4)
            for _ in range(n_hold):
                out.write(arr)
        out.release()
        print("[demo] wrote %s (%.1f s)" % (args.video, len(frames) * args.seconds), flush=True)
        return 0

    import cv2
    try:
        cv2.namedWindow("beach bird demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("beach bird demo", W, max(f.height for f in frames))
    except cv2.error as exc:
        # opencv-python-headless and opencv-python install the same `cv2`
        # package; when both are present the headless build wins and
        # namedWindow raises "The function is not implemented".
        print("[demo] cannot open a window: this cv2 has no GUI support.", flush=True)
        print("        fix: pip uninstall -y opencv-python-headless", flush=True)
        print("             pip install --force-reinstall opencv-python", flush=True)
        print("        or render a clip instead:  --video demo.mp4 --seconds 3", flush=True)
        print("        detail: %s" % exc, flush=True)
        return 2
    i = 0
    while True:
        f = frames[i]
        canvas = Image.new("RGB", (W, max(g.height for g in frames)), (18, 18, 20))
        canvas.paste(f, (0, 0))
        cv2.imshow("beach bird demo", np.array(canvas)[:, :, ::-1])
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord("q")):
            break
        if k in (32, 83, ord("d")):
            i = min(i + 1, len(frames) - 1)
        elif k in (81, ord("a")):
            i = max(i - 1, 0)
        elif k in (ord("r"),):
            i = 0
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
