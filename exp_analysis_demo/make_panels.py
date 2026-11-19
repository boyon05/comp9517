#!/usr/bin/env python3
"""Render one comparison panel per test image: GT + every method side by side.

Why: the report needs "same image, all methods" figures, and the group needs a
quick way to see which images break which method.  This script reads only the
uniform prediction files (see the prediction-format section in README.md) plus the images, so it does
not care how a method was trained.

    python exp_analysis_demo/make_panels.py \
        --images dataset/images/test \
        --gt-yolo common/dataset/labels/test --manifest common/dataset/dataset_manifest.csv \
        --preds-dir results/preds --out results/panels \
        [--limit 40] [--sort hardest] [--scale 384]

Output:
    <out>/<image_stem>.jpg      one panel: GT | method1 | method2 | method3
    <out>/index.html            contact sheet, hardest images first
    <out>/panel_stats.csv       per image / per method  TP FP FN  precision recall

Colour code inside a method column:
    green box    = matched a GT box (TP)
    orange box   = unmatched prediction (FP, labelled FP)
    green thin   = GT that nobody matched is drawn on the GT column only
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exp_analysis_demo.eval_all import (  # noqa: E402
    load_gt, load_gt_yolo, load_preds, load_manifest)

# panel colours (RGB)
C_GT = (0, 230, 90)
C_TP = (60, 170, 255)
C_FP = (255, 130, 40)
C_MISS = (255, 60, 60)
PANEL_BG = (24, 26, 30)
TEXT = (235, 235, 235)
SPARE = (150, 150, 150)


def _font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def match_one(pred_boxes, pred_scores, gt_boxes, iou_thr=0.5):
    """Greedy confidence-ordered matching -> (per-pred is_tp, unmatched gt idx)."""
    order = np.argsort(-np.asarray(pred_scores, dtype=float)) if len(pred_scores) else []
    iou = iou_matrix(np.asarray(pred_boxes, float).reshape(-1, 4),
                     np.asarray(gt_boxes, float).reshape(-1, 4))
    taken = set()
    is_tp = []
    for i in order:
        best, best_j = 0.0, -1
        for j in range(len(gt_boxes)):
            if j in taken:
                continue
            if iou[i, j] > best:
                best, best_j = iou[i, j], j
        if best_j >= 0 and best >= iou_thr:
            taken.add(best_j)
            is_tp.append(True)
        else:
            is_tp.append(False)
    missed = [j for j in range(len(gt_boxes)) if j not in taken]
    return order, is_tp, missed


def draw_boxes(draw, boxes, colour, width=2, tag=None, font=None, scores=None):
    for k, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=width)
        if tag and font is not None:
            label = tag
            if scores is not None and k < len(scores):
                label = "%s %.2f" % (tag, float(scores[k]))
            draw.text((x1 + 2, max(y1 - 12, 1)), label, fill=colour, font=font)


def panel(img: Image.Image, title: str, columns, scale: int, font_title, font_box,
          max_boxes: int = 40):
    """columns = [(header, gt_boxes_draw, preds_draw)] -> one horizontal strip.

    Only the ``max_boxes`` highest-scoring predictions of a column are drawn:
    method 1 emits ~1200 boxes per image and drawing them all just paints the
    cell black.  The tp/fp/fn header still counts every box.
    """
    w, h = img.size
    ratio = scale / float(w)
    tw, th = int(w * ratio), int(h * ratio)
    pad, head = 6, 26
    foot = 22
    total_w = len(columns) * (tw + pad) + pad
    total_h = head + th + foot + pad
    out = Image.new("RGB", (total_w, total_h), PANEL_BG)
    d = ImageDraw.Draw(out)
    d.text((pad, 4), title[:120], fill=TEXT, font=font_title)
    small = img.resize((tw, th), Image.LANCZOS)
    for c, (header, gt_boxes, pred_info) in enumerate(columns):
        x0 = pad + c * (tw + pad)
        cell = small.copy()
        cd = ImageDraw.Draw(cell)
        # ghost GT in every column so a miss is obvious
        for box in gt_boxes:
            cd.rectangle([box[0] * ratio, box[1] * ratio, box[2] * ratio, box[3] * ratio],
                         outline=C_GT, width=1)
        if pred_info is not None:
            boxes, scores, flags = pred_info
            items = list(zip(boxes, scores, flags))
            shown = items[:max_boxes] if max_boxes else items
            for box, sc, is_tp in shown:
                colour = C_TP if is_tp else C_FP
                col = [box[0] * ratio, box[1] * ratio, box[2] * ratio, box[3] * ratio]
                cd.rectangle(col, outline=colour, width=2)
                cd.text((col[0] + 1, max(col[1] - 9, 0)),
                        "%.2f" % sc, fill=colour, font=font_box)
            if len(items) > len(shown):
                cd.text((4, th - 14), "+%d more" % (len(items) - len(shown)),
                        fill=SPARE, font=font_box)
        out.paste(cell, (x0, head))
        d.text((x0 + 2, head + th + 3), header[:44], fill=TEXT, font=font_title)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="image dir of the split, e.g. dataset/images/test")
    ap.add_argument("--gt", default=None, help="GT json (COCO/flat)")
    ap.add_argument("--gt-yolo", default=None, help="YOLO label dir, e.g. dataset/labels/test")
    ap.add_argument("--manifest", default=None, help="dataset_manifest.csv (image sizes)")
    ap.add_argument("--preds-dir", default="results/preds")
    ap.add_argument("--out", default="results/panels")
    ap.add_argument("--methods", default=None, help="comma separated subset")
    ap.add_argument("--limit", type=int, default=0, help="0 = all images")
    ap.add_argument("--sort", default="hardest",
                    choices=["hardest", "easiest", "name", "dense"],
                    help="hardest = most missed+false boxes summed over methods")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--scale", type=int, default=384, help="width of one column in px")
    ap.add_argument("--jpeg-quality", type=int, default=82)
    ap.add_argument("--rank-ignore", default="",
                    help="comma separated methods left out of the difficulty ranking "
                         "(e.g. method1_ml, whose 1000+ boxes per image swamp the score)")
    ap.add_argument("--max-boxes", type=int, default=40,
                    help="draw only the N highest-scoring boxes per column (0 = all)")
    args = ap.parse_args(argv)

    if args.gt_yolo:
        gts = load_gt_yolo(args.gt_yolo, manifest=args.manifest)
    elif args.gt:
        gts = load_gt(args.gt)
    else:
        ap.error("need --gt-yolo or --gt")

    wanted = [m.strip() for m in args.methods.split(",")] if args.methods else None
    preds: Dict[str, Dict[str, dict]] = {}
    for entry in sorted(os.listdir(args.preds_dir)):
        mdir = os.path.join(args.preds_dir, entry)
        if not os.path.isdir(mdir) or (wanted and entry not in wanted):
            continue
        for cand in ("test.json", "preds.json", "%s.json" % entry):
            p = os.path.join(mdir, cand)
            if os.path.isfile(p):
                preds[entry] = load_preds(p)
                break
    if not preds:
        print("[panels] no predictions under %s -- run eval_all.py first" % args.preds_dir,
              file=sys.stderr)
        return 1

    images = {fn: os.path.join(args.images, fn) for fn in os.listdir(args.images)
              if os.path.splitext(fn)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")}
    keys = [k for k in gts if os.path.basename(k) in images]
    if not keys:
        print("[panels] no GT image names matched files in %s" % args.images, file=sys.stderr)
        return 1

    font_title = _font(14)
    font_box = _font(10)
    rank_ignore = {m.strip() for m in args.rank_ignore.split(",") if m.strip()}

    rows, scored, all_per = [], [], {}
    for key in keys:
        gt_boxes = gts[key]
        per = {}
        score_sum = 0
        for name, pmap in preds.items():
            rec = pmap.get(key) or pmap.get(os.path.basename(key)) or {"boxes": [], "scores": []}
            boxes = [list(map(float, b)) for b in rec.get("boxes", [])]
            scores = [float(s) for s in rec.get("scores", [])] or [1.0] * len(boxes)
            order, is_tp, missed = match_one(boxes, scores, gt_boxes, args.iou)
            ob = [boxes[i] for i in order]
            os_ = [scores[i] for i in order]
            per[name] = (ob, os_, is_tp, len(missed))
            if name not in rank_ignore:
                score_sum += sum(1 for f in is_tp if not f) + len(missed)
            rows.append({"image": key, "method": name, "n_gt": len(gt_boxes),
                         "n_pred": len(ob), "tp": sum(1 for f in is_tp if f),
                         "fp": sum(1 for f in is_tp if not f), "fn": len(missed),
                         "missed_gt": len(missed)})
        all_per[key] = per
        scored.append((key, score_sum, len(gt_boxes)))

    if args.sort == "hardest":
        scored.sort(key=lambda t: (-t[1], t[0]))
    elif args.sort == "dense":
        scored.sort(key=lambda t: (-t[2], t[0]))
    elif args.sort == "easiest":
        scored.sort(key=lambda t: (t[1], t[0]))
    else:
        scored.sort(key=lambda t: t[0])
    if args.limit:
        scored = scored[:args.limit]

    os.makedirs(args.out, exist_ok=True)
    cards = []
    for key, _, n_gt in scored:
        stem = os.path.splitext(os.path.basename(key))[0]
        try:
            img = Image.open(images[os.path.basename(key)]).convert("RGB")
        except Exception as exc:
            print("[panels] cannot read %s (%s)" % (key, exc), file=sys.stderr)
            continue
        gt_boxes = gts[key]
        per = all_per[key]
        cols = [("GT  n=%d" % n_gt, gt_boxes, None)]
        for name in preds:
            if name in per:
                ob, os_, is_tp, missed = per[name]
                tp = sum(1 for f in is_tp if f)
                cols.append(("%s  tp%d fp%d fn%d" % (name, tp, len(is_tp) - tp, missed),
                             gt_boxes, (ob, os_, is_tp)))
            else:
                cols.append(("%s  no preds" % name, gt_boxes, ([], [], [])))
        strip = panel(img, "%s   (%d GT boxes)" % (os.path.basename(key), n_gt), cols,
                      args.scale, font_title, font_box, args.max_boxes)
        path = os.path.join(args.out, stem + ".jpg")
        strip.save(path, quality=args.jpeg_quality, optimize=True)
        cards.append((stem, os.path.basename(key), n_gt))

    with open(os.path.join(args.out, "panel_stats.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "method", "n_gt", "n_pred",
                                          "tp", "fp", "fn", "missed_gt"])
        w.writeheader()
        w.writerows(rows)

    items = "\n".join(
        '<figure><img src="%s.jpg" loading="lazy"><figcaption>%s &middot; %d GT</figcaption></figure>'
        % (html.escape(s), html.escape(n), g) for s, n, g in cards)
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as f:
        f.write("""<!doctype html><meta charset="utf-8">
<title>Method comparison panels</title>
<style>
 body{background:#16181c;color:#eee;font:14px/1.5 system-ui;margin:16px}
 figure{margin:0 0 22px} img{width:100%%;max-width:1600px;display:block;border:1px solid #2c3038}
 figcaption{color:#9aa4b2;font-size:12px;margin-top:4px}
 h1{font-size:18px} .hint{color:#9aa4b2}
</style>
<h1>Test-set comparison &mdash; GT vs each method</h1>
<p class="hint">Green = GT / true positive. Orange = false positive. Order: %s.
Generated by <code>exp_analysis_demo/make_panels.py</code>.</p>
%s
""" % (html.escape(args.sort), items))

    print("[panels] wrote %d panels -> %s (index.html, panel_stats.csv)"
          % (len(cards), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
