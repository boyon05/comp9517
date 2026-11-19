"""Unified detection metrics for comparing across methods.

Notes
-----
* mAP@0.5 = AP@0.5 of the single class, computed by *pooling* all test images
  into one ranked list of detections. This is the same convention Ultralytics,
  Detectron2 and COCO use for a single-class problem, and it handles the
  "negative scenes" in this dataset naturally: predictions on an image with no
  annotated bird simply become false positives.
* Matching is COCO-style greedy: detections are processed by descending score
  and each one is assigned to the still-unmatched ground-truth box with the
  highest IoU; the detection is a true positive only if that IoU >= threshold.
* Only numpy is used.

Usage
-----
    from common.metrics import evaluate, summary_table

    results = [evaluate(gts, preds_dict, name="method2_fasterrcnn"),
               evaluate(gts, preds_dict_b, name="method3_detr")]
    print(summary_table(results))
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "box_iou",
    "match_image",
    "average_precision",
    "evaluate",
    "summary_table",
    "Timer",
    "measure_latency",
    "measure_train_time",
]

# --------------------------------------------------------------------------- #
# Boxes
# --------------------------------------------------------------------------- #
def _as_boxes(boxes) -> np.ndarray:
    """Convert anything box-like into a float32 (N, 4) array of x1, y1, x2, y2."""
    arr = np.asarray(boxes, dtype=np.float32) if len(boxes) else np.zeros((0, 4), np.float32)
    if arr.ndim == 1 and arr.size == 4:
        arr = arr.reshape(1, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"boxes must be (N, 4) in xyxy order, got shape {arr.shape}")
    return arr

def box_iou(box: Sequence[float], boxes) -> np.ndarray:
    """IoU of one box against an (N, 4) array of xyxy boxes."""
    box = np.asarray(box, dtype=np.float32)
    boxes = _as_boxes(boxes)
    if boxes.shape[0] == 0:
        return np.zeros((0,), np.float32)

    x1 = np.maximum(boxes[:, 0], box[0])
    y1 = np.maximum(boxes[:, 1], box[1])
    x2 = np.minimum(boxes[:, 2], box[2])
    y2 = np.minimum(boxes[:, 3], box[3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)
    area_b = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0).astype(np.float32)

# --------------------------------------------------------------------------- #
# Per-image matching
# --------------------------------------------------------------------------- #
def match_image(pred_boxes, pred_scores, gt_boxes, iou_thr: float = 0.5):
    """Greedy COCO-style matching for one image.

    Returns
    -------
    scores : (N,) float   -- one entry per prediction
    labels : (N,) int     -- 1 for true positive, 0 for false positive
    n_gt   : int          -- number of ground-truth boxes
    n_fn   : int          -- ground-truth boxes that were never matched
    """
    pred_boxes = _as_boxes(pred_boxes)
    gt_boxes = _as_boxes(gt_boxes)
    scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)

    if pred_boxes.shape[0] != scores.shape[0]:
        raise ValueError(
            f"{pred_boxes.shape[0]} boxes but {scores.shape[0]} scores for one image"
        )

    n_gt = int(gt_boxes.shape[0])
    if pred_boxes.shape[0] == 0:
        return scores, np.zeros((0,), np.int64), n_gt, n_gt

    order = np.argsort(-scores, kind="stable")
    matched = np.zeros(n_gt, dtype=bool)
    labels = np.zeros(pred_boxes.shape[0], dtype=np.int64)

    for idx in order:
        ious = box_iou(pred_boxes[idx], gt_boxes)
        if ious.size:
            ious = np.where(matched, -1.0, ious)  # cannot reuse a GT box
            best = int(np.argmax(ious))
            if ious[best] >= iou_thr:
                matched[best] = True
                labels[idx] = 1

    return scores[order], labels[order], n_gt, n_gt - int(matched.sum())

# --------------------------------------------------------------------------- #
# Average precision
# --------------------------------------------------------------------------- #
def average_precision(scores, labels, n_gt: int) -> float:
    """All-point interpolated AP (VOC2010 / COCO style)."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if n_gt <= 0 or scores.size == 0:
        return 0.0

    order = np.argsort(-scores, kind="stable")
    labels = labels[order]

    tp_cum = np.cumsum(labels)
    fp_cum = np.cumsum(1.0 - labels)
    recall = tp_cum / float(n_gt)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def best_f1(scores, labels, n_gt: int) -> Dict[str, float]:
    """Best achievable F1 over all score thresholds (for the discussion section)."""
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": float("nan")}

    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(n_gt, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    k = int(np.argmax(f1))
    return {
        "precision": float(precision[k]),
        "recall": float(recall[k]),
        "f1": float(f1[k]),
        "threshold": float(scores[order][k]),
    }

# --------------------------------------------------------------------------- #
# Dataset level
# --------------------------------------------------------------------------- #
def evaluate(
    gts: Dict[str, Iterable],
    preds: Dict[str, Dict[str, Iterable]],
    iou_thr: float = 0.5,
    name: str = "method",
    score_thr: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluate one method on one split.

    Parameters
    ----------
    gts : {image_id: [[x1, y1, x2, y2], ...]}   ground truth for every test image
    preds : {image_id: {"boxes": [...], "scores": [...]}}  predictions; images that
            are missing are treated as "no detection" (all GT of that image missed)
    iou_thr : IoU threshold for a match (0.5 = the required mAP@0.5)
    score_thr : if given, also report precision/recall/F1 after dropping
                predictions with score < score_thr

    Returns
    -------
    dict with ap50 (== mAP@0.5 for a single class), counts and F1 stats.
    """
    all_scores: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    n_gt_total = 0
    n_pred_total = 0
    per_image = {}

    for image_id, gt_boxes in gts.items():
        entry = preds.get(image_id) or {}
        p_boxes = entry.get("boxes", [])
        p_scores = entry.get("scores", [])

        scores, labels, n_gt, n_fn = match_image(p_boxes, p_scores, gt_boxes, iou_thr)
        all_scores.append(scores)
        all_labels.append(labels)
        n_gt_total += n_gt
        n_pred_total += int(scores.size)
        per_image[image_id] = {
            "n_gt": n_gt,
            "n_pred": int(scores.size),
            "n_tp": int(labels.sum()),
            "n_fn": n_fn,
        }

    scores_cat = np.concatenate(all_scores) if all_scores else np.zeros((0,), np.float32)
    labels_cat = np.concatenate(all_labels) if all_labels else np.zeros((0,), np.float32)

    ap = average_precision(scores_cat, labels_cat, n_gt_total)
    tp = int(labels_cat.sum())
    fp = int(labels_cat.size - tp)

    out = {
        "method": name,
        "ap50": ap,
        "n_gt": int(n_gt_total),
        "n_pred": int(n_pred_total),
        "tp": tp,
        "fp": fp,
        "fn": int(max(n_gt_total - tp, 0)),
        "iou_thr": float(iou_thr),
    }
    out.update({f"best_{k}": v for k, v in best_f1(scores_cat, labels_cat, n_gt_total).items()})

    if score_thr is not None:
        keep = scores_cat >= score_thr
        tp_k = float(labels_cat[keep].sum())
        pred_k = float(keep.sum())
        prec = tp_k / max(pred_k, 1.0)
        rec = tp_k / max(n_gt_total, 1)
        out.update(
            {
                "conf_thr": float(score_thr),
                "precision_at_thr": prec,
                "recall_at_thr": rec,
                "f1_at_thr": 2 * prec * rec / max(prec + rec, 1e-12),
            }
        )

    out["_per_image"] = per_image
    return out

def summary_table(results: Sequence[Dict], timing: Optional[Dict[str, Dict]] = None) -> str:
    """Markdown table for the report / terminal, with optional timing columns."""
    def f(x, nd=3):
        return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"

    head = ["Method", "mAP@0.5", "TP", "FP", "FN", "Precision", "Recall", "F1"]
    timing = timing or {}
    has_time = any(timing.get(r["method"]) for r in results)
    if has_time:
        head += ["Train (min)", "Infer (ms/img)"]

    rows = []
    for r in results:
        t = timing.get(r["method"], {})
        row = [
            r["method"],
            f(r["ap50"]),
            str(r["tp"]),
            str(r["fp"]),
            str(r["fn"]),
            f(r.get("best_precision")),
            f(r.get("best_recall")),
            f(r.get("best_f1")),
        ]
        if has_time:
            train = t.get("train_time_s")
            infer = t.get("inference_ms_per_image")
            row += [
                f(train / 60.0, 1) if train is not None else "n/a",
                f(infer, 1) if infer is not None else "n/a",
            ]
        rows.append(row)

    widths = [max(len(str(h)), *(len(str(row[i])) for row in rows)) for i, h in enumerate(head)]
    line = lambda cells: "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(head), sep] + [line(r) for r in rows])


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #
class Timer:
    """Context manager: ``with Timer() as t: ...`` then ``t.elapsed``."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._t0: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self._t0
        return None


def measure_latency(
    fn: Callable[[], None],
    n_runs: int = 10,
    warmup: int = 2,
    sync: Optional[Callable[[], None]] = None,
    n_images: int = 1,
) -> Dict[str, float]:
    """Mean/std inference latency of ``fn`` in milliseconds per image."""
    for _ in range(max(warmup, 0)):
        fn()
    if sync:
        sync()

    times: List[float] = []
    for _ in range(max(n_runs, 1)):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()  # CUDA is async: without this the timing is meaningless
        times.append(time.perf_counter() - t0)

    arr = np.asarray(times, dtype=np.float64) * 1000.0
    per_image = arr / max(n_images, 1)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "inference_ms_per_image": float(per_image.mean()),
        "fps": float(1000.0 / max(per_image.mean(), 1e-9)),
        "n_runs": len(times),
        "n_images": int(n_images),
    }

def measure_train_time(train_fn: Callable[[], object]) -> Dict[str, float]:
    """Wall-clock training time in seconds (call it around the fit/train call)."""
    with Timer() as t:
        train_fn()
    return {"train_time_s": float(t.elapsed)}
