#!/usr/bin/env python3
"""Aggregate repeated few-shot runs into mean / min / max per training ratio.

The whole point of repeating a seed is to know how big "run-to-run noise" is, so
this prints both the per-run numbers and the spread, and measures every few-shot
gap against that spread instead of against zero.

    python exp_analysis_demo/fewshot_aggregate.py            # table to stdout
    python exp_analysis_demo/fewshot_aggregate.py --write    # also rewrite the
                                                             # table inside
                                                             # results/fewshot/summary.md

Prediction directories follow the export naming in ``fewshot.py``:
``results/preds/fewshot_<ratio>`` (seed 0) and ``results/preds/fewshot_<ratio>_s<seed>``.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(REPO_ROOT, "common", "dataset", "dataset_manifest.csv")
PATTERN = re.compile(r"fewshot_(\d+)(?:_s(\d+))?$")
MARK_BEGIN = "<!-- fewshot-table:begin -->"
MARK_END = "<!-- fewshot-table:end -->"


def collect(preds_dir):
    runs = {}
    for path in sorted(glob.glob(os.path.join(preds_dir, "fewshot_*"))):
        m = PATTERN.search(os.path.basename(path))
        if not m:
            continue
        ratio, seed = int(m.group(1)), int(m.group(2) or 0)
        test_json = os.path.join(path, "test.json")
        if not os.path.isfile(test_json):
            continue
        timing = {}
        tpath = os.path.join(path, "timing.json")
        if os.path.isfile(tpath):
            timing = json.load(open(tpath, encoding="utf-8"))
        runs.setdefault(ratio, []).append(
            {"seed": seed, "preds": test_json, "timing": timing,
             "ms": timing.get("inference_ms_per_image"),
             "train_s": timing.get("train_time_s"),
             "images": timing.get("train_images")})
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds-dir", default=os.path.join(REPO_ROOT, "results", "preds"))
    ap.add_argument("--dataset", default=os.path.join(REPO_ROOT, "common", "dataset"))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--summary", default=os.path.join(REPO_ROOT, "results", "fewshot",
                                                      "summary.md"))
    args = ap.parse_args(argv)

    sys.path.insert(0, REPO_ROOT)
    from common.metrics import evaluate
    from exp_analysis_demo.eval_all import load_gt_yolo, load_preds

    manifest = MANIFEST if os.path.isfile(MANIFEST) else None
    gts = load_gt_yolo(
        os.path.join(args.dataset, "labels", "test"),
        manifest,
        os.path.join(args.dataset, "images", "test"),
    )
    runs = collect(args.preds_dir)
    if not runs:
        print("no few-shot predictions found")
        return 2

    summary = {}
    for ratio in sorted(runs):
        rows = []
        for run in runs[ratio]:
            res = evaluate(gts, load_preds(run["preds"]), 0.5)
            rows.append({"seed": run["seed"], "ap50": res["ap50"], "tp": res["tp"],
                         "fp": res["fp"], "fn": res["fn"], "ms": run["ms"],
                         "train_s": run["train_s"], "images": run["images"]})
        aps = [r["ap50"] for r in rows]
        summary[ratio] = {"rows": rows, "mean": sum(aps) / len(aps),
                          "min": min(aps), "max": max(aps), "spread": max(aps) - min(aps)}
        print("\n%d%% of the training split  (%d run%s):"
              % (ratio, len(rows), "" if len(rows) == 1 else "s"))
        for r in rows:
            print("  seed %d  mAP@0.5 %.4f  TP %4d FP %5d FN %4d  %s ms/img  %s"
                  % (r["seed"], r["ap50"], r["tp"], r["fp"], r["fn"],
                     "%.1f" % r["ms"] if r["ms"] is not None else "  n/a",
                     "%.1f min" % (r["train_s"] / 60) if r["train_s"] else "n/a"))
        print("  mean %.4f  range %.4f  spread %.4f" % (summary[ratio]["mean"],
                                                        summary[ratio]["min"],
                                                        summary[ratio]["spread"]))

    lines = [MARK_BEGIN,
             "| Train data | Images | Runs | mAP@0.5 (mean) | range | delta vs 100% | Train | Test |",
             "|---|---|---|---|---|---|---|---|"]
    base = summary.get(100, {}).get("mean")
    for ratio in sorted(summary, reverse=True):
        s = summary[ratio]
        rows = s["rows"]
        delta = "" if (base is None or ratio == 100) else "%+.2f" % (100 * (s["mean"] - base))
        train = [r["train_s"] for r in rows if r["train_s"]]
        ms = [r["ms"] for r in rows if r["ms"]]
        images = rows[0]["images"]
        lines.append("| %d%% | %s | %d | **%.4f** | %.4f-%.4f | %s | %s | %s |"
                     % (ratio, images, len(rows), s["mean"], s["min"], s["max"], delta,
                        "%.1f min" % (sum(train) / len(train) / 60) if train else "n/a",
                        "%.1f ms" % min(ms) if ms else "n/a"))  # contention only adds time
    spread = max((s["spread"] for s in summary.values()), default=0.0)
    single = max(len(s["rows"]) for s in summary.values()) < 2
    if single:
        lines += ["", ("Only one run per ratio here, so the run-to-run spread is not "
                       "estimated - no gap can be called significant from this table.")]
    else:
        lines += ["", ("Largest within-recipe spread across seeds: **%.4f**. Any gap "
                       "smaller than that is not resolvable with this many runs." % spread)]
    if not single and 100 in summary and 75 in summary:
        gap75 = summary[100]["mean"] - summary[75]["mean"]
        lines.append("75%% vs 100%% gap: %.2f points - %s the spread."
                     % (100 * gap75, "inside" if abs(gap75) <= spread else "outside"))
    if not single and 100 in summary and 50 in summary:
        gap50 = summary[100]["mean"] - summary[50]["mean"]
        lines.append("50%% vs 100%% gap: %.2f points - %s the spread."
                     % (100 * gap50, "inside" if abs(gap50) <= spread else "outside"))
    lines.append(MARK_END)
    table = "\n".join(lines)
    print("\n" + table)

    if args.write and os.path.isfile(args.summary):
        text = open(args.summary, encoding="utf-8").read()
        if MARK_BEGIN in text and MARK_END in text:
            head, _, rest = text.partition(MARK_BEGIN)
            _, _, tail = rest.partition(MARK_END)
            text = head + table + tail
        else:
            text = text.rstrip() + "\n\n## Runs and spread\n\n" + table + "\n"
        open(args.summary, "w", encoding="utf-8").write(text)
        print("\nupdated %s" % args.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
