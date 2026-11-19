"""Generate stratified few-shot subsets of the training split.

Why: the project spec explicitly allows training on a subset (75% / 50%) as
long as it is clearly reported.  We use it as a *controlled experiment* -- how
much performance do we lose when labels get scarce -- not as a compute excuse,
so the 100% run exists as well.

Rules that keep the experiment defensible (no data leakage):
  * subsets are drawn from the TRAIN split only;
  * sampling is stratified by bird density, so 50% keeps the same sparse /
    moderate / dense mix as the full training set;
  * with --by-group whole ``source_group_id`` groups (crops of one source
    frame) are kept together, mirroring the group-aware split the dataset
    itself uses -- near-duplicate crops then cannot dominate a subset;
  * the seed is fixed and written into every output file;
  * if --val/--test are given, the subset is asserted to be disjoint from both.

Usage (from the repository root)
--------------------------------
    # from the dataset manifest (recommended: uses the official density bins)
    python exp_analysis_demo/make_subsets.py \
        --manifest common/dataset/dataset_manifest.csv --by-group \
        --ratios 0.75 0.5 --out splits

    # or from YOLO/COCO/flat annotations
    python exp_analysis_demo/make_subsets.py --gt dataset/annotations/train.json ...

Output, one JSON per ratio (splits/train_75.json):
    {"source": "train", "ratio": 0.75, "seed": 0, "by_group": true,
     "n_total": 1369, "n_kept": 1028, "n_groups": 603,
     "image_ids": [...], "file_names": [...],
     "bin_stats": {"1-5": {"total": 768, "kept": 576}, ...}}
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

# Used only when the input has no official density_bin column.
BIN_EDGES = [(0, 1), (1, 3), (3, 11), (11, 51), (51, 10 ** 9)]
BIN_LABELS = ["0", "1-2", "3-10", "11-50", "51+"]


def _bin_of(count):
    for (lo, hi), label in zip(BIN_EDGES, BIN_LABELS):
        if lo <= count < hi:
            return label
    return BIN_LABELS[-1]


def load_manifest(path, split="train"):
    """-> [{name, count, group, bin}] for one split of dataset_manifest.csv."""
    out = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if split and row.get("split") != split:
                continue
            name = os.path.basename((row.get("image_path") or "").replace("\\", "/"))
            if not name:
                continue
            count = int(row.get("object_count") or 0)
            out.append({
                "name": name,
                "count": count,
                "group": row.get("source_group_id") or name,
                "bin": row.get("density_bin") or _bin_of(count),
                "split": row.get("split") or "",
            })
    return out


def load_annotations(path):
    """-> [{name, count, group=name, bin}] from COCO or flat JSON."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    recs = []
    if isinstance(d, dict) and "images" in d and "annotations" in d:
        counts = {im["id"]: 0 for im in d["images"]}
        for a in d["annotations"]:
            counts[a["image_id"]] = counts.get(a["image_id"], 0) + 1
        for im in d["images"]:
            n = counts.get(im["id"], 0)
            recs.append({"name": im["file_name"], "count": n,
                         "group": im["file_name"], "bin": _bin_of(n)})
        return recs
    if isinstance(d, dict):
        for name, v in d.items():
            if isinstance(v, dict):
                v = v.get("boxes", v.get("bboxes", []))
            recs.append({"name": name, "count": len(v),
                         "group": name, "bin": _bin_of(len(v))})
        return recs
    raise ValueError("unsupported annotation format in %s" % path)


def stratified_sample(records, ratio, seed, by_group=False):
    """Keep `ratio` of the images inside every density bin.

    With ``by_group`` each source group is first assigned to the bin held by the
    majority of its images, then whole groups are drawn inside that bin -- so a
    group is never split across bins.
    """
    rng = random.Random(seed)
    bins = sorted({r["bin"] for r in records}, key=lambda b: (len(b), b))
    kept, stats = [], {}

    if by_group:
        groups = {}
        for r in records:
            groups.setdefault(r["group"], []).append(r)
        group_bin = {}
        for g, rs in groups.items():
            group_bin[g] = Counter(r["bin"] for r in rs).most_common(1)[0][0]
        pool = {}
        for g, label in group_bin.items():
            pool.setdefault(label, []).append(g)
    else:
        pool = None

    for label in bins:
        if by_group:
            keys = sorted(pool.get(label, []))
            rng.shuffle(keys)
            target = ratio * sum(len(groups[g]) for g in keys)
            chosen, picked = [], 0
            for k in keys:                       # whole groups, until ratio reached
                if picked >= target and chosen:
                    break
                chosen.extend(groups[k])
                picked += len(groups[k])
            kept.extend(chosen)
            stats[label] = {"total": sum(len(groups[g]) for g in keys),
                            "kept": len(chosen), "groups": len(keys),
                            "groups_kept": len({r["group"] for r in chosen})}
        else:
            items = [r for r in records if r["bin"] == label]
            n = len(items)
            k = max(1, min(n, int(round(n * ratio)))) if n else 0
            kept.extend(rng.sample(items, k))
            stats[label] = {"total": n, "kept": k}

    kept.sort(key=lambda r: r["name"])
    return kept, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="dataset_manifest.csv (uses official density bins)")
    src.add_argument("--gt", help="COCO or flat annotation JSON")
    ap.add_argument("--split", default="train", help="split to subsample (manifest only)")
    ap.add_argument("--val", default=None, help="optional: assert no overlap with val")
    ap.add_argument("--test", default=None, help="optional: assert no overlap with test")
    ap.add_argument("--ratios", nargs="+", type=float, default=[0.75, 0.5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-group", action="store_true",
                    help="keep source_group_id groups together (recommended)")
    ap.add_argument("--out", default="splits")
    args = ap.parse_args(argv)

    for r in args.ratios:
        if not 0 < r <= 1:
            ap.error("ratio must be in (0, 1], got %s" % r)

    if args.manifest:
        all_recs = load_manifest(args.manifest, split="")
        records = [r for r in all_recs if r["split"] == args.split]
        # one CSV holds every split: the leakage check is automatic here
        forbidden = {r["name"] for r in all_recs if r["split"] != args.split}
        if args.val or args.test:
            print("[make_subsets] note: --val/--test ignored with --manifest "
                  "(all splits are in the same CSV and checked automatically)")
        desc = "%s[%s]" % (os.path.basename(args.manifest), args.split)
    else:
        records = load_annotations(args.gt)
        forbidden = set()
        for path in (args.val, args.test):
            if path:
                forbidden |= {r["name"] for r in load_annotations(path)}
        desc = os.path.basename(args.gt)
    if not records:
        print("[make_subsets] no records found for %s" % desc, file=sys.stderr)
        return 1

    names = [r["name"] for r in records]
    if len(set(names)) != len(names):
        print("[make_subsets] WARNING: duplicate image names in %s" % desc, file=sys.stderr)

    clash = forbidden & set(names)
    if clash:
        print("[make_subsets] ERROR: %d name(s) occur in both the sampled split and "
              "val/test, e.g. %s -- fix the dataset split first"
              % (len(clash), sorted(clash)[:3]), file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    print("[make_subsets] %s: %d images, %d groups, by_group=%s, seed=%d"
          % (desc, len(records), len({r["group"] for r in records}),
             args.by_group, args.seed))
    print("%-6s %7s %7s %8s %8s" % ("ratio", "images", "boxes", "kept%", "groups"))

    for ratio in sorted(args.ratios, reverse=True):
        kept, stats = stratified_sample(records, ratio, args.seed, by_group=args.by_group)
        assert not ({r["name"] for r in kept} & forbidden), "leakage into val/test"

        tag = "%d" % round(ratio * 100)
        out = {
            "source": args.split,
            "ratio": ratio,
            "seed": args.seed,
            "by_group": args.by_group,
            "n_total": len(records),
            "n_kept": len(kept),
            "n_groups": len({r["group"] for r in kept}),
            "n_boxes": sum(r["count"] for r in kept),
            "file_names": [r["name"] for r in kept],
            "bin_stats": stats,
        }
        path = os.path.join(args.out, "train_%s.json" % tag)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)

        print("%-6s %7d %7d %7.1f%% %8d   -> %s"
              % (tag, len(kept), out["n_boxes"], 100.0 * len(kept) / len(records),
                 out["n_groups"], path))
        print("       bins: " + "  ".join("%s %d/%d" % (k, v["kept"], v["total"])
                                          for k, v in stats.items()))

    print("[make_subsets] done. Record the ratios and seed when reporting results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
