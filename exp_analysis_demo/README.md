# Analysis and Demonstration Toolkit

This folder contains the shared export, evaluation, visualization, robustness,
few-shot, timing, and packaging utilities for the three bird-detection
methods.

## Repository paths

The scripts use the current repository layout:

```text
common/dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/

method1/method1.py
method2/configs/
method3/configs/ and method3/src/
results/
```

The dataset images and trained checkpoints are not committed. Install them
before running scripts that perform inference or training. A
`common/dataset/dataset_manifest.csv` file is optional for ordinary evaluation
but required by density-based analysis and recommended for exact image sizes.

Install the declared dependencies from the repository root before using the
toolkit:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, use `.venv\\Scripts\\activate` instead. The active Python
interpreter must be the same one selected by the IDE; otherwise imports such as
NumPy, Pillow, PyTorch, and Ultralytics may appear unresolved even when they
are listed in `requirements.txt`.

Model checkpoints are expected at these default locations when applicable:

```text
method2/results/train_runs/single_stage_baseline/weights/best.pt
method3/output/default_params.pth
method3/output/stride4.pth
```

Override checkpoint paths with the command-line options when your files are
stored elsewhere.

## Shared evaluation workflow

Run from the repository root. First export predictions in the common format:

```bash
python exp_analysis_demo/export_method1.py --split test

python exp_analysis_demo/export_method2.py \
  --weights method2/results/train_runs/single_stage_baseline/weights/best.pt \
  --images common/dataset/images/test

python exp_analysis_demo/export_method3.py \
  --weights method3/output/default_params.pth \
  --images common/dataset/images/test \
  --config method3/configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml
```

Then evaluate all exported predictions with the shared implementation in
[`common/metrics.py`](../common/metrics.py):

```bash
python exp_analysis_demo/eval_all.py \
  --gt-yolo common/dataset/labels/test \
  --images-dir common/dataset/images/test \
  --preds-dir results/preds \
  --out results/eval
```

The evaluator writes `summary.md`, `summary.csv`, and
`per_image_<method>.csv`. It computes single-class AP at IoU 0.5 using pooled,
confidence-ordered greedy matching. Predictions use absolute pixel-coordinate
`[x1, y1, x2, y2]` boxes and scores.

## File descriptions

### Prediction exporters

- [`export_method1.py`](export_method1.py): runs the training-free detector in
  [`method1/method1.py`](../method1/method1.py), writing Method 1 predictions
  and CPU timing.
- [`export_method2.py`](export_method2.py): runs YOLOv8, keeps detections with
  score at least `0.01`, and writes predictions plus median inference timing.
- [`export_method3.py`](export_method3.py): loads the RT-DETRv2 checkpoint from
  [`method3/`](../method3/), switches it to deploy mode, and exports predictions
  and timing. It uses the selected configuration's evaluation size.

### Evaluation and comparisons

- [`eval_all.py`](eval_all.py): the main evaluator and prediction/timing loader.
  It accepts YOLO labels, COCO JSON, or flat ground-truth JSON.
- [`bootstrap_ci.py`](bootstrap_ci.py): computes paired whole-image bootstrap
  confidence intervals for mAP@0.5 using exported predictions.
- [`density_compare.py`](density_compare.py): compares recall and false
  positives across density bins from `dataset_manifest.csv`. It requires that
  manifest and compatible per-image CSV data.

### Visualization

- [`demo.py`](demo.py): creates interactive or video four-panel comparisons of
  ground truth, Method 1, YOLOv8, and RT-DETRv2.
- [`webapp.py`](webapp.py): serves a local browser demo with image search,
  confidence control, keyboard navigation, and drag-and-drop inference. It
  reuses the model and drawing functions from `demo.py`.
- [`make_panels.py`](make_panels.py): creates static per-image comparison
  panels, an HTML contact sheet, and `panel_stats.csv`.

### Robustness and post-processing

- [`sweep_postprocess.py`](sweep_postprocess.py): sweeps YOLOv8 NMS IoU and
  confidence settings on validation data, then optionally applies the selected
  setting once to test data.
- [`postprocess_variants.py`](postprocess_variants.py): evaluates NMS,
  confidence, weighted box fusion, centre gating, and optional larger input
  sizes using cached validation predictions.

### Few-shot experiments

- [`make_subsets.py`](make_subsets.py): creates stratified 75% and 50% training
  subset JSON files, optionally keeping source groups together.
- [`fewshot.py`](fewshot.py): trains YOLOv8 on 100%, 75%, or 50% of the
  training split, exports test predictions, and writes a summary.
- [`fewshot_aggregate.py`](fewshot_aggregate.py): aggregates repeated few-shot
  seeds and reports mean, range, and spread.

### Timing and utilities

- [`timing_m3.py`](timing_m3.py): measures RT-DETRv2 default and stride-4
  latency with FP32 and autocast variants.
- [`timing_min.py`](timing_min.py): re-measures few-shot YOLO latency using the
  minimum median of repeated passes.
- [`re_time_main.py`](re_time_main.py): re-measures the main YOLOv8 and
  RT-DETRv2 timing files with CUDA synchronization.

## Optional inputs and dependencies

All Python dependencies used by these scripts are declared in the repository
[`requirements.txt`](../requirements.txt), including NumPy, Pillow, OpenCV,
PyTorch, torchvision, Ultralytics, PyYAML, and Matplotlib where needed.

The scripts that only read predictions require the repository root on the
Python path; they already add it when run as shown above. RT-DETR scripts add
`method3/` so that `src.core` and the vendored model modules are importable.

The following scripts require additional prior artifacts:

| Script | Required artifact |
|---|---|
| `eval_all.py` | Exported `results/preds/<method>/test.json` files |
| `demo.py`, `webapp.py` | Dataset images, Method 2 weights, Method 3 weights/config |
| `density_compare.py` | `dataset_manifest.csv` and per-image comparison CSVs |
| `sweep_postprocess.py` | YOLOv8 weights and validation/test images and labels |
| `postprocess_variants.py` | `results/sweep/raw_val.json` |
| `fewshot_aggregate.py` | `results/preds/fewshot_*/test.json` |
| `timing_min.py` | Few-shot checkpoints and corresponding `timing.json` files |

If the dataset or checkpoint is absent, these scripts now fail with the normal
file-not-found message rather than silently using an unrelated path.
