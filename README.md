# Bird Detection — COMP9517

Computer Vision project comparing three bird-detection
approaches on beach images:

1. **Method 1:** training-free classical morphological image processing;
2. **Method 2:** YOLOv8 single-stage detector with augmentation experiments;
3. **Method 3:** RT-DETRv2 end-to-end transformer detector, including a
   stride-4 feature-pyramid variant.

The repository also contains unified evaluation, robustness, few-shot,
visualization, timing, and packaging utilities.

## Repository structure

```text
.
├── common/
│   ├── data_loader.py       # Shared YOLO/XML dataset loader
│   └── metrics.py           # Shared IoU, matching, AP, timing, and summaries
├── method1/
│   ├── method1.py           # Classical detector
│   └── README.md
├── method2/
│   ├── configs/             # YOLO dataset and augmentation configurations
│   ├── src/                 # Training, evaluation, demo, and visualization
│   └── README.md
├── method3/
│   ├── configs/             # RT-DETRv2 and stride-4 configurations
│   ├── src/                 # Vendored RT-DETRv2 implementation
│   ├── tools/convert.py     # YOLO-to-COCO annotation conversion
│   ├── tools/train.py       # RT-DETRv2 training entry point
│   └── README.md
├── exp_analysis_demo/       # Shared exporters, evaluation, demos, and analysis
├── exp_robustness/          # Independent distortion robustness experiment
├── output/                  # Main method evaluation summaries
├── results/                 # Unified evaluation and experiment outputs
├── requirements.txt
└── README.md
```

## Setup

Create an environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The requirements include NumPy, OpenCV, Pillow, PyTorch, torchvision,
Ultralytics, PyYAML, pandas, Matplotlib, and RT-DETR evaluation dependencies.
The requirements file currently specifies CUDA-enabled PyTorch builds; use a
compatible environment or adjust the PyTorch installation for CPU/macOS use.

## Dataset

Place the dataset in this layout:

```text
common/dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

The YOLO label format is:

```text
class centre_x centre_y width height
```

Coordinates are normalized to `[0, 1]`, and the single class is `bird`.

Method 3 uses COCO annotations. Convert the YOLO labels after installing the
dataset:

```powershell
python method3/tools/convert.py `
  common/dataset/images/train `
  common/dataset/labels/train `
  common/dataset/annotations/instances_train.json

python method3/tools/convert.py `
  common/dataset/images/val `
  common/dataset/labels/val `
  common/dataset/annotations/instances_val.json

python method3/tools/convert.py `
  common/dataset/images/test `
  common/dataset/labels/test `
  common/dataset/annotations/instances_test.json
```

## Method 1 — classical detector

[`method1/method1.py`](method1/method1.py) detects candidate birds without
training. It uses grayscale conversion, CLAHE, Gaussian smoothing, black-hat
and top-hat morphology, thresholding, contour filtering, scoring, and NMS.

Run validation or test evaluation:

```bash
python method1/method1.py \
  --dataset common/dataset \
  --split test \
  --output output/method1 \
  --visualize 12
```

Outputs include `metrics.json`, `per_image.csv`, and annotated example images.
See [`method1/README.md`](method1/README.md) for algorithm details and tuning
options.

## Method 2 — YOLOv8

Method 2 uses Ultralytics YOLOv8 and learns bird bounding boxes from the
training split. Training and augmentation utilities are in
[`method2/src/`](method2/src/).

From the `method2` directory:

```bash
cd method2
python src/train_single_stage.py \
  --model yolov8s.pt \
  --data configs/data.yaml \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device 0
```

Run the augmentation ablation study:

```bash
python src/run_augmentation_ablation.py \
  --epochs 30 --batch 16 --device 0
```

Evaluate trained weights and run an inference demo:

```bash
python src/evaluate_metrics.py \
  --model results/train_runs/single_stage_baseline/weights/best.pt \
  --data configs/data.yaml --split test

python src/demo_inference.py \
  --model results/train_runs/single_stage_baseline/weights/best.pt \
  --source path/to/image_or_folder
```

See [`method2/README.md`](method2/README.md) for configuration and
visualization details.

## Method 3 — RT-DETRv2

Method 3 adapts the official RT-DETRv2 implementation for one-class bird
detection. The PResNet backbone uses pretrained weights, so the detector is
trained through transfer learning. The project includes a default model and a
stride-4 model that adds a higher-resolution feature map for small birds.

Train from the `method3` directory:

```powershell
cd method3
..\.venv\Scripts\python.exe tools\train.py `
  -c configs\rtdetrv2\rtdetrv2_r50vd_m_7x_coco.yml `
  --output-dir output\train-default `
  --device cuda --use-amp --seed 0
```

Use the stride-4 configuration by replacing the configuration path with:

```text
configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco_stride4.yml
```

Run single-image inference with the deployment helper:

```powershell
$env:PYTHONPATH = (Get-Location).Path
..\.venv\Scripts\python.exe references\deploy\rtdetrv2_torch.py `
  -c configs\rtdetrv2\rtdetrv2_r50vd_m_7x_coco.yml `
  -r output\default_params.pth `
  --im-file ..\common\dataset\images\test\beach_bird_1661.jpg `
  --device cuda
```

See [`method3/README.md`](method3/README.md) for the architectural changes,
transfer-learning settings, COCO conversion, and licensing attribution.

## Unified evaluation

The scripts in [`exp_analysis_demo/`](exp_analysis_demo/) export predictions
from each method into a common JSON format and evaluate them with
[`common/metrics.py`](common/metrics.py).

Export predictions:

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

Evaluate all available prediction folders:

```bash
python exp_analysis_demo/eval_all.py \
  --gt-yolo common/dataset/labels/test \
  --images-dir common/dataset/images/test \
  --preds-dir results/preds \
  --out results/eval
```

The evaluator reports pooled single-class mAP@0.5, precision, recall, F1,
TP/FP/FN counts, and timing information. It writes `summary.md`, `summary.csv`,
and per-image CSV files.

## Additional experiments

### Robustness

[`exp_robustness/`](exp_robustness/) tests Method 2 and Method 3 under noise,
blur, brightness changes, occlusion, and denoising:

```bash
python exp_robustness/generate_distortions.py
python exp_robustness/improve_robustness.py
python exp_robustness/evaluate_method2_unified.py
python exp_robustness/evaluate_method3.py
python exp_robustness/compare_models.py
```

See [`exp_robustness/README.md`](exp_robustness/README.md) for the protocol and
recorded results.

### Few-shot training

Create stratified subsets and run reduced-data YOLO experiments:

```bash
python exp_analysis_demo/make_subsets.py \
  --manifest common/dataset/dataset_manifest.csv \
  --by-group --ratios 0.75 0.5 --out exp_analysis_demo/splits

python exp_analysis_demo/fewshot.py --ratios 100,75,50
```

### Visualization

Generate static comparison panels:

```bash
python exp_analysis_demo/make_panels.py \
  --images common/dataset/images/test \
  --gt-yolo common/dataset/labels/test \
  --preds-dir results/preds --out results/panels
```

Run the local interactive web demo:

```bash
python exp_analysis_demo/webapp.py --open
```

## Current unified test results

The saved unified evaluation uses 240 test images and 1,518 annotated bird
boxes. The main results are stored in [`results/eval/summary.md`](results/eval/summary.md).

| Method | mAP@0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Method 1 | 0.0002 | 0.004 | 0.007 | 0.005 |
| YOLOv8 | 0.8778 | 0.876 | 0.805 | 0.839 |
| RT-DETRv2 default | 0.9314 | 0.897 | 0.901 | 0.899 |
| RT-DETRv2 stride-4 | 0.9264 | 0.905 | 0.883 | 0.894 |

These values use the shared evaluator rather than method-specific validation
implementations, making the methods directly comparable.

## Outputs

- [`output/`](output/): method-specific summaries and per-image results.
- [`results/eval/`](results/eval/): unified metrics and per-image analysis.
- [`results/preds/`](results/preds/): exported predictions and timing files.
- [`results/robustness/`](results/robustness/): distortion experiment results.
- [`results/sweep/`](results/sweep/): YOLOv8 post-processing sweeps.
- [`results/fewshot/`](results/fewshot/): reduced-data experiment summaries.
- [`results/panels/`](results/panels/): static visual comparison outputs.

Datasets, checkpoints, and generated image-heavy artifacts are excluded from
the submission package.

## External resources

- Dataset: [Beach Bird Detection Dataset v1.0](https://www.kaggle.com/datasets/liugalhigh/beach-bird)
- RT-DETR repository: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)
- RT-DETRv2 paper: [arXiv:2407.17140](https://arxiv.org/abs/2407.17140)
