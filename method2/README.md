# Method 2: Single-Stage Detection (YOLOv8)

Method 2 uses Ultralytics YOLOv8, a single-stage object detector. The model
directly predicts bounding boxes, confidence scores, and the `bird` class in a
single network pass.

## Dataset and setup

The default dataset configuration expects the following layout relative to the
repository root:

```text
common/dataset/
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Labels use YOLO format: `class centre_x centre_y width height`, with normalized
coordinates. The dataset configuration is in
[`configs/data.yaml`](configs/data.yaml), and defines one class: `bird`.

From the repository root, create an environment and install dependencies:

```bash
python3 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Alternatively, the core Method 2 packages are `ultralytics`, `opencv-python`,
`numpy`, `PyYAML`, `pandas`, and `matplotlib`.

## Training

Run a standard YOLOv8 training experiment from the `method2` directory:

```bash
cd method2
python src/train_single_stage.py \
  --model yolov8s.pt \
  --data configs/data.yaml \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device 0 \
  --exp_name single_stage_baseline
```

Use `--device cpu` when CUDA is unavailable. The pretrained model is downloaded
by Ultralytics if it is not already available.

To train with an augmentation configuration, add `--aug_config`:

```bash
python src/train_single_stage.py \
  --aug_config configs/aug_advanced.yaml \
  --epochs 50 \
  --batch 16
```

Training outputs are saved under `results/train_runs/<experiment-name>/`,
including the best weights at `weights/best.pt`.

## Augmentation ablation study

The ablation script trains and evaluates three experiments: baseline, basic,
and advanced augmentation.

```bash
python src/run_augmentation_ablation.py \
  --model yolov8s.pt \
  --data configs/data.yaml \
  --epochs 30 \
  --batch 16 \
  --device 0
```

Results are written to `results/ablation_study/`, including a CSV summary,
Markdown summary, comparison chart, and one training run per strategy.

The augmentation configurations are:

- [`aug_baseline.yaml`](configs/aug_baseline.yaml): no augmentation;
- [`aug_basic.yaml`](configs/aug_basic.yaml): HSV changes, horizontal flips,
  translation, scale, and erasing;
- [`aug_advanced.yaml`](configs/aug_advanced.yaml): basic transforms plus
  Mosaic, MixUp, Copy-Paste, and stronger scale/erasing settings.

## Evaluation

Evaluate trained weights on the independent test set:

```bash
python src/evaluate_metrics.py \
  --model results/train_runs/single_stage_baseline/weights/best.pt \
  --data configs/data.yaml \
  --split test \
  --conf 0.25
```

The evaluation reports mAP@0.5, mAP@0.5:0.95, precision, recall, and inference
latency. It also saves a metrics summary and up to ten successful and failed
detection images under `results/evaluation/`.

## Inference demo

Run detection on an image or directory:

```bash
python src/demo_inference.py \
  --model results/train_runs/single_stage_baseline/weights/best.pt \
  --source path/to/image_or_folder \
  --conf 0.25 \
  --imgsz 640
```

Annotated predictions are saved under `results/demo_output/predictions/` by
default. If no valid source is supplied, the script uses the generated
augmentation demo image when available.

## Visualisation utilities

### Augmentation visualisation

[`src/visualize_augmentations.py`](src/visualize_augmentations.py) creates a
three-panel figure showing the original image, basic augmentation, and a
simulated four-image Mosaic:

```bash
python src/visualize_augmentations.py \
  --image path/to/sample.jpg \
  --output results/augmentation_demo.png
```

If `--image` is omitted or does not exist, a synthetic beach scene is used.
The helper functions `apply_basic_aug()` and `apply_simulated_mosaic()` perform
the transformations used in the demonstration.

### Detection-case visualisation

[`src/evaluate_metrics.py`](src/evaluate_metrics.py) contains utilities for
examining predictions:

- `load_gt_boxes()` converts normalized YOLO labels to pixel coordinates;
- `compute_iou_matrix()` computes pairwise box IoU;
- `analyze_and_draw_cases()` matches predictions to ground truth and draws
  true positives in green, false positives in red, and missed ground-truth
  boxes in orange.

The evaluation command automatically uses these functions to save successful
and failure-case examples.

## Source files

| File | Purpose |
|---|---|
| [`src/train_single_stage.py`](src/train_single_stage.py) | Trains one YOLOv8 model using optional augmentation settings. |
| [`src/run_augmentation_ablation.py`](src/run_augmentation_ablation.py) | Runs the three augmentation experiments and compares accuracy and speed. |
| [`src/evaluate_metrics.py`](src/evaluate_metrics.py) | Evaluates weights and extracts annotated success/failure cases. |
| [`src/demo_inference.py`](src/demo_inference.py) | Runs and saves YOLO predictions for images, folders, or videos. |
| [`src/visualize_augmentations.py`](src/visualize_augmentations.py) | Generates illustrative augmentation figures. |
| [`src/setup_dataset.py`](src/setup_dataset.py) | Checks dataset installation and can create tiny mock images for pipeline tests. |
| [`src/utils.py`](src/utils.py) | Resolves dataset paths and creates a runtime YAML configuration. |
| [`configs/data.yaml`](configs/data.yaml) | Defines dataset paths and the bird class. |
| [`configs/aug_*.yaml`](configs) | Defines baseline, basic, and advanced augmentation parameters. |
