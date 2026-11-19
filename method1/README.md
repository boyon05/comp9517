# Method 1: Traditional Morphological Detection

Method 1 is a classical computer-vision detector. It does not train a neural
network. Instead, it highlights small bird-like regions using image
enhancement, morphology, thresholding, contour filtering, and non-maximum
suppression.

## How the method works

For each image, the detector:

1. Converts the image to grayscale.
2. Applies CLAHE to improve local contrast.
3. Applies Gaussian blur to reduce noise.
4. Computes black-hat and top-hat morphological responses using an elliptical kernel These responses emphasize dark or bright objects against the background.
5. Thresholds the responses to create binary candidate masks.
6. Applies morphological opening to remove small noise.
7. Finds external contours and converts them to bounding boxes.
8. Filters boxes by area and aspect ratio.
9. Assigns each candidate a simple response-based score.
10. Applies non-maximum suppression using an IoU threshold of `0.4`.

The implementation is intended as a training-free baseline and is most effective when birds contrast with their background.

## Dataset layout

By default, the program expects the dataset at `common/dataset/`:

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

Each image must have a corresponding label file with the same relative path.
Labels may be either:

- YOLO format: `class centre_x centre_y width height`, with normalized values;
- Pascal VOC XML format with `xmin`, `ymin`, `xmax`, and `ymax`.

The dataset loader is shared with the other methods and is implemented in
[`common/data_loader.py`](../common/data_loader.py).

## Setup

From the repository root, create and activate a virtual environment, then
install the project dependencies:

```bash
python3 -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Method 1 specifically requires Python packages `numpy` and `opencv-python`.
If the other methods are not being used, these can be installed instead:

```bash
pip install numpy opencv-python
```

## Running inference and evaluation

Run on the validation set:

```bash
python method1/method1.py \
  --dataset common/dataset \
  --split val \
  --output results/method1 \
  --visualize 12
```

Run on the test set:

```bash
python method1/method1.py \
  --dataset common/dataset \
  --split test \
  --output results/method1 \
  --visualize 12
```

The default `--split all` evaluates both `val` and `test`.

Useful options:

| Option | Default | Purpose |
|---|---:|---|
| `--dataset` | `common/dataset` | Dataset root directory |
| `--split` | `all` | `train`, `val`, `test`, or both `val` and `test` |
| `--output` | `results` | Directory for generated outputs |
| `--threshold` | `21` | Binary response threshold |
| `--min-area` | `6` | Minimum candidate bounding-box area |
| `--visualize` | `12` | Number of images to save with annotations |

For example, a stricter threshold and larger minimum area can be tested with:

```bash
python method1/method1.py \
  --dataset common/dataset \
  --split test \
  --threshold 25 \
  --min-area 10
```

## Outputs

For each evaluated split, the program creates a directory containing:

- `metrics.json`: split-level metrics and timing information;
- `per_image.csv`: ground-truth count, prediction count, and inference time for
  each image;
- `example_*.jpg`: optional visualizations, with ground-truth boxes in green
  and predicted boxes in red.

The reported `mAP50` is single-class AP at IoU `0.5`. Since this is a
training-free method, the reported training time is always zero.
