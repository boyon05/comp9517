# Method 3: RT-DETRv2 Detection

Method 3 adapts the official RT-DETRv2 implementation for one-class bird
detection. RT-DETRv2 is an end-to-end transformer detector: a ResNet-style
backbone extracts multi-scale features, a hybrid encoder processes those
features, and a transformer decoder directly produces the final detections.

Both models in this directory are trained through **transfer learning**. The
PResNet-50 backbone is initialized from pretrained weights, while the
task-specific detection layers are trained for the bird dataset. The optimizer
uses a smaller learning rate for the pretrained backbone than for the encoder
and decoder.

## Project-specific changes from the official implementation

Most files under [`src/`](src) retain the official RT-DETRv2 training and model
implementation. The project-specific changes are concentrated in the dataset
and experiment configuration, plus the inference helper:

### [`configs/dataset/coco_detection.yml`](configs/dataset/coco_detection.yml)

This changes the official COCO setup for the project dataset:

- sets `num_classes: 1`;
- remaps the single category to `bird`;
- points the train and validation loaders to `common/dataset/images`;
- uses the project's COCO-format annotation files under
  `common/dataset/annotations`.

The images and annotations must therefore be prepared in COCO detection format,
rather than only YOLO text-label format.

### [`configs/rtdetrv2/include/rtdetrv2_r50vd.yml`](configs/rtdetrv2/include/rtdetrv2_r50vd.yml)

This is the shared model definition based on the official RT-DETRv2 R50vd
configuration. It enables pretrained PResNet-50 weights and defines the normal
three-level feature pyramid at strides 8, 16, and 32. The project-specific
training configurations override its dataset and experiment settings for
one-class bird detection.

### [`configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml`](configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml)

This is the project's default three-level experiment. Compared with a standard
COCO run, it uses the one-class dataset, project training schedule, and
fine-tuning optimizer settings. The backbone learning rate is `1e-5`, while the
main model learning rate is `1e-4`, which preserves pretrained features while
adapting the detector to birds.

### [`configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco_stride4.yml`](configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco_stride4.yml)

This is the main architectural modification. It adds the backbone's stride-4
feature map to the detector, producing a four-level pyramid. Details are given
below.

### [`references/deploy/rtdetrv2_torch.py`](references/deploy/rtdetrv2_torch.py)

This project includes a simple PyTorch inference wrapper that loads a trained
checkpoint, switches the model and postprocessor to deploy mode, resizes an
input image according to the selected configuration, and saves an annotated
image as `results_0.jpg`.

## Dataset layout

The configuration expects the following structure:

```text
common/dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── instances_test.json
```

Each JSON file must be a valid COCO detection annotation file with one bird
category. The test annotation path is supplied as a command-line override during
held-out evaluation.

## Converting the dataset for RT-DETRv2

The original bird labels are stored in YOLO text format, while RT-DETRv2's
`CocoDetection` loader expects COCO JSON annotations. The project-specific
[`tools/convert.py`](tools/convert.py) was created to bridge this difference.
It reads each image and its matching YOLO label file, converts normalized
`class centre_x centre_y width height` values into absolute COCO
`[x, y, width, height]` boxes, and writes a one-class COCO annotation file.

Run the converter once for each dataset split from the `method3` directory:

```powershell
..\.venv\Scripts\python.exe tools\convert.py `
  ..\common\dataset\images\train `
  ..\common\dataset\labels\train `
  ..\common\dataset\annotations\instances_train.json

..\.venv\Scripts\python.exe tools\convert.py `
  ..\common\dataset\images\val `
  ..\common\dataset\labels\val `
  ..\common\dataset\annotations\instances_val.json

..\.venv\Scripts\python.exe tools\convert.py `
  ..\common\dataset\images\test `
  ..\common\dataset\labels\test `
  ..\common\dataset\annotations\instances_test.json
```

The command takes three positional arguments: the image directory, the YOLO
label directory, and the output JSON path. It records image dimensions using
Pillow and creates the `bird` category with COCO `category_id: 1`. The converter
supports `.jpg` and `.jpeg` images, and should be rerun whenever the source
images or labels change.

## Stride-4 architectural change

The standard configuration uses three backbone outputs:

```text
PResNet outputs:       C3, C4, C5
Feature strides:         8, 16, 32
```

At an input resolution of 640 x 640, the finest feature map is therefore about
80 x 80. Very small birds may occupy only a few cells at this resolution.

The stride-4 configuration changes the feature flow to:

```text
PResNet outputs:       C2, C3, C4, C5
Feature strides:         4,  8, 16, 32
Feature channels:      256, 512, 1024, 2048
```

The change is implemented through the following settings:

```yaml
PResNet:
  return_idx: [0, 1, 2, 3]

HybridEncoder:
  in_channels: [256, 512, 1024, 2048]
  feat_strides: [4, 8, 16, 32]

RTDETRTransformerv2:
  feat_channels: [256, 256, 256, 256]
  feat_strides: [4, 8, 16, 32]
  num_levels: 4
  num_points: [4, 4, 4, 4]
```

The first returned PResNet feature, `C2`, is the output of the first residual
stage and has an effective stride of 4 relative to the input. The hybrid
encoder projects all four feature maps to 256 channels and fuses them. The
transformer decoder then performs multi-scale cross-attention over all four
levels, instead of only the original three levels. Each decoder attention level
uses four sampling points per query/reference location.

The practical effect is a denser high-resolution representation: with a
640 x 640 input, the stride-4 map is approximately 160 x 160. This gives the
model more spatial evidence for tiny birds and small separated objects, while
the stride-8, stride-16, and stride-32 maps preserve context and larger-object
information. The trade-off is higher GPU memory use and computation, especially
in the encoder and multi-scale decoder.

Both experiments use 640 x 640 validation inputs and an 84-epoch schedule so
that the standard and stride-4 models are trained and evaluated under the same
conditions. At 640 x 640, the default three-level model has feature maps of
approximately 80 x 80, 40 x 40, and 20 x 20 at strides 8, 16, and 32. The
stride-4 model adds the approximately 160 x 160 stride-4 map. These settings are in
[`rtdetrv2_r50vd_m_7x_coco_stride4.yml`](configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco_stride4.yml)
and [`rtdetrv2_r50vd_m_7x_coco.yml`](configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml).

## Setup

From the repository root, create the environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The commands below use the repository virtual-environment interpreter directly,
so activation is optional. Run them from the `method3` directory.

## Transfer-learning training

Train the stride-4 model from pretrained PResNet-50 weights:

```powershell
cd method3
..\.venv\Scripts\python.exe tools\train.py `
  -c configs\rtdetrv2\rtdetrv2_r50vd_m_7x_coco_stride4.yml `
  --output-dir output\train-stride4 `
  --device cuda `
  --use-amp `
  --seed 0 `
  -u train_dataloader.total_batch_size=4 val_dataloader.total_batch_size=4
```

For the default three-level model, replace the configuration path:

```powershell
..\.venv\Scripts\python.exe tools\train.py `
  -c configs\rtdetrv2\rtdetrv2_r50vd_m_7x_coco.yml `
  --output-dir output\train-default `
  --device cuda `
  --use-amp `
  --seed 0
```

The `-u` options override YAML values without editing the configuration. Use
`--device cpu` if CUDA is unavailable, but training will be substantially
slower. Checkpoints and logs are written to the selected output directory.

## Held-out test evaluation

Evaluate a completed stride-4 checkpoint on the test split:

```powershell
..\.venv\Scripts\python.exe tools\train.py `
  -c configs\rtdetrv2\rtdetrv2_r50vd_m_7x_coco_stride4.yml `
  -r output\stride4.pth `
  --test-only `
  --output-dir ..\output\method3-stride4 `
  -u val_dataloader.dataset.img_folder=..\common\dataset\images\test val_dataloader.dataset.ann_file=..\common\dataset\annotations\instances_test.json
```

The default model is evaluated in the same way by using its configuration and
checkpoint. The resulting prediction exports are stored at the repository root
under `output/method3-default` or `output/method3-stride4`.

## PyTorch image inference

Run the deployment helper on one image:

```powershell
$env:PYTHONPATH = (Get-Location).Path
..\.venv\Scripts\python.exe references\deploy\rtdetrv2_torch.py `
  -c configs\rtdetrv2\rtdetrv2_r50vd_m_7x_coco_stride4.yml `
  -r output\stride4.pth `
  --im-file ..\common\dataset\images\test\beach_bird_1661.jpg `
  --device cuda
```

The helper writes `results_0.jpg` in the current directory, with red predicted
boxes and class/score labels. It uses the input-size setting from the selected
YAML configuration.

## Relevant implementation components

Only the components directly relevant to this adaptation are listed here; the
remaining `src` files are the upstream RT-DETRv2 training infrastructure.

- [`src/nn/backbone/presnet.py`](src/nn/backbone/presnet.py): provides PResNet
  feature maps and their stride metadata. The stride-4 configuration selects
  all four stages through `return_idx: [0, 1, 2, 3]`.
- [`src/data/dataset/coco_dataset.py`](src/data/dataset/coco_dataset.py): loads
  the project's COCO-format images and annotations.
- [`src/data/dataset/coco_utils.py`](src/data/dataset/coco_utils.py): adapted
  COCO dataset helpers used for annotation handling and evaluation.
- [`tools/convert.py`](tools/convert.py): project-created conversion tool that
  transforms the dataset's YOLO labels into the COCO JSON files required by
  RT-DETRv2.
- [`src/zoo/rtdetr/`](src/zoo/rtdetr): RT-DETR model, encoder, decoder, matcher,
  criterion, and postprocessing components used by both experiment variants.
- [`tools/train.py`](tools/train.py): command-line entry point for training,
  resuming, and test-only evaluation.
- [`references/deploy/rtdetrv2_torch.py`](references/deploy/rtdetrv2_torch.py):
  single-image inference and visualisation helper.
