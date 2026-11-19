# Robustness Experiment

This experiment measures how Method 2 (YOLOv8) and Method 3 (RT-DETRv2)
perform when test images are affected by common visual distortions. It also
tests a simple denoising step intended to recover performance after Gaussian
noise is added.

The experiment uses the held-out test split and evaluates every condition with
the shared metric implementation in [`common/metrics.py`](../common/metrics.py).
The main metric is single-class mAP@0.5, with precision and recall also
recorded.

## Experiment protocol

The test images are evaluated under these conditions:

| Condition | Transformation |
|---|---|
| Original | Unmodified test image |
| Noise | Additive Gaussian noise with standard deviation 25 |
| Blur | Gaussian blur using a 9×9 kernel and sigma 2 |
| Low brightness | Multiply image intensity by 0.5 |
| Occlusion | Cover the central 20% of the image with a black rectangle |
| Denoised | Apply OpenCV fast non-local means denoising to the noisy images |

`generate_distortions.py` uses a fixed NumPy random seed (`42`) so the Gaussian
noise is reproducible. The distortions do not change the ground-truth boxes;
the original annotations are reused for every condition.

## Files

### [`generate_distortions.py`](generate_distortions.py)

Reads the test images and creates four distorted copies under
`distorted_images/`:

- `noise/`
- `blur/`
- `low_brightness/`
- `occlusion/`

The image filenames are preserved so that predictions can be matched with the
original annotations.

### [`improve_robustness.py`](improve_robustness.py)

Applies `cv2.fastNlMeansDenoisingColored()` to the generated noisy images and
saves the results under `distorted_images/denoised/`. This is a post-processing
experiment, not additional model training.

### [`evaluate_method2_unified.py`](evaluate_method2_unified.py)

Evaluates the trained YOLOv8 model on all six conditions. It runs batched
prediction with a confidence threshold of `0.01`, NMS IoU threshold `0.7`, and
640-pixel inference size, then evaluates the predictions at IoU `0.5` using the
shared metrics code. Results are saved to
`results/method2_unified_metrics.csv`.

### [`evaluate_method3.py`](evaluate_method3.py)

Evaluates both RT-DETRv2 checkpoints:

- `default`
- `stride4`

Each checkpoint is evaluated on all six conditions. Results are saved to
`results/method3_robustness_metrics.csv`.

### [`compare_models.py`](compare_models.py)

Combines the Method 2 and Method 3 CSV files, writes
`results/all_models_robustness.csv`, and generates the line chart
`results/method2_method3_comparison.png`.

### [`visualize_predictions.py`](visualize_predictions.py)

Creates a qualitative three-panel comparison for one test image:

1. original image;
2. Gaussian-noise image;
3. denoised image.

YOLOv8 predictions are drawn on each panel and the number of detections is
shown in the title. The output is saved as
`results/prediction_comparison.png`.

## Running the experiment

Run these commands from the repository root:

```bash
python exp_robustness/generate_distortions.py
python exp_robustness/improve_robustness.py
python exp_robustness/evaluate_method2_unified.py
python exp_robustness/evaluate_method3.py
python exp_robustness/compare_models.py
python exp_robustness/visualize_predictions.py
```

The scripts expect the archived dataset and trained checkpoints at the paths
defined near the top of each script. If the dataset or checkpoint locations
differ, update those `Path` constants before running the evaluation scripts.
The evaluation scripts also import helpers from `exp_analysis_demo`, so that
folder must be available in the repository environment.

## Recorded results

The following results are read from the saved CSV files in `results/`. Values
are mAP@0.5; precision and recall are included for completeness.

| Model | Condition | mAP@0.5 | Precision | Recall |
|---|---|---:|---:|---:|
| YOLOv8 | Original | 0.8779 | 0.8760 | 0.8050 |
| YOLOv8 | Noise | 0.7200 | 0.8336 | 0.6370 |
| YOLOv8 | Blur | 0.7145 | 0.7821 | 0.6383 |
| YOLOv8 | Low brightness | 0.8746 | 0.8664 | 0.8162 |
| YOLOv8 | Occlusion | 0.7497 | 0.8453 | 0.6983 |
| YOLOv8 | Denoised | 0.7470 | 0.8826 | 0.6489 |
| RT-DETRv2 default | Original | 0.9307 | 0.8958 | 0.9005 |
| RT-DETRv2 default | Noise | 0.8587 | 0.8984 | 0.7866 |
| RT-DETRv2 default | Blur | 0.8368 | 0.8794 | 0.7589 |
| RT-DETRv2 default | Low brightness | 0.9202 | 0.9044 | 0.8722 |
| RT-DETRv2 default | Occlusion | 0.8204 | 0.8831 | 0.7866 |
| RT-DETRv2 default | Denoised | 0.8044 | 0.8546 | 0.7398 |
| RT-DETRv2 stride-4 | Original | 0.9271 | 0.9037 | 0.8841 |
| RT-DETRv2 stride-4 | Noise | 0.8514 | 0.8715 | 0.8043 |
| RT-DETRv2 stride-4 | Blur | 0.8362 | 0.8648 | 0.7668 |
| RT-DETRv2 stride-4 | Low brightness | 0.9200 | 0.9005 | 0.8709 |
| RT-DETRv2 stride-4 | Occlusion | 0.8215 | 0.8875 | 0.7846 |
| RT-DETRv2 stride-4 | Denoised | 0.7761 | 0.8676 | 0.6950 |

### Summary

- RT-DETRv2 has the strongest original-image result: `0.9307` mAP@0.5 for the
  default model and `0.9271` for the stride-4 model.
- The default RT-DETRv2 model has the highest average mAP across the five
  altered conditions, approximately `0.8481`.
- YOLOv8 is most affected by blur (`0.7145` mAP@0.5) and noise (`0.7200`).
- RT-DETRv2 is more robust to noise, with `0.8587` mAP for the default model
  and `0.8514` for the stride-4 model.
- Low brightness has little effect on any model, while central occlusion causes
  a larger degradation.
- The denoising step does not recover mAP in this experiment: YOLOv8 reaches
  `0.7470`, compared with `0.7200` on noisy images, while both RT-DETRv2
  variants perform below their noisy-image scores after denoising.

The full machine-readable results are available in
[`all_models_robustness.csv`](results/all_models_robustness.csv), and the
visual comparison is available in
[`method2_method3_comparison.png`](results/method2_method3_comparison.png).
