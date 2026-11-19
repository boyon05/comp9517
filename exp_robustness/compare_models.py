"""Combine results from Method 2 and Method 3, and plot a comparison of their robustness metrics."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "exp_robustness/results"

m2 = pd.read_csv(RESULTS / "method2_unified_metrics.csv")
m3 = pd.read_csv(RESULTS / "method3_robustness_metrics.csv")

df = pd.concat([m2, m3], ignore_index=True)

names = {
    "method2": "YOLOv8",
    "default": "RT-DETRv2 Default",
    "stride4": "RT-DETRv2 Stride4",
}

conditions = [
    "original", "noise", "blur",
    "low_brightness", "occlusion", "denoised"
]

labels = [
    "Original", "Noise", "Blur",
    "Low Brightness", "Occlusion", "Denoised"
]

plt.figure(figsize=(11, 6))

for model, label in names.items():
    subset = df[df["model"] == model].set_index("condition")
    values = [subset.loc[c, "map50"] for c in conditions]
    plt.plot(labels, values, marker="o", linewidth=2, label=label)

plt.title("Robustness Comparison: YOLOv8 vs RT-DETRv2")
plt.xlabel("Test Condition")
plt.ylabel("mAP@0.5")
plt.ylim(0.6, 1.0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output = RESULTS / "method2_method3_comparison.png"
plt.savefig(output, dpi=300)
plt.close()

df.to_csv(RESULTS / "all_models_robustness.csv", index=False)

print("Saved:", output)
print("Saved:", RESULTS / "all_models_robustness.csv")
