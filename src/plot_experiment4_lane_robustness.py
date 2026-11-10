
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "outputs"
    / "experiment4_evaluation"
    / "run_20261110_174348"
    / "weather_metrics_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "outputs"
    / "experiment4_evaluation"
    / "final_figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WEATHER_ORDER = [
    "clear_noon",
    "sunset",
    "light_rain",
    "heavy_rain",
    "light_fog",
    "dense_fog",
    "clear_night",
    "rainy_night",
]

WEATHER_LABELS = [
    "Clear Noon",
    "Sunset",
    "Light Rain",
    "Heavy Rain",
    "Light Fog",
    "Dense Fog",
    "Clear Night",
    "Rainy Night",
]

MODELS = {
    "Experiment 1": "Experiment 1 - U-Net",
    "Experiment 3": "Experiment 3 - Multitask U-Net",
}

data = {}

with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["task"] != "lane":
            continue

        model = row["model"]
        weather = row["weather"]

        if model in MODELS:
            data[(model, weather)] = float(row["iou_mean"])

missing = [
    (model, weather)
    for model in MODELS
    for weather in WEATHER_ORDER
    if (model, weather) not in data
]

if missing:
    raise ValueError(f"Missing metric entries: {missing}")

exp1_values = [data[("Experiment 1", w)] for w in WEATHER_ORDER]
exp3_values = [data[("Experiment 3", w)] for w in WEATHER_ORDER]

x = np.arange(len(WEATHER_ORDER))
width = 0.36

fig, ax = plt.subplots(figsize=(13, 6.5))

bars1 = ax.bar(
    x - width / 2,
    exp1_values,
    width,
    label=MODELS["Experiment 1"],
)

bars3 = ax.bar(
    x + width / 2,
    exp3_values,
    width,
    label=MODELS["Experiment 3"],
)

ax.set_title(
    "Lane Detection Robustness Across Weather Conditions",
    fontsize=15,
    pad=16,
)

ax.set_xlabel("Weather Condition", fontsize=12)
ax.set_ylabel("Macro IoU", fontsize=12)

ax.set_xticks(x)
ax.set_xticklabels(WEATHER_LABELS, rotation=25, ha="right")

ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

ax.legend(loc="upper right", frameon=True)

ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
ax.bar_label(bars3, fmt="%.3f", padding=2, fontsize=8)

fig.tight_layout()

png_path = OUTPUT_DIR / "figure4_lane_robustness.png"
pdf_path = OUTPUT_DIR / "figure4_lane_robustness.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

plt.close(fig)

print("Lane robustness figure created successfully.")
print(f"PNG: {png_path}")
print(f"PDF: {pdf_path}")
print("Metrics used: per-image macro IoU.")
