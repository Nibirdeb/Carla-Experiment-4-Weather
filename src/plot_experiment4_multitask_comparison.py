
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR / "outputs" / "experiment4_evaluation"
    / "run_20261110_174348" / "weather_metrics_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR / "outputs" / "experiment4_evaluation"
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

data = {}

with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    for row in csv.DictReader(file):
        if row["model"] == "Experiment 3":
            data[(row["task"], row["weather"])] = float(row["iou_mean"])

missing = [
    (task, weather)
    for task in ["lane", "road"]
    for weather in WEATHER_ORDER
    if (task, weather) not in data
]

if missing:
    raise ValueError(f"Missing metric entries: {missing}")

lane_values = [data[("lane", w)] for w in WEATHER_ORDER]
road_values = [data[("road", w)] for w in WEATHER_ORDER]

x = np.arange(len(WEATHER_ORDER))
width = 0.36

fig, ax = plt.subplots(figsize=(13, 6.5))

lane_bars = ax.bar(
    x - width / 2,
    lane_values,
    width,
    label="Lane Segmentation",
)

road_bars = ax.bar(
    x + width / 2,
    road_values,
    width,
    label="Road Segmentation",
)

ax.set_title(
    "Experiment 3: Lane and Road Segmentation Across Weather Conditions",
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

ax.legend(loc="upper right")

ax.bar_label(lane_bars, fmt="%.3f", padding=2, fontsize=8)
ax.bar_label(road_bars, fmt="%.3f", padding=2, fontsize=8)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure4_multitask_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "figure4_multitask_comparison.pdf",
    bbox_inches="tight",
)

plt.close(fig)

print("Experiment 3 multitask comparison figure created successfully.")
print("PNG and PDF saved in:", OUTPUT_DIR)
