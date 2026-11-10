
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

MODEL_TASKS = [
    ("Experiment 1", "lane", "Exp 1 - U-Net (Lane)"),
    ("Experiment 2", "road", "Exp 2 - DeepLabV3+ (Road)"),
    ("Experiment 3", "lane", "Exp 3 - Multitask (Lane)"),
    ("Experiment 3", "road", "Exp 3 - Multitask (Road)"),
]

data = {}

with INPUT_FILE.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        key = (row["model"], row["task"], row["weather"])
        data[key] = float(row["iou_mean"])

missing = [
    (model, task, weather)
    for model, task, _ in MODEL_TASKS
    for weather in WEATHER_ORDER
    if (model, task, weather) not in data
]

if missing:
    raise ValueError(f"Missing metric entries: {missing}")

matrix = np.array([
    [
        data[(model, task, weather)]
        for weather in WEATHER_ORDER
    ]
    for model, task, _ in MODEL_TASKS
])

fig, ax = plt.subplots(figsize=(13, 5.5))

image = ax.imshow(
    matrix,
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    aspect="auto",
)

ax.set_title(
    "Experiment 4: Weather Robustness Overview",
    fontsize=15,
    pad=16,
)

ax.set_xlabel("Weather Condition", fontsize=12)
ax.set_ylabel("Model and Segmentation Task", fontsize=12)

ax.set_xticks(np.arange(len(WEATHER_LABELS)))
ax.set_xticklabels(WEATHER_LABELS, rotation=25, ha="right")

ax.set_yticks(np.arange(len(MODEL_TASKS)))
ax.set_yticklabels([item[2] for item in MODEL_TASKS])

# Add metric values to each heatmap cell.
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        value = matrix[i, j]

        text_color = "white" if value < 0.25 or value > 0.75 else "black"

        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=9,
        )

colorbar = fig.colorbar(image, ax=ax, pad=0.02)
colorbar.set_label("Macro IoU", fontsize=11)

ax.set_xticks(np.arange(-0.5, len(WEATHER_LABELS), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(MODEL_TASKS), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
ax.tick_params(which="minor", bottom=False, left=False)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "figure4_weather_robustness_heatmap.png",
    dpi=300,
    bbox_inches="tight",
)

fig.savefig(
    OUTPUT_DIR / "figure4_weather_robustness_heatmap.pdf",
    bbox_inches="tight",
)

plt.close(fig)

print("Experiment 4 weather robustness heatmap created successfully.")
print("PNG and PDF saved in:", OUTPUT_DIR)
print("Heatmap dimensions:", matrix.shape)
print("Metrics used: per-image macro IoU.")
