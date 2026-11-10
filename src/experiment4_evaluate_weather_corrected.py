from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

THESIS_DIR = PROJECT_DIR.parent

EXP1_DIR = THESIS_DIR / "Lane_Detection_project"
EXP2_DIR = THESIS_DIR / "Road Segmentation"
EXP3_DIR = THESIS_DIR / "Carla_MultiTask_Experiment"

EXP1_MODEL_FILE = EXP1_DIR / "src" / "model.py"
EXP2_MODEL_FILE = EXP2_DIR / "src" / "model.py"
EXP3_MODEL_FILE = EXP3_DIR / "src" / "model.py"

EXP1_CHECKPOINT = EXP1_DIR / "checkpoints" / "best_model.pth"
EXP2_CHECKPOINT = EXP2_DIR / "checkpoints_road" / "best_model.pth"
EXP3_CHECKPOINT = (
    EXP3_DIR
    / "experiment3_controlled"
    / "checkpoints"
    / "best_model.pth"
)

OUTPUT_ROOT = PROJECT_DIR / "outputs" / "experiment4_evaluation"

# The exact eight validated pilot runs.
WEATHER_RUNS = {
    "clear_noon": "run_20261107_230522",
    "sunset": "run_20261107_230818",
    "light_rain": "run_20261107_230902",
    "heavy_rain": "run_20261107_230945",
    "light_fog": "run_20261107_231030",
    "dense_fog": "run_20261107_231114",
    "clear_night": "run_20261107_231156",
    "rainy_night": "run_20261107_231333",
}

THRESHOLD = 0.5


# ============================================================
# 2. MODEL LOADING
# ============================================================

def import_model_module(module_path: Path, module_name: str):
    if not module_path.is_file():
        raise FileNotFoundError(f"Model definition not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model file: {module_path}")

    module = importlib.util.module_from_spec(spec)

    # Some model files use imports from their own project folder.
    project_src = str(module_path.parent)
    if project_src not in sys.path:
        sys.path.insert(0, project_src)

    spec.loader.exec_module(module)
    return module


def create_model_from_class(model_class, model_name: str):
    """
    Instantiate the exact model architecture used in each experiment.
    All three model classes use zero-argument constructors.
    """
    if model_name not in ("Experiment 1", "Experiment 2", "Experiment 3"):
        raise ValueError(f"Unknown model name: {model_name}")

    return model_class()


def get_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in (
            "model_state_dict",
            "model_state",
            "state_dict",
        ):
            if key in checkpoint:
                return checkpoint[key]

    # Also support a checkpoint that directly contains the state dict.
    if isinstance(checkpoint, dict) and checkpoint:
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint

    raise KeyError(
        "Could not locate model weights in checkpoint. "
        "Available keys: "
        f"{list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)}"
    )


def load_one_model(
    name: str,
    model_file: Path,
    checkpoint_file: Path,
    class_name: str,
    device: torch.device,
):
    print(f"\nLoading {name}...")

    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    module = import_model_module(
        model_file,
        f"experiment4_{name.lower().replace(' ', '_')}_model",
    )

    if not hasattr(module, class_name):
        raise AttributeError(
            f"Class '{class_name}' not found in {model_file}. "
            f"Available classes: "
            f"{[x for x in dir(module) if not x.startswith('_')]}"
        )

    model_class = getattr(module, class_name)
    model = create_model_from_class(model_class, name).to(device)

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )

    state = get_checkpoint_state(checkpoint)

    # Strict loading prevents silent architecture/checkpoint mismatches.
    model.load_state_dict(state, strict=True)
    model.eval()

    print(f"  Model class: {class_name}")
    print(f"  Checkpoint: {checkpoint_file}")
    print("  Strict checkpoint loading: PASSED")

    return model


# ============================================================
# 3. DATASET DISCOVERY
# ============================================================

def find_run_directory(run_id: str) -> Path:
    matches = [
        path
        for path in PROJECT_DIR.rglob(run_id)
        if path.is_dir() and path.name == run_id
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find dataset run '{run_id}' under {PROJECT_DIR}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"More than one folder named '{run_id}' was found:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


def find_image_folder(run_dir: Path, kind: str) -> Path:
    """
    Locate the RGB, lane-mask or road-mask directory.
    Supports common folder-name variations.
    """
    directories = [p for p in run_dir.rglob("*") if p.is_dir()]

    if kind == "rgb":
        accepted = {"rgb", "images", "rgb_images"}
    elif kind == "lane":
        accepted = {"lane_masks", "lane_mask", "lanemasks"}
    elif kind == "road":
        accepted = {"road_masks", "road_mask", "roadmasks"}
    else:
        raise ValueError(f"Unknown folder type: {kind}")

    matches = [
        path for path in directories
        if path.name.lower() in accepted
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {kind} folder in {run_dir}, "
            f"but found {len(matches)}:\n"
            + "\n".join(str(path) for path in matches)
        )

    return matches[0]


def get_weather_samples(weather_name: str, run_id: str):
    run_dir = find_run_directory(run_id)

    rgb_dir = find_image_folder(run_dir, "rgb")
    lane_dir = find_image_folder(run_dir, "lane")
    road_dir = find_image_folder(run_dir, "road")

    rgb_files = sorted(rgb_dir.glob("*.png"))
    lane_files = sorted(lane_dir.glob("*.png"))
    road_files = sorted(road_dir.glob("*.png"))

    if not rgb_files:
        raise RuntimeError(f"No RGB PNG images found in {rgb_dir}")

    rgb_by_name = {p.name: p for p in rgb_files}
    lane_by_name = {p.name: p for p in lane_files}
    road_by_name = {p.name: p for p in road_files}

    if set(rgb_by_name) != set(lane_by_name):
        raise RuntimeError(
            f"RGB and lane-mask filenames do not match for {weather_name}"
        )

    if set(rgb_by_name) != set(road_by_name):
        raise RuntimeError(
            f"RGB and road-mask filenames do not match for {weather_name}"
        )

    filenames = sorted(rgb_by_name)

    samples = [
        {
            "weather": weather_name,
            "filename": filename,
            "rgb_path": rgb_by_name[filename],
            "lane_path": lane_by_name[filename],
            "road_path": road_by_name[filename],
        }
        for filename in filenames
    ]

    return run_dir, samples


# ============================================================
# 4. IMAGE PREPROCESSING
# ============================================================

def load_rgb(path: Path, size: tuple[int, int]) -> torch.Tensor:
    """
    PIL reads the saved PNG as RGB.
    size is (width, height).
    """
    image = Image.open(path).convert("RGB")
    image = image.resize(size, Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())

    return tensor.unsqueeze(0)


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    """
    Nearest-neighbour resizing preserves binary mask labels.
    """
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, Image.Resampling.NEAREST)

    array = np.asarray(mask, dtype=np.uint8)
    return array > 127


# ============================================================
# 5. METRICS
# ============================================================

def calculate_metrics(prediction: np.ndarray, target: np.ndarray):
    pred = prediction.astype(bool)
    truth = target.astype(bool)

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())

    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0

    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 1.0
    )

    # Undefined when there are no predicted positive pixels.
    precision = tp / (tp + fp) if tp + fp else float("nan")

    # Undefined when the ground truth contains no positive pixels.
    recall = tp / (tp + fn) if tp + fn else float("nan")

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
# ============================================================
# 6. MODEL INFERENCE
# ============================================================

def extract_single_output(output):
    if isinstance(output, dict):
        # Some segmentation implementations return a dictionary.
        if "out" in output:
            return output["out"]

        raise RuntimeError(
            f"Unexpected single-task model output keys: {list(output.keys())}"
        )

    if isinstance(output, (tuple, list)):
        return output[0]

    return output


def predict_single_task(model, image_tensor, device):
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        output = model(image_tensor)
        output = extract_single_output(output)
        probabilities = torch.sigmoid(output)

    return probabilities[0, 0].cpu().numpy() >= THRESHOLD


def predict_multitask(model, image_tensor, device):
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        output = model(image_tensor)

        if not isinstance(output, dict):
            raise RuntimeError(
                "Experiment 3 model was expected to return a dictionary "
                "with 'lane' and 'road' outputs."
            )

        if "lane" not in output or "road" not in output:
            raise RuntimeError(
                f"Unexpected multitask output keys: {list(output.keys())}"
            )

        lane_prediction = (
            torch.sigmoid(output["lane"])[0, 0].cpu().numpy() >= THRESHOLD
        )

        road_prediction = (
            torch.sigmoid(output["road"])[0, 0].cpu().numpy() >= THRESHOLD
        )

    return lane_prediction, road_prediction


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_single_task(
    model,
    samples,
    task: str,
    model_name: str,
    image_size: tuple[int, int],
    device: torch.device,
):
    results = []

    for index, sample in enumerate(samples, start=1):
        image = load_rgb(sample["rgb_path"], image_size)

        mask_path = sample[f"{task}_path"]
        target = load_mask(mask_path, image_size)

        prediction = predict_single_task(model, image, device)

        metrics = calculate_metrics(prediction, target)

        results.append({
            "model": model_name,
            "task": task,
            "weather": sample["weather"],
            "filename": sample["filename"],
            **metrics,
        })

        if index % 25 == 0 or index == len(samples):
            print(
                f"    {model_name} / {task}: "
                f"{index}/{len(samples)}"
            )

    return results


def evaluate_multitask(
    model,
    samples,
    device: torch.device,
):
    results = []

    for index, sample in enumerate(samples, start=1):
        image = load_rgb(sample["rgb_path"], (1248, 384))

        lane_target = load_mask(sample["lane_path"], (1248, 384))
        road_target = load_mask(sample["road_path"], (1248, 384))

        lane_prediction, road_prediction = predict_multitask(
            model, image, device
        )

        for task, prediction, target in (
            ("lane", lane_prediction, lane_target),
            ("road", road_prediction, road_target),
        ):
            metrics = calculate_metrics(prediction, target)

            results.append({
                "model": "Experiment 3",
                "task": task,
                "weather": sample["weather"],
                "filename": sample["filename"],
                **metrics,
            })

        if index % 25 == 0 or index == len(samples):
            print(f"    Experiment 3: {index}/{len(samples)}")

    return results


# ============================================================
# 8. CSV AND METADATA OUTPUT
# ============================================================

def save_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_summary(per_image_rows):
    summary = []

    for model_name in (
        "Experiment 1",
        "Experiment 2",
        "Experiment 3",
    ):
        for task in ("lane", "road"):
            matching = [
                row for row in per_image_rows
                if row["model"] == model_name and row["task"] == task
            ]

            if not matching:
                continue

            weather_names = list(WEATHER_RUNS.keys())

            for weather in weather_names:
                weather_rows = [
                    row for row in matching
                    if row["weather"] == weather
                ]

                if not weather_rows:
                    continue

                summary_row = {
                    "model": model_name,
                    "task": task,
                    "weather": weather,
                    "image_count": len(weather_rows),
                }

                for metric in ("iou", "dice", "precision", "recall"):
                    values = np.asarray(
                        [float(row[metric]) for row in weather_rows],
                        dtype=np.float64,
                    )

                    valid_values = values[~np.isnan(values)]

                    summary_row[f"{metric}_mean"] = (
                        float(np.mean(valid_values))
                        if len(valid_values) > 0 else float("nan")
                    )

                    summary_row[f"{metric}_std"] = (
                        float(np.std(valid_values))
                        if len(valid_values) > 0 else float("nan")
                    )

                    summary_row[f"{metric}_undefined_count"] = int(
                        np.isnan(values).sum()
                    )

                summary.append(summary_row)

    return summary

# ============================================================
# 9. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Weather robustness evaluation"
    )

    parser.add_argument(
        "--limit-per-weather",
        type=int,
        default=2,
        help=(
            "Number of images per weather for the initial test. "
            "Use 0 to evaluate all images."
        ),
    )

    args = parser.parse_args()

    if args.limit_per_weather < 0:
        raise ValueError("--limit-per-weather must be 0 or greater.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("EXPERIMENT 4 - WEATHER ROBUSTNESS EVALUATION")
    print("=" * 65)
    print(f"Project: {PROJECT_DIR}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Discover and validate all eight datasets first.
    all_weather_samples = {}
    dataset_metadata = {}

    print("\nChecking the eight pilot datasets...")

    for weather, run_id in WEATHER_RUNS.items():
        run_dir, samples = get_weather_samples(weather, run_id)

        if args.limit_per_weather:
            samples = samples[:args.limit_per_weather]

        if not samples:
            raise RuntimeError(f"No samples available for {weather}")

        all_weather_samples[weather] = samples
        dataset_metadata[weather] = {
            "run_id": run_id,
            "run_directory": str(run_dir),
            "images_evaluated": len(samples),
        }

        print(
            f"  {weather:15s}: {len(samples)} images "
            f"({run_id})"
        )

    # Load the three frozen models.
    exp1_model = load_one_model(
        "Experiment 1",
        EXP1_MODEL_FILE,
        EXP1_CHECKPOINT,
        "UNet",
        device,
    )

    exp2_model = load_one_model(
        "Experiment 2",
        EXP2_MODEL_FILE,
        EXP2_CHECKPOINT,
        "DeepLabV3Plus",
        device,
    )

    exp3_model = load_one_model(
        "Experiment 3",
        EXP3_MODEL_FILE,
        EXP3_CHECKPOINT,
        "MultiTaskUNet",
        device,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    all_results = []

    print("\nStarting evaluation...")

    for weather, samples in all_weather_samples.items():
        print(f"\nWEATHER: {weather.upper()}")

        # Experiment 1: lane detection at 512 x 256.
        all_results.extend(
            evaluate_single_task(
                exp1_model,
                samples,
                "lane",
                "Experiment 1",
                (512, 256),
                device,
            )
        )

        # Experiment 2: road segmentation at 512 x 256.
        all_results.extend(
            evaluate_single_task(
                exp2_model,
                samples,
                "road",
                "Experiment 2",
                (512, 256),
                device,
            )
        )

        # Experiment 3: both tasks at native 1248 x 384.
        all_results.extend(
            evaluate_multitask(
                exp3_model,
                samples,
                device,
            )
        )

    per_image_fields = [
        "model",
        "task",
        "weather",
        "filename",
        "iou",
        "dice",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
    ]

    per_image_path = output_dir / "weather_metrics_per_image.csv"
    save_csv(per_image_path, all_results, per_image_fields)

    summary = create_summary(all_results)

    summary_fields = [
        "model",
        "task",
        "weather",
        "image_count",
        "iou_mean",
        "iou_std",
        "iou_undefined_count",
        "dice_mean",
        "dice_std",
        "dice_undefined_count",
        "precision_mean",
        "precision_std",
        "precision_undefined_count",
        "recall_mean",
        "recall_std",
        "recall_undefined_count",
    ]

    summary_path = output_dir / "weather_metrics_summary.csv"
    save_csv(summary_path, summary, summary_fields)

    metadata = {
        "experiment": "Experiment 4 - Weather Robustness Evaluation",
        "evaluation_timestamp": datetime.now().isoformat(),
        "project_directory": str(PROJECT_DIR),
        "device": str(device),
        "threshold": THRESHOLD,
        "averaging": "Mean and standard deviation of per-image metrics",
        "metrics": ["IoU", "Dice", "Precision", "Recall"],
        "prediction_threshold": THRESHOLD,
        "mask_threshold": 127,
        "experiment1_input_resolution": "512x256",
        "experiment2_input_resolution": "512x256",
        "experiment3_input_resolution": "1248x384",
        "mask_resize_method": "Nearest neighbour",
        "rgb_resize_method": "Bilinear",
        "checkpoints": {
            "experiment1": str(EXP1_CHECKPOINT),
            "experiment2": str(EXP2_CHECKPOINT),
            "experiment3": str(EXP3_CHECKPOINT),
        },
        "datasets": dataset_metadata,
        "evaluation_mode": (
            "Pilot test" if args.limit_per_weather
            else "Full pilot dataset evaluation"
        ),
        "limit_per_weather": args.limit_per_weather,
        "total_images_per_model": sum(
            len(samples) for samples in all_weather_samples.values()
        ),
    }

    metadata_path = output_dir / "evaluation_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)
    print(f"Output directory: {output_dir}")
    print(f"Per-image results: {per_image_path}")
    print(f"Weather summary: {summary_path}")
    print(f"Metadata: {metadata_path}")

    print("\nWeather summary:")
    for row in summary:
        print(
            f"{row['model']:14s} | {row['task']:4s} | "
            f"{row['weather']:12s} | "
            f"IoU {row['iou_mean']:.4f} | "
            f"Dice {row['dice_mean']:.4f}"
        )


if __name__ == "__main__":
    main()