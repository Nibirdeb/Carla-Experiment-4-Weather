from pathlib import Path

# ==========================================================
# Experiment 4 - Weather Dataset Configuration
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

# ==========================================================
# CARLA Connection
# ==========================================================

HOST = "localhost"
PORT = 2000

# ==========================================================
# Dataset Parameters
# ==========================================================

IMAGE_WIDTH = 1248
IMAGE_HEIGHT = 384

FPS = 10

# Number of images per weather condition
NUM_IMAGES = 1000

# Save one image every N simulation frames
SAVE_EVERY_N_FRAMES = 10

# ==========================================================
# Vehicle Configuration
# ==========================================================

VEHICLE_BLUEPRINT = "vehicle.tesla.model3"

# ==========================================================
# Camera Configuration
# ==========================================================

CAMERA_X = 1.5
CAMERA_Y = 0.0
CAMERA_Z = 2.4

CAMERA_FOV = 90

# ==========================================================
# Weather Conditions
# ==========================================================

WEATHER_CONDITIONS = [
    "clear_noon",
    "sunset",
    "light_rain",
    "heavy_rain",
    "light_fog",
    "dense_fog",
    "clear_night",
    "rainy_night",
]