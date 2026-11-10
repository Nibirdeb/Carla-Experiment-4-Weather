
import argparse
import csv
import queue
import random
from datetime import datetime
from pathlib import Path

import carla
import cv2
import numpy as np

from config import (
    HOST,
    PORT,
    FPS,
    NUM_IMAGES,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    VEHICLE_BLUEPRINT,
    WEATHER_CONDITIONS,
)

from camera import create_rgb_camera, create_semantic_camera
from mask_generator import generate_lane_mask, generate_road_mask


# ==========================================================
# Weather configurations
# ==========================================================

WEATHER_PRESETS = {
    "clear_noon": {
        "cloudiness": 0.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 0.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 75.0,
        "fog_density": 0.0,
        "fog_distance": 1000.0,
        "wetness": 0.0,
    },

    "sunset": {
        "cloudiness": 10.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 0.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 5.0,
        "fog_density": 0.0,
        "fog_distance": 1000.0,
        "wetness": 0.0,
    },

    "light_rain": {
        "cloudiness": 60.0,
        "precipitation": 25.0,
        "precipitation_deposits": 20.0,
        "wind_intensity": 20.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 60.0,
        "fog_density": 0.0,
        "fog_distance": 1000.0,
        "wetness": 30.0,
    },

    "heavy_rain": {
        "cloudiness": 90.0,
        "precipitation": 90.0,
        "precipitation_deposits": 80.0,
        "wind_intensity": 80.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 45.0,
        "fog_density": 0.0,
        "fog_distance": 1000.0,
        "wetness": 90.0,
    },

    "light_fog": {
        "cloudiness": 20.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 0.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 60.0,
        "fog_density": 20.0,
        "fog_distance": 75.0,
        "wetness": 0.0,
    },

    "dense_fog": {
        "cloudiness": 40.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 0.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": 45.0,
        "fog_density": 80.0,
        "fog_distance": 20.0,
        "wetness": 0.0,
    },

    "clear_night": {
        "cloudiness": 0.0,
        "precipitation": 0.0,
        "precipitation_deposits": 0.0,
        "wind_intensity": 0.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": -20.0,
        "fog_density": 0.0,
        "fog_distance": 1000.0,
        "wetness": 0.0,
    },

    "rainy_night": {
        "cloudiness": 90.0,
        "precipitation": 70.0,
        "precipitation_deposits": 70.0,
        "wind_intensity": 50.0,
        "sun_azimuth_angle": 0.0,
        "sun_altitude_angle": -20.0,
        "fog_density": 0.0,
        "fog_distance": 1000.0,
        "wetness": 80.0,
    },
}


# ==========================================================
# Dataset Generator
# ==========================================================

class WeatherDatasetGenerator:

    def __init__(self, weather_name, num_images, save_every):

        self.weather_name = weather_name
        self.num_images = num_images
        self.save_every = save_every

        self.base_dir = Path(__file__).resolve().parent.parent

        self.weather_dir = (
            self.base_dir / "data" / weather_name
        )

        # Unique folder prevents overwriting previous runs.
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.output_dir = (
            self.weather_dir / f"run_{run_id}"
        )

        self.rgb_dir = self.output_dir / "rgb"
        self.lane_dir = self.output_dir / "lane_masks"
        self.road_dir = self.output_dir / "road_masks"

        # Fail rather than overwrite if the run folder exists.
        self.rgb_dir.mkdir(parents=True, exist_ok=False)
        self.lane_dir.mkdir(parents=True, exist_ok=False)
        self.road_dir.mkdir(parents=True, exist_ok=False)

        self.client = carla.Client(HOST, PORT)
        self.client.set_timeout(20.0)

        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()

        self.blueprints = self.world.get_blueprint_library()

        self.vehicle = None
        self.rgb_camera = None
        self.semantic_camera = None

        self.rgb_queue = queue.Queue()
        self.semantic_queue = queue.Queue()

        self.saved_count = 0

    # ------------------------------------------------------
    # Apply weather
    # ------------------------------------------------------

    def apply_weather(self):

        values = WEATHER_PRESETS[self.weather_name]

        weather = carla.WeatherParameters(**values)

        self.world.set_weather(weather)

        print("\nWeather applied:", self.weather_name)
        print(values)

    # ------------------------------------------------------
    # Synchronous mode
    # ------------------------------------------------------

    def enable_synchronous_mode(self):

        settings = self.world.get_settings()

        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS

        self.world.apply_settings(settings)

        print("Synchronous mode enabled.")

    # ------------------------------------------------------
    # Spawn vehicle
    # ------------------------------------------------------

    def spawn_vehicle(self):

        vehicle_bp = self.blueprints.find(
            VEHICLE_BLUEPRINT
        )

        spawn_points = (
            self.world.get_map().get_spawn_points()
        )

        if not spawn_points:
            raise RuntimeError(
                "No vehicle spawn points available."
            )

        random.shuffle(spawn_points)

        for spawn_point in spawn_points:

            self.vehicle = self.world.try_spawn_actor(
                vehicle_bp,
                spawn_point
            )

            if self.vehicle is not None:
                break

        if self.vehicle is None:
            raise RuntimeError(
                "Vehicle could not be spawned."
            )

        self.vehicle.set_autopilot(True)

        print("Vehicle spawned:", self.vehicle.id)

    # ------------------------------------------------------
    # Spawn cameras
    # ------------------------------------------------------

    def spawn_cameras(self):

        self.rgb_camera = create_rgb_camera(
            self.world,
            self.blueprints,
            self.vehicle
        )

        self.semantic_camera = create_semantic_camera(
            self.world,
            self.blueprints,
            self.vehicle
        )

        self.rgb_camera.listen(
            self.rgb_queue.put
        )

        self.semantic_camera.listen(
            self.semantic_queue.put
        )

        print("RGB camera attached.")
        print("Semantic camera attached.")

    # ------------------------------------------------------
    # Retrieve synchronized frame
    # ------------------------------------------------------

    @staticmethod
    def get_frame_for_tick(sensor_queue, expected_frame):

        while True:

            measurement = sensor_queue.get(
                timeout=10.0
            )

            if measurement.frame == expected_frame:
                return measurement

            if measurement.frame > expected_frame:

                raise RuntimeError(
                    f"Sensor skipped frame {expected_frame}. "
                    f"Received frame {measurement.frame}."
                )

    # ------------------------------------------------------
    # Convert RGB
    # ------------------------------------------------------

    @staticmethod
    def convert_rgb(image):

        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        )

        array = array.reshape(
            (image.height, image.width, 4)
        )

        return array[:, :, :3].copy()

    # ------------------------------------------------------
    # Convert semantic labels
    # ------------------------------------------------------

    @staticmethod
    def convert_semantic(image):

        array = np.frombuffer(
            image.raw_data,
            dtype=np.uint8
        )

        array = array.reshape(
            (image.height, image.width, 4)
        )

        # Preserves the channel convention used
        # in the original CARLA generator.
        return array[:, :, 2].copy()

    # ------------------------------------------------------
    # Save sample
    # ------------------------------------------------------

    def save_sample(
        self,
        rgb,
        lane_mask,
        road_mask,
        frame_id,
        writer
    ):

        filename = f"{self.saved_count:06d}.png"

        rgb_path = self.rgb_dir / filename
        lane_path = self.lane_dir / filename
        road_path = self.road_dir / filename

        if not cv2.imwrite(str(rgb_path), rgb):
            raise IOError(f"Failed to save {rgb_path}")

        if not cv2.imwrite(str(lane_path), lane_mask):
            raise IOError(f"Failed to save {lane_path}")

        if not cv2.imwrite(str(road_path), road_mask):
            raise IOError(f"Failed to save {road_path}")

        writer.writerow([
            filename,
            self.weather_name,
            frame_id,
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        ])

        self.saved_count += 1

    # ------------------------------------------------------
    # Main collection
    # ------------------------------------------------------

    def run(self):

        metadata_path = (
            self.output_dir / "metadata.csv"
        )

        try:

            self.apply_weather()

            self.enable_synchronous_mode()

            # Let weather and lighting settle.
            for _ in range(20):
                self.world.tick()

            self.spawn_vehicle()
            self.spawn_cameras()

            # Allow sensors to start.
            for _ in range(5):
                self.world.tick()

            print("\nStarting collection.")
            print("Weather:", self.weather_name)
            print("Target images:", self.num_images)
            print("Save interval:", self.save_every)
            print("Output:", self.output_dir)

            with open(
                metadata_path,
                "w",
                newline="",
                encoding="utf-8"
            ) as csv_file:

                writer = csv.writer(csv_file)

                writer.writerow([
                    "filename",
                    "weather",
                    "carla_frame",
                    "image_width",
                    "image_height",
                ])

                simulation_frames = 0

                while self.saved_count < self.num_images:

                    expected_frame = self.world.tick()

                    simulation_frames += 1

                    # Consume both sensor measurements
                    # for every simulation frame.
                    rgb_measurement = self.get_frame_for_tick(
                        self.rgb_queue,
                        expected_frame
                    )

                    semantic_measurement = self.get_frame_for_tick(
                        self.semantic_queue,
                        expected_frame
                    )

                    if simulation_frames % self.save_every != 0:
                        continue

                    rgb = self.convert_rgb(
                        rgb_measurement
                    )

                    labels = self.convert_semantic(
                        semantic_measurement
                    )

                    lane_mask = generate_lane_mask(
                        labels
                    )

                    road_mask = generate_road_mask(
                        labels
                    )

                    if rgb.shape[:2] != (
                        IMAGE_HEIGHT,
                        IMAGE_WIDTH
                    ):
                        raise RuntimeError(
                            f"Unexpected RGB dimensions: {rgb.shape}"
                        )

                    if lane_mask.shape != (
                        IMAGE_HEIGHT,
                        IMAGE_WIDTH
                    ):
                        raise RuntimeError(
                            "Unexpected lane-mask dimensions."
                        )

                    if road_mask.shape != (
                        IMAGE_HEIGHT,
                        IMAGE_WIDTH
                    ):
                        raise RuntimeError(
                            "Unexpected road-mask dimensions."
                        )

                    self.save_sample(
                        rgb,
                        lane_mask,
                        road_mask,
                        expected_frame,
                        writer
                    )

                    if self.saved_count % 10 == 0:

                        csv_file.flush()

                        print(
                            f"[{self.saved_count}/"
                            f"{self.num_images}] "
                            f"{self.weather_name}"
                        )

            print("\nCollection completed.")
            print("Images saved:", self.saved_count)
            print("Output folder:", self.output_dir)

        finally:
            self.cleanup()

    # ------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------

    def cleanup(self):

        print("\nCleaning up CARLA actors...")

        for sensor in [
            self.rgb_camera,
            self.semantic_camera
        ]:

            if sensor is not None:

                try:
                    sensor.stop()
                except Exception:
                    pass

                try:
                    sensor.destroy()
                except Exception:
                    pass

        if self.vehicle is not None:

            try:
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
            except Exception:
                pass

        try:

            self.world.apply_settings(
                self.original_settings
            )

            print("Original world settings restored.")

        except Exception as error:

            print(
                "Could not restore world settings:",
                error
            )

        cv2.destroyAllWindows()


# ==========================================================
# Command-line interface
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Experiment 4 CARLA Weather Dataset Generator"
    )

    parser.add_argument(
        "--weather",
        required=True,
        choices=WEATHER_CONDITIONS
    )

    parser.add_argument(
        "--num-images",
        type=int,
        default=NUM_IMAGES
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10
    )

    args = parser.parse_args()

    if args.num_images < 1:
        parser.error("--num-images must be positive.")

    if args.save_every < 1:
        parser.error("--save-every must be positive.")

    generator = WeatherDatasetGenerator(
        args.weather,
        args.num_images,
        args.save_every
    )

    generator.run()


if __name__ == "__main__":
    main()
