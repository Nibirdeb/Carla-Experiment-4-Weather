
import carla
import numpy as np
import queue

HOST = "localhost"
PORT = 2000
WIDTH = 1248
HEIGHT = 384

client = carla.Client(HOST, PORT)
client.set_timeout(20.0)

world = client.get_world()
blueprints = world.get_blueprint_library()

vehicle = None
camera = None

try:
    spawn_points = world.get_map().get_spawn_points()

    for spawn_point in spawn_points:
        vehicle = world.try_spawn_actor(
            blueprints.find("vehicle.tesla.model3"),
            spawn_point
        )
        if vehicle is not None:
            break

    if vehicle is None:
        raise RuntimeError("Could not spawn a diagnostic vehicle.")

    camera_bp = blueprints.find("sensor.camera.semantic_segmentation")
    camera_bp.set_attribute("image_size_x", str(WIDTH))
    camera_bp.set_attribute("image_size_y", str(HEIGHT))
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=2.4)
    )

    camera = world.spawn_actor(
        camera_bp,
        camera_transform,
        attach_to=vehicle
    )

    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    print("Waiting for semantic camera frame...")

    image = image_queue.get(timeout=15.0)

    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    raw = raw.reshape((image.height, image.width, 4))

    print("\nImage shape:", raw.shape)

    for channel in range(4):
        values, counts = np.unique(
            raw[:, :, channel],
            return_counts=True
        )

        print(f"\nChannel {channel}:")
        print(dict(zip(values.tolist(), counts.tolist())))

    labels = raw[:, :, 2]
    values, counts = np.unique(labels, return_counts=True)

    print("\nCARLA class IDs (channel 2):")
    print(dict(zip(values.tolist(), counts.tolist())))

finally:
    if camera is not None:
        camera.stop()
        camera.destroy()

    if vehicle is not None:
        vehicle.destroy()

    print("\nDiagnostic actors cleaned up.")
