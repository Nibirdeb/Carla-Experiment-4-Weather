import carla

from config import (
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    CAMERA_X,
    CAMERA_Y,
    CAMERA_Z,
    CAMERA_FOV,
)


def create_rgb_camera(world, blueprints, vehicle):

    camera_bp = blueprints.find("sensor.camera.rgb")

    camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    camera_bp.set_attribute("fov", str(CAMERA_FOV))

    transform = carla.Transform(
        carla.Location(
            x=CAMERA_X,
            y=CAMERA_Y,
            z=CAMERA_Z
        )
    )

    camera = world.spawn_actor(
        camera_bp,
        transform,
        attach_to=vehicle
    )

    return camera


def create_semantic_camera(world, blueprints, vehicle):

    camera_bp = blueprints.find(
        "sensor.camera.semantic_segmentation"
    )

    camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    camera_bp.set_attribute("fov", str(CAMERA_FOV))

    transform = carla.Transform(
        carla.Location(
            x=CAMERA_X,
            y=CAMERA_Y,
            z=CAMERA_Z
        )
    )

    camera = world.spawn_actor(
        camera_bp,
        transform,
        attach_to=vehicle
    )

    return camera