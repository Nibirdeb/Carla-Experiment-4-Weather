
import numpy as np

# CARLA 0.9.15 semantic class IDs
ROAD_ID = 1
ROAD_LINES_ID = 24


def generate_lane_mask(semantic_labels):
    lane_mask = np.zeros(
        semantic_labels.shape,
        dtype=np.uint8
    )

    lane_mask[semantic_labels == ROAD_LINES_ID] = 255

    return lane_mask


def generate_road_mask(semantic_labels):
    road_mask = np.zeros(
        semantic_labels.shape,
        dtype=np.uint8
    )

    road_mask[semantic_labels == ROAD_ID] = 255

    return road_mask
