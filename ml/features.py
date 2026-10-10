"""Shared raw hand-and-pose sequence feature extraction for training and inference."""

from __future__ import annotations

import numpy as np

from config import FEATURES_PER_FRAME, RAW_FRAME_SIZE, SEQUENCE_LENGTH

EPSILON = 1e-3
HAND_COORDINATE_COUNT = 21 * 3
POSE_LANDMARK_COUNT = 33
BODY_FEATURES_PER_FRAME = 63 + 42 + 8 + 4


def window_features(raw_window: np.ndarray) -> np.ndarray:
    """Convert (30, 129) raw windows into (30, 117) body-aware features.

    Feature layout per frame: A, 63 wrist-relative 3D hand-shape coordinates;
    B, 42 x/y hand coordinates relative to the shoulder midpoint and median
    shoulder-width scale; C, 8 x/y vectors from the nose to the wrist, thumb tip,
    index tip, and middle tip, using that same scale; D, 4 wrist body-space
    displacement-from-window-start and frame-to-frame velocity coordinates.

    The operations are vectorised over the complete window, so training and
    inference use the identical per-window coordinate normalization.
    """
    raw = np.asarray(raw_window, dtype=np.float32)
    expected_shape = (SEQUENCE_LENGTH, RAW_FRAME_SIZE)
    if raw.shape != expected_shape:
        raise ValueError(f"raw_window must have shape {expected_shape}, got {raw.shape}.")
    if not np.isfinite(raw).all():
        raise ValueError("raw_window must contain only finite coordinates.")

    hand = raw[:, :HAND_COORDINATE_COUNT].reshape(SEQUENCE_LENGTH, 21, 3)
    pose = raw[:, HAND_COORDINATE_COUNT:].reshape(SEQUENCE_LENGTH, POSE_LANDMARK_COUNT, 2)

    wrist = hand[:, 0, :]
    hand_scale = np.maximum(np.linalg.norm(hand[:, 9, :] - wrist, axis=1), EPSILON)
    hand_shape = ((hand - wrist[:, None, :]) / hand_scale[:, None, None]).reshape(
        SEQUENCE_LENGTH, 63
    )

    shoulders = pose[:, [11, 12], :]
    anchors = shoulders.mean(axis=1)
    shoulder_width = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    scale = max(float(np.median(shoulder_width)), EPSILON)
    body_hand = ((hand[:, :, :2] - anchors[:, None, :]) / scale).reshape(
        SEQUENCE_LENGTH, 42
    )

    hand_points = hand[:, [0, 4, 8, 12], :2]
    nose_vectors = (hand_points - pose[:, 0, None, :]) / scale
    nose_features = nose_vectors.reshape(SEQUENCE_LENGTH, 8)

    wrist_body = (wrist[:, :2] - anchors) / scale
    wrist_offset = wrist_body - wrist_body[0]
    wrist_velocity = np.zeros_like(wrist_body)
    wrist_velocity[1:] = wrist_body[1:] - wrist_body[:-1]
    motion_features = np.concatenate((wrist_offset, wrist_velocity), axis=1)

    features = np.concatenate(
        (hand_shape, body_hand, nose_features, motion_features), axis=1
    )
    assert features.shape == (SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    assert FEATURES_PER_FRAME == BODY_FEATURES_PER_FRAME
    return features.astype(np.float32, copy=False)
