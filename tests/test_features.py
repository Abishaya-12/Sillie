"""Tests for the shared body-aware sequence feature transform."""

from __future__ import annotations

import numpy as np

from config import FEATURES_PER_FRAME, RAW_FRAME_SIZE, SEQUENCE_LENGTH
from ml.features import window_features


def make_window(hand_height: float = 0.42) -> np.ndarray:
    """Build a repeatable hand-and-pose sequence in normalized image coordinates."""
    window = np.zeros((SEQUENCE_LENGTH, RAW_FRAME_SIZE), dtype=np.float64)
    hand = window[:, :63].reshape(SEQUENCE_LENGTH, 21, 3)
    pose = window[:, 63:].reshape(SEQUENCE_LENGTH, 33, 2)

    pose[:, 0] = [0.5, 0.2]
    pose[:, 7] = [0.35, 0.25]
    pose[:, 8] = [0.65, 0.25]
    pose[:, 11] = [0.4, 0.4]
    pose[:, 12] = [0.6, 0.4]

    point_offsets = np.stack(
        (
            np.linspace(0.0, 0.08, 21),
            np.linspace(0.0, 0.12, 21),
            np.linspace(0.0, 0.03, 21),
        ),
        axis=1,
    )
    hand[:] = point_offsets[None, :, :]
    hand[:, :, 0] += 0.5
    hand[:, :, 1] += hand_height
    hand[:, :, 2] += np.linspace(0.0, 0.01, SEQUENCE_LENGTH)[:, None]
    return window


def test_window_features_output_shape() -> None:
    """Features have one 117-value row for each configured sequence frame."""
    features = window_features(make_window())

    assert features.shape == (SEQUENCE_LENGTH, FEATURES_PER_FRAME)


def test_common_translation_does_not_change_features() -> None:
    """Applying the same x/y translation to hand and pose is cancelled."""
    original = make_window()
    translated = original.copy()
    hand = translated[:, :63].reshape(SEQUENCE_LENGTH, 21, 3)
    pose = translated[:, 63:].reshape(SEQUENCE_LENGTH, 33, 2)
    hand[:, :, :2] += [0.13, -0.07]
    pose += [0.13, -0.07]

    np.testing.assert_allclose(
        window_features(translated), window_features(original), atol=1e-6
    )


def test_uniform_coordinate_scale_does_not_change_features() -> None:
    """Uniformly scaling all hand and pose coordinates preserves normalized features."""
    original = make_window()

    np.testing.assert_allclose(
        window_features(original * 1.7), window_features(original), atol=1e-6
    )


def test_body_features_distinguish_forehead_and_chest_positions() -> None:
    """The same hand shape changes body-space features at forehead and chest height."""
    forehead = window_features(make_window(hand_height=0.26))
    chest = window_features(make_window(hand_height=0.62))

    assert not np.allclose(forehead[:, 63:105], chest[:, 63:105])
