"""Tests for timestamped video import and dataset-only training splits."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from config import RAW_FRAME_SIZE, SEQUENCE_LENGTH
from ml import import_videos
from ml.train import split_dataset


def landmark_frame(value: float) -> import_videos.FrameLandmarks:
    """Build a distinguishable synthetic frame with the raw frame width."""
    return import_videos.FrameLandmarks(
        raw=[value] * RAW_FRAME_SIZE,
        two_hands=False,
        shoulder_width=0.2,
        hand_scale=0.1,
    )


def test_timestamp_resampling_padding_and_window_stride() -> None:
    """Resampling follows timestamps; short clips pad and long clips slide."""
    first, second, third = (landmark_frame(value) for value in (1, 2, 3))
    sampled = import_videos.resample_timestamped_frames(
        [(0.0, first), (0.02, second), (0.06, third)],
        duration_seconds=0.1,
    )
    assert sampled == [first, second, third]

    padded = import_videos.build_windows([first, second, third])
    assert len(padded) == 1
    assert len(padded[0]) == SEQUENCE_LENGTH
    assert padded[0][0] == first.raw
    assert padded[0][-1] == third.raw

    frames = [landmark_frame(float(index)) for index in range(SEQUENCE_LENGTH + 5)]
    windows = import_videos.build_windows(frames)
    assert len(windows) == 2
    assert windows[0][0] == frames[0].raw
    assert windows[1][0] == frames[5].raw


def test_synthetic_video_import_writes_dataset_session_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tiny video processed with a fake extractor writes valid raw-window CSV."""
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "sample-dataset"
    label_dir = source / "hello"
    label_dir.mkdir(parents=True)
    video_path = label_dir / "clip.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        30.0,
        (64, 64),
    )
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV MJPG video encoding is unavailable.")
    for index in range(40):
        frame = np.full((64, 64, 3), index, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    class FakeExtractor:
        """Return valid deterministic landmarks without invoking MediaPipe."""

        def __init__(self) -> None:
            self.frame_number = 0

        def process(
            self, rgb_frame: np.ndarray
        ) -> tuple[import_videos.FrameLandmarks, bool]:
            del rgb_frame
            self.frame_number += 1
            return landmark_frame(float(self.frame_number)), False

        def close(self) -> None:
            return None

    monkeypatch.setattr(import_videos, "create_landmark_extractor", FakeExtractor)
    monkeypatch.setattr(
        import_videos.config,
        "data_dir",
        tmp_path / "training-data",
    )
    results = import_videos.import_dataset(source, labels={"hello"})

    output = tmp_path / "training-data" / "hello" / "ds-sample-dataset-clip.csv"
    assert results["hello"]["used"] == 1
    assert results["hello"]["windows"] >= 1
    assert output.is_file()
    with output.open("r", newline="", encoding="utf-8") as sample_file:
        rows = list(csv.reader(sample_file))
    assert len(rows) == results["hello"]["windows"]
    assert all(len(row) == SEQUENCE_LENGTH * RAW_FRAME_SIZE for row in rows)


def test_training_split_keeps_dataset_sessions_out_of_validation_and_test() -> None:
    """Dataset-origin windows appear only in train while own sessions are held out."""
    groups: list[str] = []
    class_ids: list[int] = []
    raw_values: list[int] = []
    for class_id in range(2):
        sessions = [
            f"label-{class_id}:own-a",
            f"label-{class_id}:own-b",
            f"label-{class_id}:own-c",
            f"label-{class_id}:ds-data-signer",
        ]
        for session_index, session in enumerate(sessions):
            for window_index in range(2):
                groups.append(session)
                class_ids.append(class_id)
                raw_values.append(class_id * 100 + session_index * 10 + window_index)

    windows = np.asarray(raw_values, dtype=np.float32).reshape(-1, 1, 1)
    class_array = np.asarray(class_ids, dtype=np.int64)
    group_array = np.asarray(groups, dtype=str)
    train, validation, test, *_ = split_dataset(
        windows,
        class_array,
        group_array,
    )

    source_by_value = dict(zip(raw_values, groups))
    train_sources = {source_by_value[int(value)] for value in train[:, 0, 0]}
    validation_sources = {
        source_by_value[int(value)] for value in validation[:, 0, 0]
    }
    test_sources = {source_by_value[int(value)] for value in test[:, 0, 0]}
    assert any(session.rsplit(":", 1)[-1].startswith("ds-") for session in train_sources)
    assert not any(
        session.rsplit(":", 1)[-1].startswith("ds-")
        for session in validation_sources | test_sources
    )
