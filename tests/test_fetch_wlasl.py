"""Offline tests for WLASL manifest matching and path handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.fetch_wlasl import (
    detect_label_field,
    download_with_retries,
    load_manifest,
    match_samples,
    resolve_repo_filepath,
)


def test_manifest_matching_is_case_insensitive(tmp_path: Path) -> None:
    """A small FiftyOne export can be read and selected without network access."""
    manifest_path = tmp_path / "samples.json"
    manifest_path.write_text(
        json.dumps({
            "media_type": "video",
            "samples": [
                {
                    "filepath": "data/yes-1.mp4",
                    "ground_truth": {"label": "Yes"},
                },
                {
                    "filepath": "data/hello-1.mp4",
                    "ground_truth": "hello",
                },
                {
                    "filepath": "data/no-1.mp4",
                    "ground_truth": {"label": "no"},
                },
            ],
        }),
        encoding="utf-8",
    )

    payload, samples = load_manifest(manifest_path)
    label_field = detect_label_field(samples, "ground_truth")
    matches = match_samples(samples, ["YES", "Hello", "unknown"], label_field)

    assert list(payload) == ["media_type", "samples"]
    assert [sample["filepath"] for sample in matches["yes"]] == [
        "data/yes-1.mp4"
    ]
    assert [sample["filepath"] for sample in matches["hello"]] == [
        "data/hello-1.mp4"
    ]
    assert matches["unknown"] == []


def test_label_field_detection_uses_about_two_thousand_distinct_values() -> None:
    """Automatic selection favors the field with the WLASL class cardinality."""
    samples = [
        {
            "filepath": f"data/{index}.mp4",
            "label": {"label": f"word-{index % 2000}"},
        }
        for index in range(2200)
    ]

    assert detect_label_field(samples) == "label"


def test_repo_filepath_resolution_handles_exported_absolute_and_relative_paths() -> None:
    """FiftyOne local paths resolve to the repository's data-relative filenames."""
    assert resolve_repo_filepath("./data/hello/clip.mp4") == "data/hello/clip.mp4"
    assert resolve_repo_filepath(
        r"C:\cache\Voxel51\WLASL\data\hello\clip.mp4"
    ) == "data/hello/clip.mp4"


def test_transient_download_errors_retry_three_times(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient errors receive three exponential-backoff retries."""
    from ml import fetch_wlasl

    attempts = 0
    delays: list[int] = []

    def fake_download(filename: str, token: str | bool) -> str:
        nonlocal attempts
        del filename, token
        attempts += 1
        if attempts <= 3:
            raise ConnectionError("temporary network failure")
        return "cached-video.mp4"

    monkeypatch.setattr(fetch_wlasl.time, "sleep", delays.append)

    result = download_with_retries("data/clip.mp4", False, fake_download)

    assert result == "cached-video.mp4"
    assert attempts == 4
    assert delays == [1, 2, 4]
