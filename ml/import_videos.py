"""Convert timestamped sign videos into the shared raw-window CSV format."""

from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from app.storage import write_sample_rows
from config import RAW_FRAME_SIZE, SAMPLE_RATE_HZ, SEQUENCE_LENGTH, config

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi"}
WINDOW_STRIDE = 5
MAX_WINDOWS_PER_VIDEO = 8
MIN_VIDEO_DURATION_SECONDS = 0.5
MIN_SHOULDER_VISIBILITY = 0.5
TWO_HAND_SKIP_RATIO = 0.30
SHOULDER_WIDTH_OFFSET = 63 + 11 * 2
HAND_SCALE_LANDMARK_OFFSET = 9 * 3
SAFE_COMPONENT_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass(frozen=True)
class FrameLandmarks:
    """A valid raw frame and its normalized-space sanity metrics."""

    raw: list[float]
    two_hands: bool
    shoulder_width: float
    hand_scale: float


class LandmarkExtractor(Protocol):
    """Contract used by video processing and synthetic-video tests."""

    def process(self, rgb_frame: np.ndarray) -> tuple[FrameLandmarks | None, bool]:
        """Extract a valid raw frame and report if multiple hands were detected."""

    def close(self) -> None:
        """Release detector resources."""


class LegacyLandmarkExtractor:
    """Use legacy MediaPipe Hands and Pose to match browser landmarks."""

    def __init__(self, mediapipe: Any) -> None:
        self.hands = mediapipe.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.pose = mediapipe.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, rgb_frame: np.ndarray) -> tuple[FrameLandmarks | None, bool]:
        """Extract hand and pose points from an unmirrored RGB frame."""
        hand_result = self.hands.process(rgb_frame)
        pose_result = self.pose.process(rgb_frame)
        hand_sets = hand_result.multi_hand_landmarks or []
        if not hand_sets:
            return None, False
        two_hands = len(hand_sets) > 1
        scores = []
        for classification in hand_result.multi_handedness or []:
            scores.append(
                classification.classification[0].score
                if classification.classification
                else 0.0
            )
        hand_index = max(
            range(len(hand_sets)),
            key=lambda index: scores[index] if index < len(scores) else 0.0,
        )
        selected_hand = hand_sets[hand_index].landmark
        pose = pose_result.pose_landmarks.landmark if pose_result.pose_landmarks else None
        if not valid_landmarks(selected_hand, pose):
            return None, two_hands
        return make_frame_landmarks(selected_hand, pose, two_hands), two_hands

    def close(self) -> None:
        """Release the legacy Hands and Pose graph resources."""
        self.hands.close()
        self.pose.close()


class TasksLandmarkExtractor:
    """Fallback to MediaPipe Tasks when legacy solutions are unavailable."""

    def __init__(self, mediapipe: Any) -> None:
        hand_model = os.getenv("SILLIE_HAND_LANDMARKER_MODEL")
        pose_model = os.getenv("SILLIE_POSE_LANDMARKER_MODEL")
        if not hand_model or not pose_model:
            raise RuntimeError(
                "Legacy MediaPipe solutions are unavailable. Set "
                "SILLIE_HAND_LANDMARKER_MODEL and SILLIE_POSE_LANDMARKER_MODEL "
                "to local Tasks .task model files to use the fallback."
            )
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self.mediapipe = mediapipe
        self.hand_detector = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_model),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
        ))

    def process(self, rgb_frame: np.ndarray) -> tuple[FrameLandmarks | None, bool]:
        """Extract a valid raw frame from MediaPipe Tasks results."""
        image = self.mediapipe.Image(
            image_format=self.mediapipe.ImageFormat.SRGB,
            data=rgb_frame,
        )
        hands = self.hand_detector.detect(image)
        pose_result = self.pose_detector.detect(image)
        hand_sets = hands.hand_landmarks or []
        if not hand_sets:
            return None, False
        two_hands = len(hand_sets) > 1
        scores = [
            categories[0].score if categories else 0.0
            for categories in (hands.handedness or [])
        ]
        hand_index = max(
            range(len(hand_sets)),
            key=lambda index: scores[index] if index < len(scores) else 0.0,
        )
        selected_hand = hand_sets[hand_index]
        pose_sets = pose_result.pose_landmarks or []
        pose = pose_sets[0] if pose_sets else None
        if not valid_landmarks(selected_hand, pose):
            return None, two_hands
        return make_frame_landmarks(selected_hand, pose, two_hands), two_hands

    def close(self) -> None:
        """Release MediaPipe Tasks resources."""
        self.hand_detector.close()
        self.pose_detector.close()


def create_landmark_extractor() -> LandmarkExtractor:
    """Select legacy MediaPipe solutions, falling back to Tasks if unavailable."""
    import mediapipe as mp

    solutions = getattr(mp, "solutions", None)
    if solutions is not None and hasattr(solutions, "hands") and hasattr(solutions, "pose"):
        return LegacyLandmarkExtractor(mp)
    return TasksLandmarkExtractor(mp)


def valid_landmarks(hand: Any, pose: Any) -> bool:
    """Check landmark counts and required pose shoulder visibility."""
    if hand is None or len(hand) != 21 or pose is None or len(pose) != 33:
        return False
    return all(
        np.isfinite(getattr(pose[index], "visibility", None))
        and pose[index].visibility >= MIN_SHOULDER_VISIBILITY
        for index in (11, 12)
    )


def make_frame_landmarks(hand: Any, pose: Any, two_hands: bool) -> FrameLandmarks:
    """Flatten landmarks in the exact browser hand-XYZ then pose-XY order."""
    raw = [
        coordinate
        for point in hand
        for coordinate in (float(point.x), float(point.y), float(point.z))
    ]
    raw.extend(
        coordinate
        for point in pose
        for coordinate in (float(point.x), float(point.y))
    )
    if len(raw) != RAW_FRAME_SIZE or not np.isfinite(raw).all():
        raise ValueError(f"Expected {RAW_FRAME_SIZE} finite coordinates per frame.")
    shoulder_width = float(
        np.linalg.norm(
            np.asarray([pose[11].x, pose[11].y])
            - np.asarray([pose[12].x, pose[12].y])
        )
    )
    hand_scale = float(
        np.linalg.norm(
            np.asarray([hand[9].x, hand[9].y, hand[9].z])
            - np.asarray([hand[0].x, hand[0].y, hand[0].z])
        )
    )
    return FrameLandmarks(raw, two_hands, shoulder_width, hand_scale)


def resample_timestamped_frames(
    timestamped_frames: list[tuple[float, FrameLandmarks | None]],
    duration_seconds: float,
) -> list[FrameLandmarks | None]:
    """Sample the latest decoded result at each configured timestamp interval."""
    if not timestamped_frames or duration_seconds <= 0:
        return []
    tick_count = int(np.ceil(duration_seconds * SAMPLE_RATE_HZ))
    result: list[FrameLandmarks | None] = []
    frame_index = 0
    latest: FrameLandmarks | None = None
    for tick in range(tick_count):
        timestamp = tick / SAMPLE_RATE_HZ
        while (
            frame_index < len(timestamped_frames)
            and timestamped_frames[frame_index][0] <= timestamp
        ):
            latest = timestamped_frames[frame_index][1]
            frame_index += 1
        if tick == 0 and latest is None:
            latest = timestamped_frames[0][1]
            frame_index = max(frame_index, 1)
        result.append(latest)
    return result


def build_windows(
    sampled_frames: list[FrameLandmarks | None],
) -> list[list[list[float]]]:
    """Extract the active valid segment, pad short clips, and slide raw windows."""
    valid_indices = [
        index for index, frame in enumerate(sampled_frames) if frame is not None
    ]
    if not valid_indices:
        return []
    segment = sampled_frames[valid_indices[0]:valid_indices[-1] + 1]
    if len(segment) < SEQUENCE_LENGTH:
        missing = SEQUENCE_LENGTH - len(segment)
        left_pad = missing // 2
        right_pad = missing - left_pad
        first = next(frame for frame in segment if frame is not None)
        last = next(frame for frame in reversed(segment) if frame is not None)
        segment = [first] * left_pad + segment + [last] * right_pad

    windows: list[list[list[float]]] = []
    for start in range(0, len(segment) - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
        window_frames = segment[start:start + SEQUENCE_LENGTH]
        if any(frame is None for frame in window_frames):
            continue
        windows.append([frame.raw for frame in window_frames if frame is not None])
    return cap_windows(windows, MAX_WINDOWS_PER_VIDEO)


def cap_windows(
    windows: list[list[list[float]]], maximum: int
) -> list[list[list[float]]]:
    """Select up to maximum windows evenly across the clip."""
    if len(windows) <= maximum:
        return windows
    indices = np.linspace(0, len(windows) - 1, maximum, dtype=int)
    return [windows[index] for index in indices]


def process_video(
    video_path: Path,
    extractor: LandmarkExtractor,
) -> tuple[list[list[list[float]]], str | None, list[FrameLandmarks]]:
    """Decode and timestamp-resample one video into validated raw windows."""
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return [], "unreadable video", []
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    nominal_duration = frame_count / fps if fps > 0 else 0.0

    decoded: list[tuple[float, FrameLandmarks | None]] = []
    two_hand_frames = 0
    decoded_count = 0
    last_timestamp = -1.0
    while True:
        success, bgr_frame = capture.read()
        if not success:
            break
        decoded_count += 1
        timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if not np.isfinite(timestamp) or timestamp < 0 or timestamp <= last_timestamp:
            timestamp = decoded_count - 1
            timestamp = timestamp / fps if fps > 0 else max(last_timestamp, 0.0)
        last_timestamp = timestamp
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        try:
            landmark_frame, two_hands = extractor.process(rgb_frame)
        except Exception as error:
            capture.release()
            return [], f"landmark extraction failed: {error}", []
        two_hand_frames += int(two_hands)
        decoded.append((timestamp, landmark_frame))
    capture.release()

    if decoded_count == 0:
        return [], "unreadable video (no frames)", []
    observed_duration = last_timestamp + (1.0 / fps if fps > 0 else 0.0)
    duration = max(nominal_duration, observed_duration)
    if duration < MIN_VIDEO_DURATION_SECONDS:
        return [], f"shorter than {MIN_VIDEO_DURATION_SECONDS:.1f} seconds", []
    if two_hand_frames / decoded_count > TWO_HAND_SKIP_RATIO:
        return [], "two-handed video (>30% of frames contain two hands)", []

    sampled = resample_timestamped_frames(decoded, duration)
    valid_frames = [frame for frame in sampled if frame is not None]
    windows = build_windows(sampled)
    if not windows:
        return [], "no usable windows (missing hand or visible shoulders)", valid_frames
    return windows, None, valid_frames


def load_signers(source: Path) -> dict[str, str]:
    """Read optional signer identities keyed by relative path and filename."""
    signer_file = source / "signers.csv"
    if not signer_file.exists():
        return {}
    signers: dict[str, str] = {}
    with signer_file.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not {"file", "signer"}.issubset(reader.fieldnames or []):
            raise ValueError("signers.csv must contain columns named 'file' and 'signer'.")
        for row in reader:
            filename = (row.get("file") or "").strip().replace("\\", "/")
            signer = (row.get("signer") or "").strip()
            if filename and signer:
                signers[filename] = signer
                signers[Path(filename).name] = signer
    return signers


def safe_component(value: str) -> str:
    """Return an ASCII filename-safe component."""
    safe = SAFE_COMPONENT_PATTERN.sub("-", value).strip("-_")
    return safe or "unknown"


def session_name(
    dataset: str,
    signer: str | None,
    video_path: Path,
) -> str:
    """Build a dataset-prefixed session identifier within the storage constraints."""
    identity = signer or video_path.stem
    return safe_component(f"ds-{dataset}-{identity}")[:120]


def sanity_metrics(
    windows: list[list[list[float]]],
) -> tuple[float | None, float | None]:
    """Return median shoulder width and hand scale for a collection of windows."""
    shoulder_widths: list[float] = []
    hand_scales: list[float] = []
    for window in windows:
        for frame in window:
            if len(frame) != RAW_FRAME_SIZE:
                continue
            hand = np.asarray(frame[:63], dtype=float).reshape(21, 3)
            pose = np.asarray(frame[63:], dtype=float).reshape(33, 2)
            shoulder_widths.append(float(np.linalg.norm(pose[11] - pose[12])))
            hand_scales.append(float(np.linalg.norm(hand[9] - hand[0])))
    return (
        float(np.median(shoulder_widths)) if shoulder_widths else None,
        float(np.median(hand_scales)) if hand_scales else None,
    )


def compare_sanity_metrics(
    source: Path,
    imported_windows: dict[str, list[list[list[float]]]],
) -> None:
    """Compare normalized shoulder and hand scales with user-collected sessions."""
    from ml.train import collect_label_files, load_raw_window_file

    print("Sanity report (median normalized shoulder width / hand scale):")
    for label, dataset_windows in sorted(imported_windows.items()):
        own_windows: list[list[list[float]]] = []
        for csv_file, session in collect_label_files(config.data_dir).get(label, []):
            if session.startswith("ds-"):
                continue
            own_raw = load_raw_window_file(csv_file)
            if own_raw:
                own_windows.extend(window.tolist() for window in own_raw)
        ds_width, ds_hand = sanity_metrics(dataset_windows)
        own_width, own_hand = sanity_metrics(own_windows)
        if own_width is None or own_hand is None:
            print(f"  {label}: dataset {display_metrics(ds_width, ds_hand)}; no own data")
            continue
        print(
            f"  {label}: dataset {display_metrics(ds_width, ds_hand)}; "
            f"own {display_metrics(own_width, own_hand)}"
        )
        warn_metric_difference(label, "shoulder width", own_width, ds_width)
        warn_metric_difference(label, "hand scale", own_hand, ds_hand)


def display_metrics(width: float | None, hand: float | None) -> str:
    """Format optional metric values for the sanity report."""
    width_text = f"{width:.4f}" if width is not None else "n/a"
    hand_text = f"{hand:.4f}" if hand is not None else "n/a"
    return f"{width_text} / {hand_text}"


def warn_metric_difference(
    label: str, metric: str, own_value: float | None, dataset_value: float | None
) -> None:
    """Warn when either scale differs by more than thirty percent."""
    if not own_value or not dataset_value:
        return
    difference = abs(dataset_value - own_value) / own_value
    if difference > 0.30:
        print(
            f"WARNING: {label} dataset {metric} differs from own data by "
            f"{difference:.0%} (>30%)."
        )


def import_dataset(
    source: Path,
    labels: set[str] | None = None,
    max_per_label: int = 40,
    dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """Process video files, store session windows, and print label summaries."""
    if max_per_label < 1:
        raise ValueError("--max-per-label must be at least 1.")
    if not source.is_dir():
        raise ValueError(f"Dataset source directory does not exist: {source}")
    dataset_name = safe_component(source.name)
    signers = load_signers(source)
    extractor = create_landmark_extractor()
    results: dict[str, dict[str, Any]] = {}
    windows_by_session: dict[tuple[str, str], list[list[list[float]]]] = defaultdict(list)
    metrics_by_label: dict[str, list[list[list[float]]]] = defaultdict(list)
    try:
        for label_dir in sorted(path for path in source.iterdir() if path.is_dir()):
            label = label_dir.name
            if labels is not None and label not in labels:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
                print(f"Skipping invalid label directory '{label}'.")
                continue
            results[label] = {"used": 0, "skipped": [], "windows": 0}
            videos = sorted(
                path for path in label_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
            )
            accepted = 0
            for video_path in videos:
                if accepted >= max_per_label:
                    results[label]["skipped"].append(
                        (video_path.name, "max-per-label limit reached")
                    )
                    continue
                relative_path = video_path.relative_to(source).as_posix()
                signer = signers.get(relative_path) or signers.get(video_path.name)
                try:
                    windows, reason, valid_frames = process_video(video_path, extractor)
                except OSError as error:
                    windows, reason, valid_frames = [], f"read error: {error}", []
                if reason:
                    results[label]["skipped"].append((video_path.name, reason))
                    print(f"SKIP {label}/{video_path.name}: {reason}")
                    continue
                accepted += 1
                results[label]["used"] += 1
                results[label]["windows"] += len(windows)
                output_session = session_name(dataset_name, signer, video_path)
                windows_by_session[(label, output_session)].extend(windows)
                metrics_by_label[label].extend(
                    [[frame.raw] for frame in valid_frames]
                )
                print(
                    f"USE {label}/{video_path.name}: {len(windows)} windows "
                    f"-> {label}/{output_session}.csv"
                )
    finally:
        extractor.close()

    if not dry_run:
        for (label, session), windows in windows_by_session.items():
            rows = [
                np.asarray(window, dtype=np.float32).reshape(-1).tolist()
                for window in windows
            ]
            write_sample_rows(
                config.data_dir,
                label,
                session,
                rows,
                SEQUENCE_LENGTH * RAW_FRAME_SIZE,
            )

    for label, summary in sorted(results.items()):
        skipped_by_reason: dict[str, int] = {}
        for _, reason in summary["skipped"]:
            skipped_by_reason[reason] = skipped_by_reason.get(reason, 0) + 1
        skipped = (
            ", ".join(f"{reason}: {count}" for reason, count in skipped_by_reason.items())
            or "none"
        )
        action = "would write" if dry_run else "written"
        print(
            f"{label}: videos used={summary['used']}, "
            f"videos skipped={len(summary['skipped'])} ({skipped}), "
            f"windows {action}={summary['windows']}"
        )

    compare_sanity_metrics(source, dict(metrics_by_label))
    return results


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for a named video dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--labels",
        help="Comma-separated labels to import (default: every label directory).",
    )
    parser.add_argument("--max-per-label", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(arguments)
    args.labels = (
        {label.strip() for label in args.labels.split(",") if label.strip()}
        if args.labels
        else None
    )
    return args


def main(arguments: list[str] | None = None) -> None:
    """Run the video importer from the command line."""
    args = parse_args(arguments)
    try:
        import_dataset(args.source, args.labels, args.max_per_label, args.dry_run)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
