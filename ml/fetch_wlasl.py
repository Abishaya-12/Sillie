"""Fetch selected WLASL videos from the Voxel51 Hugging Face dataset."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from config import config

REPO_ID = "Voxel51/WLASL"
SAMPLES_FILENAME = "samples.json"
LABEL_COUNT_TARGET = 2000
LABEL_COUNT_MINIMUM = 1000
LABEL_COUNT_MAXIMUM = 3000
DOWNLOAD_RETRIES = 3
BACKOFF_SECONDS = 1
Sample = dict[str, Any]


def load_manifest(manifest_path: Path) -> tuple[Any, list[Sample]]:
    """Read a FiftyOne samples manifest and return its sample documents."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_samples = payload
    elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        raw_samples = payload["samples"]
    else:
        raise ValueError("samples.json must be a list or contain a 'samples' list.")
    if not all(isinstance(sample, dict) for sample in raw_samples):
        raise ValueError("Every item in samples.json must be a sample object.")
    return payload, raw_samples


def extract_label(value: Any) -> str | None:
    """Return a text label from a string or FiftyOne classification object."""
    if isinstance(value, str):
        label = value
    elif isinstance(value, dict) and isinstance(value.get("label"), str):
        label = value["label"]
    else:
        return None
    label = label.strip()
    return label or None


def detect_label_field(samples: list[Sample], override: str | None = None) -> str:
    """Find the string-like sample field with cardinality near 2,000 labels."""
    if override:
        if any(extract_label(sample.get(override)) is not None for sample in samples):
            return override
        raise ValueError(f"Label field {override!r} is missing or has no text labels.")

    values_by_field: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        for field, value in sample.items():
            label = extract_label(value)
            if label is not None:
                values_by_field[field].add(label.casefold())

    candidates = [
        (field, values)
        for field, values in values_by_field.items()
        if LABEL_COUNT_MINIMUM <= len(values) <= LABEL_COUNT_MAXIMUM
    ]
    if not candidates:
        cardinalities = sorted(
            ((field, len(values)) for field, values in values_by_field.items()),
            key=lambda item: abs(item[1] - LABEL_COUNT_TARGET),
        )[:8]
        raise ValueError(
            "Could not identify a label field with about 2,000 distinct values. "
            f"Closest fields: {cardinalities}. Use --label-field to override."
        )
    candidates.sort(key=lambda item: abs(len(item[1]) - LABEL_COUNT_TARGET))
    best_distance = abs(len(candidates[0][1]) - LABEL_COUNT_TARGET)
    best = [
        field
        for field, values in candidates
        if abs(len(values) - LABEL_COUNT_TARGET) == best_distance
    ]
    if len(best) != 1:
        raise ValueError(
            f"Label field detection is ambiguous ({', '.join(best)}); "
            "use --label-field to choose one."
        )
    return best[0]


def unique_labels(samples: list[Sample], label_field: str) -> list[str]:
    """Return sorted unique labels from a sample field."""
    labels = {
        label
        for sample in samples
        if (label := extract_label(sample.get(label_field))) is not None
    }
    return sorted(labels, key=str.casefold)


def match_samples(
    samples: list[Sample], words: list[str], label_field: str
) -> dict[str, list[Sample]]:
    """Match requested words to sample labels without case sensitivity."""
    matches = {word.casefold(): [] for word in words}
    for sample in samples:
        label = extract_label(sample.get(label_field))
        if label is not None and label.casefold() in matches:
            matches[label.casefold()].append(sample)
    return matches


def resolve_repo_filepath(filepath: str) -> str:
    """Normalize a FiftyOne filepath to a safe Hugging Face repo-relative path."""
    normalized = filepath.strip().replace("\\", "/")
    if normalized.startswith("file://"):
        normalized = normalized[7:]
    windows_absolute = re.match(r"^[A-Za-z]:/", normalized) is not None
    path = PurePosixPath(normalized)
    parts = [part for part in path.parts if part not in ("/", ".", "")]

    if path.is_absolute() or windows_absolute:
        data_index = next(
            (index for index, part in enumerate(parts) if part.casefold() == "data"),
            None,
        )
        if data_index is None:
            raise ValueError(
                f"Absolute media path {filepath!r} has no repo 'data' directory."
            )
        parts = parts[data_index:]

    if not parts or any(part == ".." for part in parts):
        raise ValueError(f"Unsafe or empty media filepath: {filepath!r}.")
    return PurePosixPath(*parts).as_posix()


def sample_size_bytes(sample: Sample) -> int | None:
    """Return media size metadata when the export provides it."""
    metadata = sample.get("metadata")
    if not isinstance(metadata, dict):
        return None
    size = metadata.get("size_bytes")
    if isinstance(size, int) and not isinstance(size, bool) and size >= 0:
        return size
    return None


def resolve_hf_token() -> str | bool:
    """Prefer HF_TOKEN, then cached credentials, and otherwise use anonymous access."""
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        print("Using the HF_TOKEN environment variable.")
        return env_token

    from huggingface_hub import get_token

    cached_token = get_token()
    if cached_token:
        print("Using the cached Hugging Face login.")
        return cached_token

    print(
        "Warning: no Hugging Face token found; continuing anonymously. "
        "Unauthenticated requests may be rate limited.",
        file=sys.stderr,
    )
    return False


def download_from_hub(
    filename: str,
    token: str | bool,
) -> str:
    """Download one file from the WLASL Hugging Face dataset repository."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        token=token,
    )


def is_transient_error(error: Exception) -> bool:
    """Return whether an error is likely to recover on a later request."""
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code in (408, 429) or 500 <= status_code <= 599
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    return error.__class__.__name__ in {
        "ConnectError",
        "ConnectTimeout",
        "NetworkError",
        "PoolTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "WriteError",
        "WriteTimeout",
    }


def download_with_retries(
    filename: str,
    token: str | bool,
    download_function: Callable[[str, str | bool], str] = download_from_hub,
) -> str:
    """Download one repo file with three retries for transient errors."""
    for attempt in range(DOWNLOAD_RETRIES + 1):
        try:
            return download_function(filename, token)
        except Exception as error:
            if attempt == DOWNLOAD_RETRIES or not is_transient_error(error):
                raise
            retry_number = attempt + 1
            delay = BACKOFF_SECONDS * (2 ** attempt)
            print(
                f"Transient error downloading {filename!r}: {error}. "
                f"Retrying in {delay}s ({retry_number}/{DOWNLOAD_RETRIES}).",
                file=sys.stderr,
            )
            time.sleep(delay)
    raise RuntimeError(f"Could not download {filename!r} after retries.")


def parse_words(value: str) -> list[str]:
    """Parse and validate comma-separated word labels for output directories."""
    words: list[str] = []
    seen: set[str] = set()
    for raw_word in value.split(","):
        word = raw_word.strip()
        if not word:
            continue
        if word in {".", ".."} or "/" in word or "\\" in word:
            raise ValueError(f"Invalid word label for an output directory: {word!r}.")
        normalized = word.casefold()
        if normalized not in seen:
            words.append(normalized)
            seen.add(normalized)
    if not words:
        raise ValueError("Provide at least one word in --words.")
    return words


def positive_int(value: str) -> int:
    """Parse a positive integer for --max-per-word."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the fetch command's argument parser."""
    parser = argparse.ArgumentParser(
        description="Fetch selected WLASL word videos from Hugging Face."
    )
    parser.add_argument("--words", required=True, help="Comma-separated WLASL words")
    parser.add_argument(
        "--max-per-word",
        type=positive_int,
        default=12,
        help="Maximum videos to fetch for each word (default: 12)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matching videos and known sizes without downloading media",
    )
    parser.add_argument(
        "--label-field",
        help="Override automatic detection of the sample word-label field",
    )
    return parser


def print_schema(payload: Any, samples: list[Sample]) -> None:
    """Print manifest and first-sample keys and filepath before inference."""
    top_level_keys = list(payload) if isinstance(payload, dict) else []
    first_sample = samples[0]
    print(f"Top-level keys: {top_level_keys}")
    print(f"First sample keys: {list(first_sample)}")
    print(f"First sample filepath: {first_sample.get('filepath')}")


def print_detected_label(samples: list[Sample], label_field: str) -> None:
    """Print the inferred label field and its value for the first sample."""
    first_sample = samples[0]
    print(f"Detected label field: {label_field}")
    print(f"First sample label: {extract_label(first_sample.get(label_field))}")


def closest_labels(word: str, labels: list[str]) -> list[str]:
    """Return up to five nearest labels for a missing requested word."""
    return difflib.get_close_matches(word, labels, n=5, cutoff=0.0)


def format_bytes(size: int) -> str:
    """Format a byte count as a compact human-readable size."""
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def print_summary(summary: dict[str, dict[str, int]]) -> None:
    """Print requested, downloaded, and failed counts in aligned columns."""
    print("\nSummary")
    print(f"{'Word':<20} {'Requested':>10} {'Downloaded':>12} {'Failed':>8}")
    print("-" * 54)
    for word, counts in summary.items():
        print(
            f"{word:<20} {counts['requested']:>10} "
            f"{counts['downloaded']:>12} {counts['failed']:>8}"
        )


def selected_samples(
    matches: dict[str, list[Sample]], maximum: int
) -> dict[str, list[Sample]]:
    """Apply the per-word cap while retaining requested word order."""
    return {word: samples[:maximum] for word, samples in matches.items()}


def run_fetch(args: argparse.Namespace) -> int:
    """Fetch the manifest, report its schema, and optionally retrieve media."""
    try:
        words = parse_words(args.words)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    token = resolve_hf_token()
    try:
        manifest_path = Path(download_from_hub(SAMPLES_FILENAME, token))
        payload, samples = load_manifest(manifest_path)
        if not samples:
            raise ValueError("samples.json contains no samples.")
        print_schema(payload, samples)
        label_field = detect_label_field(samples, args.label_field)
        print_detected_label(samples, label_field)
    except Exception as error:
        print(f"Error loading WLASL metadata: {error}", file=sys.stderr)
        return 1

    matches = match_samples(samples, words, label_field)
    labels = unique_labels(samples, label_field)
    for word, word_samples in matches.items():
        if not word_samples:
            suggestions = closest_labels(word, labels)
            print(
                f"No exact match for {word!r}. Closest labels: "
                f"{', '.join(suggestions) if suggestions else '(none)'}"
            )

    chosen = selected_samples(matches, args.max_per_word)
    summary = {
        word: {"requested": len(word_samples), "downloaded": 0, "failed": 0}
        for word, word_samples in chosen.items()
    }

    if args.dry_run:
        known_sizes: list[int] = []
        unknown_sizes = 0
        for word, word_samples in chosen.items():
            print(f"{word}: {len(word_samples)} videos would be fetched.")
            for sample in word_samples:
                size = sample_size_bytes(sample)
                if size is None:
                    unknown_sizes += 1
                else:
                    known_sizes.append(size)
        if known_sizes:
            print(
                f"Total known video size: {format_bytes(sum(known_sizes))}; "
                f"size unknown for {unknown_sizes} video(s)."
            )
        else:
            print("Total video size: unknown.")
        print_summary(summary)
        return 0

    destination_root = config.datasets_dir / "wlasl"
    for word, word_samples in chosen.items():
        for sample in word_samples:
            try:
                repo_filepath = resolve_repo_filepath(str(sample["filepath"]))
                output_path = destination_root / word / PurePosixPath(repo_filepath).name
                if output_path.is_file():
                    print(f"Skipping existing file: {output_path}")
                    continue
                cached_path = download_with_retries(repo_filepath, token)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached_path, output_path)
                summary[word]["downloaded"] += 1
                print(f"Downloaded {repo_filepath} -> {output_path}")
            except Exception as error:
                summary[word]["failed"] += 1
                print(
                    f"Failed {sample.get('filepath')!r} for {word!r}: {error}",
                    file=sys.stderr,
                )

    print_summary(summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse command-line options and run the selected WLASL download."""
    return run_fetch(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())