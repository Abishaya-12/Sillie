"""Shared validation and CSV storage for raw training windows."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def normalize_sample_rows(
    samples: object, expected_length: int
) -> list[list[float]] | None:
    """Validate sample rows and convert their numeric values to finite floats."""
    if not isinstance(samples, list):
        return None

    rows: list[list[float]] = []
    for sample in samples:
        if not isinstance(sample, list) or len(sample) != expected_length:
            return None
        if any(not is_finite_number(value) for value in sample):
            return None
        rows.append([float(value) for value in sample])
    return rows


def write_sample_rows(
    data_dir: Path,
    label: str,
    session: str,
    rows: object,
    expected_length: int,
    *,
    overwrite: bool = False,
) -> int:
    """Validate and store raw rows in the same per-label session CSV format."""
    if not LABEL_PATTERN.fullmatch(label) or not LABEL_PATTERN.fullmatch(session):
        raise ValueError("Label and session must contain letters, digits, '_' or '-'.")
    normalized_rows = normalize_sample_rows(rows, expected_length)
    if normalized_rows is None:
        raise ValueError(
            f"Rows must be lists of exactly {expected_length} finite numbers."
        )

    label_dir = data_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    sample_file = label_dir / f"{session}.csv"
    mode = "w" if overwrite else "a"
    with sample_file.open(mode, newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerows(normalized_rows)
    return len(normalized_rows)


def is_finite_number(value: object) -> bool:
    """Return whether a non-boolean numeric value is finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
