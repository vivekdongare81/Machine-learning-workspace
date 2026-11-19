"""Dataset acquisition and data-contract validation.

The raw file is downloaded byte-for-byte and its SHA-256 compared with the value
pinned in config. That pins the dataset version: if the upstream file changes,
training stops with a clear error instead of quietly training on different data.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TARGET = "Class"
FEATURE_COLUMNS: list[str] = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
EXPECTED_COLUMNS: list[str] = [*FEATURE_COLUMNS, TARGET]


class DataValidationError(ValueError):
    """The data does not meet the expected contract."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, timeout: float = 120) -> None:
    """Download to a temp file, then rename, so a failed download never leaves a partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (
        urllib.request.urlopen(url, timeout=timeout) as response,
        tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp,
    ):
        shutil.copyfileobj(response, tmp)
    Path(tmp.name).replace(dest)


def ensure_dataset(
    path: str | Path, url: str, sha256: str | None = None, download_missing: bool = True
) -> str:
    """Make sure the raw file exists and matches the pinned checksum. Returns its SHA-256."""
    path = Path(path)
    if not path.exists():
        if not download_missing:
            raise FileNotFoundError(path)
        logger.info("downloading dataset", extra={"url": url, "path": str(path)})
        download(url, path)
    digest = sha256_file(path)
    if sha256 is None:
        logger.warning(
            "no sha256 pinned in config; dataset version is not verified", extra={"sha256": digest}
        )
    elif digest != sha256.lower():
        raise DataValidationError(
            f"checksum mismatch for {path}: expected {sha256}, got {digest}. "
            "The data changed; review it, then update data.sha256 deliberately."
        )
    return digest


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Enforce the data contract and return only the expected columns."""
    missing = [c for c in EXPECTED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(f"missing columns: {missing}")
    if frame.empty:
        raise DataValidationError("dataset is empty")
    non_numeric = [c for c in EXPECTED_COLUMNS if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise DataValidationError(f"columns must be numeric: {non_numeric}")
    if frame[TARGET].isna().any():
        raise DataValidationError(f"target {TARGET!r} contains missing values")
    if not set(frame[TARGET].unique()) <= {0, 1}:
        raise DataValidationError(f"target {TARGET!r} must contain only 0 and 1")
    for column in ("Time", "Amount"):
        if (frame[column] < 0).any():
            raise DataValidationError(f"{column!r} must be non-negative")

    extra = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
    if extra:
        logger.warning("dropping unexpected columns", extra={"columns": extra})
    clean = frame[EXPECTED_COLUMNS].copy()
    clean[TARGET] = clean[TARGET].astype("int8")
    return clean


def load_data(
    path: str | Path, url: str, sha256: str | None = None, download_missing: bool = True
) -> tuple[pd.DataFrame, str]:
    """Return the validated dataset and the SHA-256 of the raw file it came from."""
    digest = ensure_dataset(path, url, sha256, download_missing)
    frame = validate_frame(pd.read_csv(path))
    logger.info(
        "dataset loaded",
        extra={
            "rows": len(frame),
            "fraud_rate": round(float(frame[TARGET].mean()), 6),
            "duplicate_rows": int(frame.duplicated().sum()),
        },
    )
    return frame, digest
