from pathlib import Path

import pytest

from fraud_ml.data import DataValidationError, load_data, sha256_file, validate_frame
from tests.conftest import make_synthetic_frame

URL = "https://example.invalid/unused.csv"


def test_load_data_validates_contract(synthetic_csv: Path):
    frame, digest = load_data(synthetic_csv, URL, download_missing=False)
    assert frame["Class"].dtype == "int8"
    assert digest == sha256_file(synthetic_csv)


def test_pinned_checksum_must_match(synthetic_csv: Path):
    load_data(synthetic_csv, URL, sha256=sha256_file(synthetic_csv), download_missing=False)
    with pytest.raises(DataValidationError, match="checksum mismatch"):
        load_data(synthetic_csv, URL, sha256="0" * 64, download_missing=False)


def test_missing_file_without_download_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "absent.csv", URL, download_missing=False)


def test_rejects_missing_columns():
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_frame(make_synthetic_frame(20).drop(columns=["V3"]))


def test_rejects_bad_target_values():
    frame = make_synthetic_frame(20)
    frame.loc[0, "Class"] = 2
    with pytest.raises(DataValidationError, match="only 0 and 1"):
        validate_frame(frame)


def test_rejects_negative_amount():
    frame = make_synthetic_frame(20)
    frame.loc[0, "Amount"] = -1.0
    with pytest.raises(DataValidationError, match="non-negative"):
        validate_frame(frame)


def test_drops_unexpected_columns():
    frame = make_synthetic_frame(20).assign(debug_column=1)
    assert "debug_column" not in validate_frame(frame).columns
