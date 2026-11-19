from pathlib import Path

import pytest

from churn_ml.data import DataValidationError, load_data, sha256_file, validate_frame
from tests.conftest import make_synthetic_frame

URL = "https://example.invalid/unused.csv"


def test_load_data_validates_contract(synthetic_csv: Path):
    frame, digest = load_data(synthetic_csv, URL, download_missing=False)
    assert set(frame["Churn"].unique()) == {0, 1}
    assert frame["TotalCharges"].dtype.kind == "f"  # text with blanks became numeric + NaN
    assert digest == sha256_file(synthetic_csv)


def test_blank_total_charges_become_missing():
    raw = make_synthetic_frame(50)
    clean = validate_frame(raw)
    assert clean["TotalCharges"].isna().sum() == (raw["TotalCharges"] == " ").sum()


def test_pinned_checksum_must_match(synthetic_csv: Path):
    load_data(synthetic_csv, URL, sha256=sha256_file(synthetic_csv), download_missing=False)
    with pytest.raises(DataValidationError, match="checksum mismatch"):
        load_data(synthetic_csv, URL, sha256="0" * 64, download_missing=False)


def test_missing_file_without_download_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "absent.csv", URL, download_missing=False)


def test_rejects_missing_columns():
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_frame(make_synthetic_frame(20).drop(columns=["Contract"]))


def test_rejects_duplicate_customer_ids():
    frame = make_synthetic_frame(20)
    frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    with pytest.raises(DataValidationError, match="unique"):
        validate_frame(frame)


def test_rejects_bad_target_values():
    frame = make_synthetic_frame(20)
    frame.loc[0, "Churn"] = "Maybe"
    with pytest.raises(DataValidationError, match="'Yes' and 'No'"):
        validate_frame(frame)


def test_rejects_negative_tenure():
    frame = make_synthetic_frame(20)
    frame.loc[0, "tenure"] = -3
    with pytest.raises(DataValidationError, match="non-negative"):
        validate_frame(frame)
