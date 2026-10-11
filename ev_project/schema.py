"""Shared input contract for the synthetic EV-interest prototype.

Observed training ranges are descriptive, not customer eligibility rules.
Environmental concern is a category because its survey anchors are unknown.
"""

from __future__ import annotations

import csv
import io
from collections import Counter

import numpy as np
import pandas as pd

SCHEMA_VERSION = "1.0"
ID_COLUMN = "id"
TARGET_COLUMN = "Will_Buy_EV"
NUMERIC_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
]
CATEGORICAL_FEATURES = [
    "Environmental_Concern_Level",
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS
INTEGER_FEATURES = [column for column in NUMERIC_FEATURES if column != "Daily_Commute_km"]
KNOWN_CATEGORIES = {
    "Environmental_Concern_Level": ("1", "2", "3", "4", "5"),
    "Gender": ("Female", "Male", "Other"),
    "City_Type": ("Rural", "Suburban", "Urban"),
    "Current_Car_Type": ("Hatchback", "SUV", "Sedan", "Truck"),
    "Home_Charging_Possible": ("No", "Yes"),
    "Subsidy_Available": ("No", "Yes"),
    "Range_Anxiety_Level": ("High", "Low", "Medium"),
}


class ProfileValidationError(ValueError):
    """An uploaded table cannot be safely scored under the input contract."""


def _normalized_headers(columns) -> list[str]:
    if any(not isinstance(column, str) for column in columns):
        raise ProfileValidationError("CSV column names must be text.")
    names = [column.strip().lstrip("\ufeff") for column in columns]
    if any(not name for name in names):
        raise ProfileValidationError("The CSV contains a blank column name.")
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        raise ProfileValidationError("Duplicate CSV column names: " + ", ".join(duplicates))
    return names


def read_profile_csv(content: bytes) -> pd.DataFrame:
    """Read UTF-8 CSV without losing ID leading zeros or concealing bad headers.

    The CSV reader checks row widths before pandas can silently repair rows.
    Values stay as text until the explicit schema validator converts features.
    """
    if not isinstance(content, (bytes, bytearray)):
        raise ProfileValidationError("Provide CSV file bytes.")
    if not content or not content.strip():
        raise ProfileValidationError("The CSV file is empty.")
    try:
        text = bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ProfileValidationError("Save the CSV using UTF-8 encoding and upload it again.") from error
    if "\x00" in text:
        raise ProfileValidationError("The uploaded file contains null bytes and is not a supported CSV.")
    try:
        reader = csv.reader(io.StringIO(text, newline=""), strict=True)
        headers = next(reader, None)
        if not headers:
            raise ProfileValidationError("The CSV file has no header row.")
        normalized_headers = _normalized_headers(headers)
        rows = []
        for row in reader:
            if not row:  # Empty lines between records do not represent customers.
                continue
            if len(row) != len(headers):
                raise ProfileValidationError(
                    f"CSV row ending at line {reader.line_num} has {len(row)} fields; "
                    f"the header has {len(headers)}. Check commas and quotation marks."
                )
            rows.append(row)
    except csv.Error as error:
        raise ProfileValidationError(f"Malformed CSV: {error}") from error
    if not rows:
        raise ProfileValidationError("The CSV contains headers but no customer rows.")
    frame = pd.DataFrame(rows, columns=normalized_headers)
    if ID_COLUMN in frame:
        frame[ID_COLUMN] = frame[ID_COLUMN].astype("string")
    return frame


def _bad_rows(mask: pd.Series | np.ndarray) -> str:
    positions = np.flatnonzero(np.asarray(mask)) + 1
    shown = ", ".join(str(position) for position in positions[:5])
    return shown + (" ..." if len(positions) > 5 else "")


def validate_profiles(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a normalized ID-plus-feature table and visible cleaning warnings.

    Numeric fields must be finite and nonnegative; age must be at most 120.
    These broad validity checks deliberately do not enforce observed train
    minima or maxima. Integer fields retain exact whole-number semantics.
    Extra columns (including outcomes) never enter the saved pipeline.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ProfileValidationError("Customer profiles must be provided as a table.")
    if frame.empty:
        raise ProfileValidationError("The customer table is empty.")
    headers = _normalized_headers(frame.columns)
    normalized = frame.copy()
    normalized.columns = headers
    warnings: list[str] = []
    if headers != list(frame.columns):
        warnings.append("Trimmed surrounding whitespace from column names.")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in headers]
    if missing_columns:
        raise ProfileValidationError("Missing required columns: " + ", ".join(missing_columns))
    extras = [column for column in headers if column not in REQUIRED_COLUMNS]
    if extras:
        warnings.append("Ignored extra columns before prediction: " + ", ".join(extras) + ".")
    normalized = normalized.loc[:, REQUIRED_COLUMNS].copy()

    ids = normalized[ID_COLUMN]
    if ids.isna().any():
        raise ProfileValidationError("IDs are missing at customer rows " + _bad_rows(ids.isna()) + ".")
    ids = ids.astype("string").str.strip()
    blank_ids = ids.eq("")
    if blank_ids.any():
        raise ProfileValidationError("IDs are blank at customer rows " + _bad_rows(blank_ids) + ".")
    if ids.duplicated(keep=False).any():
        raise ProfileValidationError("Customer IDs must be unique; duplicate IDs were found.")
    normalized[ID_COLUMN] = ids

    for column in NUMERIC_FEATURES:
        values = normalized[column]
        if values.map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ProfileValidationError(f"{column} must contain numbers, not True/False values.")
        converted = pd.to_numeric(values, errors="coerce")
        finite = np.isfinite(converted.to_numpy(dtype=float, na_value=np.nan))
        if not finite.all():
            raise ProfileValidationError(
                f"{column} has missing, nonnumeric, or infinite values at customer rows {_bad_rows(~finite)}."
            )
        if (converted < 0).any():
            raise ProfileValidationError(f"{column} cannot contain negative values.")
        if column == "Age" and (converted > 120).any():
            raise ProfileValidationError("Age must be between 0 and 120 years for this input form.")
        if column in INTEGER_FEATURES:
            if (converted % 1 != 0).any():
                raise ProfileValidationError(f"{column} must contain whole numbers.")
            # Values above this limit cannot be cast safely to signed int64.
            if (converted >= 2**63).any():
                raise ProfileValidationError(f"{column} is too large to represent safely as a whole number.")
            normalized[column] = converted.astype("int64")
        else:
            normalized[column] = converted.astype("float64")

    concern = normalized["Environmental_Concern_Level"]
    if concern.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ProfileValidationError("Environmental_Concern_Level must use codes 1, 2, 3, 4, or 5.")
    concern_numbers = pd.to_numeric(concern, errors="coerce")
    valid_concern = concern_numbers.isin([1, 2, 3, 4, 5])
    if not valid_concern.all():
        raise ProfileValidationError("Environmental_Concern_Level must use codes 1, 2, 3, 4, or 5.")
    normalized["Environmental_Concern_Level"] = concern_numbers.astype("int64").astype(str)

    trimmed_columns = []
    for column in CATEGORICAL_FEATURES:
        if column == "Environmental_Concern_Level":
            continue
        values = normalized[column]
        if values.isna().any():
            raise ProfileValidationError(f"{column} contains missing values.")
        text_values = values.astype("string")
        trimmed = text_values.str.strip()
        if not text_values.equals(trimmed):
            trimmed_columns.append(column)
        invalid = ~trimmed.isin(KNOWN_CATEGORIES[column])
        if invalid.any():
            choices = ", ".join(KNOWN_CATEGORIES[column])
            raise ProfileValidationError(f"{column} contains unknown or blank categories. Accepted values: {choices}.")
        # Plain strings keep sklearn's one-hot encoder compatible across pandas versions.
        normalized[column] = trimmed.astype(object)
    if trimmed_columns:
        warnings.append("Trimmed surrounding whitespace in: " + ", ".join(trimmed_columns) + ".")
    return normalized, warnings
