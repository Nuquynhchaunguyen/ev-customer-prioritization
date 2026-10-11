"""Load a trusted local pipeline and rank customer profiles reproducibly."""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
    SCHEMA_VERSION,
    ProfileValidationError,
    read_profile_csv,
    validate_profiles,
)

TIE_BREAK_SEED = 42


def _validate_bundle(bundle: dict) -> dict:
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("The saved model must contain a pipeline and metadata dictionary.")
    metadata = bundle.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("The saved model metadata must be a dictionary.")
    version = bundle.get("schema_version", metadata.get("schema_version"))
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported model schema version {version!r}; expected {SCHEMA_VERSION!r}.")
    pipeline = bundle["pipeline"]
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ValueError("The saved pipeline does not support probability prediction.")
    declared_features = metadata.get("feature_columns", bundle.get("feature_columns"))
    if declared_features is not None and list(declared_features) != FEATURE_COLUMNS:
        raise ValueError("The saved model feature schema does not match this application.")
    if hasattr(pipeline, "feature_names_in_") and list(pipeline.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("The pipeline input columns do not match this application's feature schema.")
    return bundle


def load_bundle(path: str | Path) -> dict:
    """Load only a project-owned local artifact, never an uploaded model file.

    Joblib can execute Python while loading; this function belongs behind the
    application's fixed model path rather than a user-supplied upload widget.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model not found: {path}")
    return _validate_bundle(joblib.load(path))


def rank_scores(ids, scores) -> pd.DataFrame:
    """Descending probabilities with stable, seeded ID hashes resolving ties.

    The tie rule does not depend on upload order or numeric ID order. It does
    not constitute evidence that tied customers differ in estimated interest.
    """
    ids = pd.Series(ids, dtype="string").reset_index(drop=True)
    try:
        probabilities = np.asarray(scores, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Scores must be numeric probabilities.") from error
    if probabilities.ndim != 1 or len(ids) != len(probabilities):
        raise ValueError("Provide one probability for each customer ID.")
    if len(ids) == 0:
        raise ProfileValidationError("The customer table is empty.")
    ids = ids.str.strip()
    if ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ProfileValidationError("Ranked customer IDs must be nonblank and unique.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("The model returned invalid probabilities; each score must be finite and between 0 and 1.")
    ties = [hashlib.sha256(f"{TIE_BREAK_SEED}:{profile_id}".encode("utf-8")).hexdigest() for profile_id in ids]
    ranked = pd.DataFrame({ID_COLUMN: ids, "yes_probability": probabilities, "_tie_break": ties})
    ranked = ranked.sort_values(["yes_probability", "_tie_break", ID_COLUMN], ascending=[False, True, True], kind="mergesort")
    ranked = ranked.reset_index(drop=True).drop(columns="_tie_break")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1, dtype=np.int64))
    return ranked


def _range_warnings(profiles: pd.DataFrame, metadata: dict) -> list[str]:
    ranges = metadata.get("training_ranges", metadata.get("observed_train_ranges", {}))
    warnings = []
    if not isinstance(ranges, dict):
        return warnings
    for column, limits in ranges.items():
        if column not in FEATURE_COLUMNS:
            continue
        if isinstance(limits, dict):
            minimum, maximum = limits.get("min"), limits.get("max")
        elif isinstance(limits, (list, tuple)) and len(limits) == 2:
            minimum, maximum = limits
        else:
            continue
        if minimum is None or maximum is None:
            continue
        values = pd.to_numeric(profiles[column], errors="coerce")
        if values.isna().any():  # Nominal categories have no meaningful numeric range.
            continue
        outside = (values < minimum) | (values > maximum)
        if outside.any():
            warnings.append(
                f"{column}: {int(outside.sum())} profile(s) outside the model's observed training range "
                f"[{minimum:g}, {maximum:g}]. Values were retained without clipping; estimates here may be less reliable."
            )
    return warnings


def score_profiles(frame: pd.DataFrame, bundle: dict, k: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Validate, predict from features alone, and return a ranking and top K."""
    bundle = _validate_bundle(bundle)
    profiles, warnings = validate_profiles(frame)
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(profiles):
        raise ProfileValidationError(f"Contact capacity K must be a whole number between 1 and {len(profiles)}.")
    pipeline = bundle["pipeline"]
    classes = np.asarray(getattr(pipeline, "classes_", []))
    matches = np.flatnonzero(classes == 1)
    if classes.ndim != 1 or len(matches) != 1:
        raise ValueError("The saved pipeline must contain the positive class 1 (the supplied Yes label).")
    probabilities = np.asarray(pipeline.predict_proba(profiles.loc[:, FEATURE_COLUMNS]))
    if probabilities.ndim != 2 or probabilities.shape != (len(profiles), len(classes)):
        raise ValueError("The pipeline returned a probability table with an unexpected shape.")
    ranked = rank_scores(profiles[ID_COLUMN], probabilities[:, int(matches[0])])
    warnings.extend(_range_warnings(profiles, bundle.get("metadata", {})))
    return ranked, ranked.head(int(k)).copy(), warnings
