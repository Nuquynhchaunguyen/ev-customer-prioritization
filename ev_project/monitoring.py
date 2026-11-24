"""Monitor validated batches against a model-linked training reference.

Only aggregate histograms are persisted. PSI flags input distribution changes;
it does not measure prediction accuracy, concept drift, fairness, or sales.
Run ``python -m ev_project.monitoring --demo`` for a reproducible local example.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, ID_COLUMN, KNOWN_CATEGORIES,
    NUMERIC_FEATURES, REQUIRED_COLUMNS, SCHEMA_VERSION, TARGET_COLUMN, read_profile_csv,
    validate_profiles,
)

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_VERSION = "1.0"
POLICY = {
    "minimum_batch_size": 200,
    "review_psi": 0.1,
    "investigate_psi": 0.2,
    "smoothing_probability": 1e-6,
    "numeric_quantile_bins": 10,
    "calibration": "Prototype investigation settings; not calibrated error guarantees or significance tests.",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_digest(reference: dict) -> str:
    payload = {key: value for key, value in reference.items() if key != "reference_id"}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _histogram(values: np.ndarray, cut_points: list[float]) -> list[int]:
    counts, _ = np.histogram(values, bins=[-np.inf, *cut_points, np.inf])
    return [int(value) for value in counts]


def create_reference(frame: pd.DataFrame, *, model_path: str | Path,
                     source: dict | None = None) -> dict:
    """Create aggregate bins; the caller must supply training rows only.

    The public rebuild command enforces saved training split membership and
    model source hashes. This small helper also supports isolated unit fixtures.
    """
    profiles, _ = validate_profiles(frame)
    features: dict[str, dict] = {}
    for column in NUMERIC_FEATURES:
        values = profiles[column].to_numpy(dtype=float)
        low, high = float(values.min()), float(values.max())
        quantiles = np.quantile(values, np.linspace(0, 1, 11)[1:-1])
        # Separate tails retain observations outside the complete train range.
        cuts = np.unique(np.r_[low, quantiles, np.nextafter(high, np.inf)])
        cuts = [float(value) for value in cuts if np.isfinite(value)]
        features[column] = {
            "type": "numeric", "minimum": low, "maximum": high,
            "cut_points": cuts, "counts": _histogram(values, cuts),
        }
    for column in CATEGORICAL_FEATURES:
        categories = list(KNOWN_CATEGORIES[column])
        counts = profiles[column].value_counts()
        features[column] = {
            "type": "categorical", "categories": categories,
            "counts": [int(counts.get(value, 0)) for value in categories],
        }
    reference = {
        "reference_version": REFERENCE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha256_file(model_path),
        "reference_rows": len(profiles),
        "source": source or {"selection_scope": "caller_supplied_training_rows"},
        "feature_columns": list(FEATURE_COLUMNS),
        "policy": dict(POLICY), "features": features,
    }
    reference["reference_id"] = _reference_digest(reference)
    return validate_reference(reference, model_path=model_path)


def validate_reference(reference: dict, *, model_path: str | Path) -> dict:
    """Reject stale, incomplete, or accidentally edited reference artifacts.

    The digest is an integrity check, not a signature or an access control.
    Model replacement requires an explicit reference rebuild and review.
    """
    try:
        if not isinstance(reference, dict):
            raise ValueError("Expected a monitoring reference dictionary.")
        if reference["reference_version"] != REFERENCE_VERSION:
            raise ValueError("Unsupported monitoring reference version.")
        if reference["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Monitoring reference schema does not match the application.")
        if reference["feature_columns"] != list(FEATURE_COLUMNS):
            raise ValueError("Monitoring reference feature schema does not match.")
        if reference["model_sha256"] != sha256_file(model_path):
            raise ValueError("Monitoring reference model SHA256 does not match the saved model; rebuild the reference.")
        if reference["policy"] != POLICY:
            raise ValueError("Unsupported monitoring policy; review and rebuild the reference.")
        rows = reference["reference_rows"]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
            raise ValueError("Monitoring reference row count must be positive.")
        if not isinstance(reference["source"], dict):
            raise ValueError("Monitoring source provenance is missing.")
        features = reference["features"]
        if not isinstance(features, dict) or set(features) != set(FEATURE_COLUMNS):
            raise ValueError("Monitoring reference must contain every expected feature.")
        for column in FEATURE_COLUMNS:
            feature = features[column]
            counts = feature["counts"]
            if not isinstance(counts, list) or not counts or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
            ) or sum(counts) != rows:
                raise ValueError(f"Invalid monitoring histogram counts for {column}.")
            if column in NUMERIC_FEATURES:
                cuts = np.asarray(feature["cut_points"], dtype=float)
                low, high = float(feature["minimum"]), float(feature["maximum"])
                if feature["type"] != "numeric" or cuts.ndim != 1 or len(cuts) < 1:
                    raise ValueError(f"Invalid monitoring numeric bins for {column}.")
                if not np.isfinite(cuts).all() or not (np.diff(cuts) > 0).all():
                    raise ValueError(f"Monitoring bin cuts must be finite and increasing for {column}.")
                if not np.isfinite([low, high]).all() or low > high or len(counts) != len(cuts) + 1:
                    raise ValueError(f"Invalid monitoring numeric bounds for {column}.")
                if cuts[0] != low or cuts[-1] != np.nextafter(high, np.inf):
                    raise ValueError(f"Monitoring tail bins must bracket the training range for {column}.")
            elif feature["type"] != "categorical" or feature["categories"] != list(KNOWN_CATEGORIES[column]) or len(counts) != len(feature["categories"]):
                raise ValueError(f"Invalid monitoring categories for {column}.")
        if reference["reference_id"] != _reference_digest(reference):
            raise ValueError("Monitoring reference integrity digest failed.")
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed monitoring reference artifact.") from error
    return reference


def load_reference(path: str | Path, *, model_path: str | Path) -> dict:
    return validate_reference(json.loads(Path(path).read_text(encoding="utf-8")), model_path=model_path)


def population_stability_index(reference_counts, current_counts) -> float:
    """Sum (current share - reference share) * log(current/reference).

    Each share is floored at 1e-6, then normalized again. Zero-frequency bins
    therefore produce finite evidence instead of disappearing from comparison.
    """
    ref = np.asarray(reference_counts, dtype=float)
    cur = np.asarray(current_counts, dtype=float)
    if ref.ndim != 1 or ref.shape != cur.shape or len(ref) < 1:
        raise ValueError("PSI requires matching one-dimensional histograms.")
    if not np.isfinite(ref).all() or not np.isfinite(cur).all() or (ref < 0).any() or (cur < 0).any() or ref.sum() <= 0 or cur.sum() <= 0:
        raise ValueError("PSI requires finite nonnegative counts with positive totals.")
    epsilon = POLICY["smoothing_probability"]
    ref = np.maximum(ref / ref.sum(), epsilon)
    cur = np.maximum(cur / cur.sum(), epsilon)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _status(psi: float) -> str:
    if psi >= POLICY["investigate_psi"]:
        return "investigate"
    if psi >= POLICY["review_psi"]:
        return "review"
    return "stable"


def evaluate_batch(frame: pd.DataFrame, reference: dict, *, model_path: str | Path) -> dict:
    """Validate a batch, then return aggregates without input rows or IDs."""
    reference = validate_reference(reference, model_path=model_path)
    profiles, warnings = validate_profiles(frame)
    enough = len(profiles) >= POLICY["minimum_batch_size"]
    metrics = []
    for column in FEATURE_COLUMNS:
        feature = reference["features"][column]
        outside = None
        if feature["type"] == "numeric":
            values = profiles[column].to_numpy(dtype=float)
            counts = _histogram(values, feature["cut_points"])
            outside = int(((values < feature["minimum"]) | (values > feature["maximum"])).sum())
        else:
            frequencies = profiles[column].value_counts()
            counts = [int(frequencies.get(value, 0)) for value in feature["categories"]]
        psi = population_stability_index(feature["counts"], counts)
        metrics.append({
            "feature": column, "type": feature["type"], "psi": round(psi, 8),
            "status": _status(psi) if enough else "insufficient_data",
            "current_bin_counts": counts, "outside_training_range_count": outside,
        })
    investigate = [row["feature"] for row in metrics if row["status"] == "investigate"]
    review = [row["feature"] for row in metrics if row["status"] == "review"]
    status = "insufficient_data" if not enough else "investigate" if investigate else "review" if review else "stable"
    return {
        "report_version": "1.0", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_id": reference["reference_id"], "model_sha256": reference["model_sha256"],
        "schema_version": SCHEMA_VERSION, "n": len(profiles), "reference_rows": reference["reference_rows"],
        "minimum_batch_size": POLICY["minimum_batch_size"], "drift_evaluated": enough,
        "status": status, "investigate_features": investigate, "review_features": review,
        "input_quality": {"valid": True, "normalization_warning_count": len(warnings)},
        "feature_metrics": metrics,
        "interpretation": "Input distribution alert only. Prediction quality, concept drift and real sales are not measured.",
        "next_action": (
            "Collect a representative batch of at least 200 rows before making drift decisions."
            if not enough else
            "Inspect data collection, units, category mix, and population changes; seek verified outcomes before changing the model. Do not automatically retrain."
            if investigate or review else
            "Continue periodic monitoring; stable marginals do not prove stable accuracy or fairness."
        ),
    }


def select_training_profiles(profiles: pd.DataFrame, membership: pd.DataFrame,
                             expected_counts: dict | None = None) -> pd.DataFrame:
    """Use saved membership; exclude nontraining values before validation."""
    if membership.columns.tolist() != [ID_COLUMN, "split"]:
        raise ValueError("Saved membership must contain exactly id,split.")
    if profiles[ID_COLUMN].isna().any() or membership[ID_COLUMN].isna().any():
        raise ValueError("Missing IDs in training provenance.")
    data_ids = profiles[ID_COLUMN].astype(str)
    member_ids = membership[ID_COLUMN].astype(str)
    if data_ids.duplicated().any() or member_ids.duplicated().any():
        raise ValueError("Duplicate IDs in training provenance.")
    if set(data_ids) != set(member_ids):
        raise ValueError("Saved membership does not match the cleaned source IDs.")
    if set(membership["split"]) != {"train", "validation", "holdout"}:
        raise ValueError("Saved membership must identify train, validation and holdout.")
    counts = membership["split"].value_counts().to_dict()
    if expected_counts is not None and counts != expected_counts:
        raise ValueError("Saved membership counts do not match model metadata.")
    train_ids = member_ids.loc[membership["split"].eq("train")]
    indexed = profiles.copy()
    indexed[ID_COLUMN] = data_ids
    selected = indexed.set_index(ID_COLUMN).loc[train_ids].reset_index()
    normalized, _ = validate_profiles(selected.loc[:, REQUIRED_COLUMNS])
    return normalized


def verify_membership_from_seed(ids: pd.Series, target: pd.Series,
                                membership: pd.DataFrame, *, seed: int) -> None:
    """Reject changed split assignments even if their row counts still match.

    Labels reproduce the original stratified split identities only. Neither
    labels nor excluded feature values are used to fit monitoring histograms.
    """
    from sklearn.model_selection import train_test_split
    labels = pd.to_numeric(target, errors="raise").to_numpy()
    if len(ids) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("The cleaned source must have binary labels for split identity verification.")
    positions = np.arange(len(ids))
    train, remainder = train_test_split(positions, test_size=.30, stratify=labels, random_state=seed)
    validation, holdout = train_test_split(remainder, test_size=.50, stratify=labels[remainder], random_state=seed + 1)
    assignments = np.empty(len(ids), dtype=object)
    assignments[train], assignments[validation], assignments[holdout] = "train", "validation", "holdout"
    expected = dict(zip(ids.astype(str), assignments))
    actual = dict(zip(membership[ID_COLUMN].astype(str), membership["split"]))
    if actual != expected:
        raise ValueError("Saved split membership does not reproduce the model's source and seed.")


def rebuild_reference(root: Path = ROOT) -> dict:
    """Rebuild from the exact cleaned training source and saved train split."""
    root = Path(root)
    model_path = root / "models/model_bundle.joblib"
    source_path = root / "data/processed/train_clean.csv"
    membership_path = root / "data/splits/split_membership.csv"
    # Import lazily so read-only batch monitoring does not deserialize a model.
    from .scoring import load_bundle
    metadata = load_bundle(model_path)["metadata"]
    source_hash = sha256_file(source_path)
    if source_hash != metadata.get("source_sha256", {}).get("train_clean.csv"):
        raise ValueError("Cleaned training source SHA256 does not match the saved model; reproduce training first.")
    profiles = pd.read_csv(source_path, usecols=REQUIRED_COLUMNS + [TARGET_COLUMN], dtype={ID_COLUMN: "string"})
    membership = pd.read_csv(membership_path, dtype={ID_COLUMN: "string"})
    training = select_training_profiles(profiles, membership, metadata["split_counts"])
    verify_membership_from_seed(profiles[ID_COLUMN], profiles[TARGET_COLUMN], membership, seed=metadata["seed"])
    reference = create_reference(training, model_path=model_path, source={
        "selection_scope": "saved_training_split_only",
        "included_split": "train", "excluded_splits": ["validation", "holdout"],
        "cleaned_source": "data/processed/train_clean.csv", "cleaned_source_sha256": source_hash,
        "membership": "data/splits/split_membership.csv", "membership_sha256": sha256_file(membership_path),
        "selection_seed": metadata["seed"],
        "contains_raw_profiles": False, "contains_labels": False,
    })
    path = root / "models/monitoring_reference.json"
    _save_json(path, reference)
    print(f"Saved training-only reference for {reference['reference_rows']:,} rows to {path}.")
    return reference


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def demonstrate(root: Path = ROOT, output_dir: Path | None = None) -> dict:
    root = Path(root)
    output_dir = root / "docs/reports/monitoring" if output_dir is None else Path(output_dir)
    model_path = root / "models/model_bundle.joblib"
    reference = load_reference(root / "models/monitoring_reference.json", model_path=model_path)
    profiles = read_profile_csv((root / "data/demo/customer_profiles_demo.csv").read_bytes())
    baseline = evaluate_batch(profiles, reference, model_path=model_path)
    shifted = profiles.copy()
    shifted["Age"] = 75
    shifted["Annual_Income_USD"] = 250000
    shifted["Daily_Commute_km"] = 150.0
    shifted["Home_Charging_Possible"] = "No"
    drift = evaluate_batch(shifted, reference, model_path=model_path)
    if baseline["status"] != "stable" or drift["status"] != "investigate":
        raise RuntimeError("Drift demonstration did not produce its expected stable/shifted signals.")
    for name, report in [("demo_baseline", baseline), ("demo_shifted", drift)]:
        _save_json(output_dir / f"{name}.json", report)
    summary = {
        "demo": "Unmodified included 500 synthetic profiles versus deliberately changed copies of the same profiles.",
        "reference_id": reference["reference_id"], "model_sha256": reference["model_sha256"],
        "reference_rows": reference["reference_rows"], "batch_rows": len(profiles),
        "baseline_status": baseline["status"], "shifted_status": drift["status"],
        "baseline_max_psi": max(row["psi"] for row in baseline["feature_metrics"]),
        "shifted_max_psi": max(row["psi"] for row in drift["feature_metrics"]),
        "shifted_investigate_features": drift["investigate_features"],
        "changes": {"Age": 75, "Annual_Income_USD": 250000, "Daily_Commute_km": 150.0, "Home_Charging_Possible": "No"},
        "quality_evaluation": "Not performed; demonstration checks drift detection, not prediction accuracy.",
        "privacy": "Output reports contain aggregate histograms and no profile rows, identifiers or labels.",
        "model_unchanged": sha256_file(model_path) == reference["model_sha256"],
    }
    _save_json(output_dir / "demo_summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    columns = list(FEATURE_COLUMNS)
    baseline_psi = [row["psi"] for row in baseline["feature_metrics"]]
    shifted_psi = [row["psi"] for row in drift["feature_metrics"]]
    fig, ax = plt.subplots(figsize=(9, 5.6), layout="constrained")
    positions = np.arange(len(columns))
    ax.barh(positions - 0.18, baseline_psi, height=.34, label="Included demo", color="#397a9a")
    ax.barh(positions + 0.18, shifted_psi, height=.34, label="Deliberately shifted demo", color="#c66d36")
    ax.axvline(POLICY["investigate_psi"], color="#8c3636", linestyle="--", label="Investigation setting 0.2")
    ax.set_yticks(positions, [column.replace("_", " ") for column in columns], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Population stability index (larger means more distribution change)")
    ax.set_title("Input monitoring detects the deliberately changed batch", loc="left", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output_dir / "drift_demo.png", dpi=180)
    plt.close(fig)
    print(f"Baseline: {baseline['status']}; shifted batch: {drift['status']}.")
    print(f"Investigate: {', '.join(drift['investigate_features'])}.")
    print(f"Saved aggregate monitoring evidence to {output_dir}.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-reference", action="store_true", help="Requires full reproduced EDA/modeling data.")
    action.add_argument("--demo", action="store_true", help="Uses included model, reference and demo; no Kaggle credentials required.")
    action.add_argument("--input", type=Path, help="Validate and monitor a customer profile CSV.")
    parser.add_argument("--output", type=Path, help="JSON report path for --input or output directory for --demo.")
    args = parser.parse_args()
    if args.build_reference:
        if args.output is not None:
            parser.error("--build-reference writes models/monitoring_reference.json; do not pass --output.")
        rebuild_reference()
    elif args.demo:
        demonstrate(output_dir=args.output)
    else:
        model_path = ROOT / "models/model_bundle.joblib"
        reference = load_reference(ROOT / "models/monitoring_reference.json", model_path=model_path)
        profiles = read_profile_csv(args.input.read_bytes())
        report = evaluate_batch(profiles, reference, model_path=model_path)
        output = args.output or ROOT / "docs/reports/monitoring/latest_batch.json"
        _save_json(output, report)
        print(f"Monitoring status: {report['status']} ({report['n']} rows).")
        print(report["next_action"])
        print(f"Saved aggregate report to {output}.")


if __name__ == "__main__":
    main()
