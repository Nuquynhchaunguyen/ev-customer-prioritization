"""Demonstrate scoring from the saved project model in a fresh Python process."""

from pathlib import Path

import numpy as np

from .schema import read_profile_csv
from .scoring import load_bundle, score_profiles


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "models" / "model_bundle.joblib"
    demo_path = root / "data" / "demo" / "customer_profiles_demo.csv"
    profiles = read_profile_csv(demo_path.read_bytes())

    bundle = load_bundle(model_path)
    ranked, shortlist, warnings = score_profiles(profiles, bundle, k=50)
    reloaded, _, _ = score_profiles(profiles, load_bundle(model_path), k=50)
    max_difference = np.max(
        np.abs(ranked["yes_probability"] - reloaded["yes_probability"])
    )
    if (
        len(ranked) != len(profiles)
        or len(shortlist) != 50
        or not ranked["id"].equals(reloaded["id"])
        or max_difference > 1e-12
    ):
        raise SystemExit("Saved model check failed: rank count or reload scores differ.")

    first = shortlist.iloc[0]
    print(f"Model: {bundle['metadata']['selected_model']}")
    print(f"Ranked {len(ranked)} demo profiles and shortlisted {len(shortlist)}.")
    print(f"First profile: {first['id']} (Yes score {first['yes_probability']:.6f}).")
    print(f"Ranking matches after reload. Maximum score difference: {max_difference:g}.")
    for warning in warnings:
        print(f"Profile warning: {warning}")


if __name__ == "__main__":
    main()
