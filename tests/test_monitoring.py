"""Behavioral evidence for train-only, versioned aggregate drift monitoring."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ev_project.monitoring import (
    create_reference, evaluate_batch, load_reference, population_stability_index,
    select_training_profiles, validate_reference, verify_membership_from_seed,
)
from ev_project.schema import ProfileValidationError, read_profile_csv

ROOT = Path(__file__).resolve().parents[1]


class MonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = ROOT / "models/model_bundle.joblib"
        cls.profiles = read_profile_csv((ROOT / "data/demo/customer_profiles_demo.csv").read_bytes())
        cls.reference = create_reference(cls.profiles, model_path=cls.model_path)

    def test_identical_marginals_are_stable_and_saved_aggregates_have_no_rows(self):
        report = evaluate_batch(self.profiles, self.reference, model_path=self.model_path)
        self.assertEqual(report["status"], "stable")
        self.assertTrue(report["drift_evaluated"])
        self.assertTrue(all(row["psi"] == 0 for row in report["feature_metrics"]))
        self.assertNotIn("id", report)
        self.assertNotIn("Will_Buy_EV", json.dumps(report))
        self.assertNotIn("profiles", report)
        self.assertNotIn("rows", self.reference)
        self.assertEqual(self.reference["reference_rows"], len(self.profiles))
        for row in report["feature_metrics"]:
            self.assertEqual(sum(row["current_bin_counts"]), len(self.profiles))

    def test_numeric_tails_and_category_changes_produce_investigation(self):
        changed = self.profiles.copy()
        changed["Age"] = 75
        changed["Annual_Income_USD"] = 250000
        changed["Home_Charging_Possible"] = "No"
        report = evaluate_batch(changed, self.reference, model_path=self.model_path)
        self.assertEqual(report["status"], "investigate")
        self.assertIn("Age", report["investigate_features"])
        self.assertIn("Annual_Income_USD", report["investigate_features"])
        self.assertIn("Home_Charging_Possible", report["investigate_features"])
        metrics = {row["feature"]: row for row in report["feature_metrics"]}
        self.assertEqual(metrics["Age"]["outside_training_range_count"], len(changed))
        self.assertEqual(metrics["Age"]["current_bin_counts"][-1], len(changed))
        self.assertTrue(np.isfinite(metrics["Age"]["psi"]))

    def test_small_batch_does_not_claim_a_drift_decision(self):
        changed = self.profiles.head(199).copy()
        changed["Age"] = 75
        report = evaluate_batch(changed, self.reference, model_path=self.model_path)
        self.assertEqual(report["status"], "insufficient_data")
        self.assertFalse(report["drift_evaluated"])
        self.assertEqual(report["investigate_features"], [])
        self.assertTrue(all(row["status"] == "insufficient_data" for row in report["feature_metrics"]))

    def test_invalid_batches_fail_before_a_monitoring_result(self):
        for change in [lambda frame: frame.drop(columns="Age"),
                       lambda frame: frame.assign(id="duplicate"),
                       lambda frame: frame.assign(Age=-1),
                       lambda frame: frame.assign(Gender="unknown")]:
            with self.subTest(change=change), self.assertRaises(ProfileValidationError):
                evaluate_batch(change(self.profiles), self.reference, model_path=self.model_path)

    def test_reference_rejects_model_replacement_or_accidental_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "changed.joblib"
            model_path.write_bytes(b"a replaced model must require rebuilding the reference")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                validate_reference(self.reference, model_path=model_path)
            reference_path = Path(directory) / "reference.json"
            reference_path.write_text(json.dumps(self.reference), encoding="utf-8")
            self.assertEqual(load_reference(reference_path, model_path=self.model_path)["reference_id"], self.reference["reference_id"])
        changed = copy.deepcopy(self.reference)
        changed["source"]["unexpected_change"] = True
        with self.assertRaisesRegex(ValueError, "integrity"):
            validate_reference(changed, model_path=self.model_path)

    def test_malformed_reference_bins_and_versions_are_rejected(self):
        changes = [
            lambda ref: ref.update(reference_version="future"),
            lambda ref: ref.update(schema_version="bad"),
            lambda ref: ref.update(reference_rows=True),
            lambda ref: ref["features"].pop("Gender"),
            lambda ref: ref["features"]["Age"].update(cut_points=[30, 20]),
            lambda ref: ref["features"]["Age"].update(counts=[-1]),
            lambda ref: ref["features"]["Gender"].update(categories=["other"]),
        ]
        for change in changes:
            ref = copy.deepcopy(self.reference)
            change(ref)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_reference(ref, model_path=self.model_path)

    def test_training_selection_excludes_validation_holdout_values_and_labels(self):
        frame = self.profiles.head(6).copy()
        frame["Will_Buy_EV"] = [0, 1, 0, 1, 0, 1]
        membership = pd.DataFrame({"id": frame["id"], "split": ["train", "train", "train", "train", "validation", "holdout"]})
        original = select_training_profiles(frame, membership, {"train": 4, "validation": 1, "holdout": 1})
        changed = frame.astype(object).copy()
        changed.loc[changed.index[-2:], "Age"] = "INVALID HELDOUT DATA"
        changed.loc[changed.index[-2:], "Will_Buy_EV"] = "INVALID HELDOUT LABEL"
        selected = select_training_profiles(changed, membership, {"train": 4, "validation": 1, "holdout": 1})
        assert_frame_equal(original, selected)
        self.assertNotIn("Will_Buy_EV", selected)
        self.assertEqual(set(selected["id"]), set(frame["id"].head(4)))

    def test_training_membership_rejects_overlap_or_missing_records(self):
        frame = self.profiles.head(6).copy()
        membership = pd.DataFrame({"id": frame["id"], "split": ["train", "train", "train", "train", "validation", "holdout"]})
        bad = membership.copy()
        bad.loc[bad.index[-1], "id"] = bad.iloc[0]["id"]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            select_training_profiles(frame, bad)
        with self.assertRaisesRegex(ValueError, "match"):
            select_training_profiles(frame, membership.head(5))
        with self.assertRaisesRegex(ValueError, "counts"):
            select_training_profiles(frame, membership, {"train": 5, "validation": 0, "holdout": 1})

    def test_seed_verification_rejects_swapped_memberships_with_unchanged_counts(self):
        from sklearn.model_selection import train_test_split
        ids = pd.Series([str(value) for value in range(200)])
        target = pd.Series(np.arange(200) % 2)
        positions = np.arange(200)
        train, remainder = train_test_split(positions, test_size=.30, stratify=target, random_state=42)
        validation, holdout = train_test_split(remainder, test_size=.50, stratify=target.iloc[remainder], random_state=43)
        assignments = np.empty(200, dtype=object)
        assignments[train], assignments[validation], assignments[holdout] = "train", "validation", "holdout"
        membership = pd.DataFrame({"id": ids, "split": assignments})
        verify_membership_from_seed(ids, target, membership, seed=42)
        # A swap preserves counts and IDs, but would leak heldout rows into a reference.
        membership.loc[train[0], "split"] = "holdout"
        membership.loc[holdout[0], "split"] = "train"
        with self.assertRaisesRegex(ValueError, "source and seed"):
            verify_membership_from_seed(ids, target, membership, seed=42)

    def test_psi_zero_frequency_bins_stay_finite_and_histograms_are_checked(self):
        self.assertEqual(population_stability_index([10, 0], [10, 0]), 0)
        self.assertGreater(population_stability_index([10, 0], [0, 10]), 1)
        for ref, cur in [([1, 2], [1]), ([0, 0], [1, 2]), ([1, -1], [1, 2]), ([1, 2], [1, np.inf])]:
            with self.subTest(ref=ref, cur=cur), self.assertRaises(ValueError):
                population_stability_index(ref, cur)


if __name__ == "__main__":
    unittest.main()
