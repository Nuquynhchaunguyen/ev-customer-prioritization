"""Tests of input integrity, feature isolation, and reproducible ranking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ev_project.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    SCHEMA_VERSION,
    ProfileValidationError,
    read_profile_csv,
    validate_profiles,
)
from ev_project.scoring import load_bundle, rank_scores, score_profiles


def profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["0001", "0020", "customer-3", "0040"],
            "Age": [35, 44, 60, 27],
            "Annual_Income_USD": [50000, 80000, 110000, 30000],
            "Daily_Commute_km": [12.5, 40, 55.2, 5],
            "Number_of_Cars_Owned": [1, 2, 3, 1],
            "Charging_Stations_Near_Home": [0, 3, 5, 1],
            "Charging_Stations_Near_Work": [1, 4, 8, 2],
            "Environmental_Concern_Level": [1, 3.0, "5", "2"],
            "Gender": ["Female", "Male", "Other", "Female"],
            "City_Type": ["Urban", "Suburban", "Rural", "Urban"],
            "Current_Car_Type": ["Sedan", "SUV", "Truck", "Hatchback"],
            "Home_Charging_Possible": ["No", "Yes", "Yes", "No"],
            "Subsidy_Available": ["No", "Yes", "Yes", "No"],
            "Range_Anxiety_Level": ["Medium", "Low", "High", "Low"],
        }
    )


class RecordingPipeline:
    classes_ = np.array([1, 0])  # Reverse order checks positive-class extraction.
    feature_names_in_ = np.array(FEATURE_COLUMNS)

    def predict_proba(self, features):
        self.seen_features = features.copy()
        scores = features["Annual_Income_USD"].to_numpy() / 200000
        return np.column_stack([scores, 1 - scores])


def bundle(pipeline=None):
    return {
        "pipeline": pipeline or RecordingPipeline(),
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "feature_columns": FEATURE_COLUMNS,
            "training_ranges": {
                "Age": {"min": 25, "max": 69},
                "Daily_Commute_km": {"min": 5, "max": 98.7},
            },
        },
    }


class SchemaTests(unittest.TestCase):
    def test_concern_is_nominal_and_id_strings_keep_leading_zeros(self):
        normalized, warnings = validate_profiles(profiles())
        self.assertEqual(normalized["id"].tolist(), ["0001", "0020", "customer-3", "0040"])
        self.assertEqual(normalized["Environmental_Concern_Level"].tolist(), ["1", "3", "5", "2"])
        self.assertEqual(normalized.columns.tolist(), ["id"] + FEATURE_COLUMNS)
        self.assertEqual(warnings, [])

    def test_missing_columns_and_missing_values_are_rejected(self):
        with self.assertRaisesRegex(ProfileValidationError, "Missing required columns.*Age"):
            validate_profiles(profiles().drop(columns="Age"))
        for column in ["Age", "Gender", "Environmental_Concern_Level", "id"]:
            frame = profiles()
            frame.loc[0, column] = None
            with self.subTest(column=column), self.assertRaises(ProfileValidationError):
                validate_profiles(frame)

    def test_domain_errors_are_rejected(self):
        bad_values = {
            "Age": [121, -1, 30.5, True],
            "Annual_Income_USD": [-1, 30000.5, float("inf"), "not a number"],
            "Daily_Commute_km": [-0.1, float("nan"), float("inf")],
            "Number_of_Cars_Owned": [-1, 1.5],
            "Charging_Stations_Near_Home": [-1, 2.5],
            "Environmental_Concern_Level": [0, 6, 2.5, True],
            "Gender": ["Unknown", ""],
            "City_Type": ["City"],
            "Current_Car_Type": ["Van"],
            "Home_Charging_Possible": ["Maybe"],
            "Subsidy_Available": ["Y"],
            "Range_Anxiety_Level": ["Very High"],
        }
        for column, values in bad_values.items():
            for value in values:
                frame = profiles().astype(object)
                frame.loc[0, column] = value
                with self.subTest(column=column, value=value), self.assertRaises(ProfileValidationError):
                    validate_profiles(frame)

    def test_zero_counts_and_plausible_outside_training_values_are_retained(self):
        frame = profiles()
        frame.loc[0, ["Age", "Annual_Income_USD", "Number_of_Cars_Owned"]] = [75, 20000, 0]
        frame.loc[0, "Daily_Commute_km"] = 103.9
        normalized, _ = validate_profiles(frame)
        self.assertEqual(normalized.loc[0, "Daily_Commute_km"], 103.9)
        self.assertEqual(normalized.loc[0, "Number_of_Cars_Owned"], 0)
        _, _, warnings = score_profiles(frame, bundle(), 1)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(all("retained without clipping" in warning for warning in warnings))

    def test_duplicate_columns_and_duplicate_or_blank_ids_are_rejected(self):
        for ids in [["1", "1", "3", "4"], ["1", " 1 ", "3", "4"], ["1", "", "3", "4"]]:
            frame = profiles()
            frame["id"] = ids
            with self.subTest(ids=ids), self.assertRaises(ProfileValidationError):
                validate_profiles(frame)
        frame = pd.concat([profiles(), profiles()[["Age"]]], axis=1)
        with self.assertRaisesRegex(ProfileValidationError, "Duplicate CSV column names"):
            validate_profiles(frame)

    def test_csv_bom_quotes_and_leading_zero_ids(self):
        frame = profiles()
        frame.loc[0, "id"] = "0001, quoted"
        content = ("\ufeff" + frame.to_csv(index=False)).encode("utf-8")
        decoded = read_profile_csv(content)
        normalized, _ = validate_profiles(decoded)
        self.assertEqual(normalized.loc[0, "id"], "0001, quoted")
        self.assertEqual(normalized.loc[1, "id"], "0020")

    def test_malformed_csv_and_duplicate_headers_are_rejected(self):
        invalid = [b"", b" \n", b"id,Age\n", b"id,Age,Age\n1,25,25\n", b"id, Age,Age\n1,25,25\n",
                   b"id,Age\n1,25,extra\n", b"id,Age\n1\n", b'id,Age\n"unterminated,25', b"id,Age\n1,\x00\n"]
        for content in invalid:
            with self.subTest(content=content), self.assertRaises(ProfileValidationError):
                read_profile_csv(content)

    def test_empty_dataframe_is_rejected(self):
        with self.assertRaisesRegex(ProfileValidationError, "empty"):
            validate_profiles(profiles().iloc[:0])


class ScoringTests(unittest.TestCase):
    def test_target_and_ids_never_enter_prediction(self):
        frame = profiles()
        frame["Will_Buy_EV"] = ["nonsense"] * len(frame)
        frame["free_text"] = "unused"
        model = bundle()
        full, shortlist, warnings = score_profiles(frame, model, 2)
        self.assertEqual(model["pipeline"].seen_features.columns.tolist(), FEATURE_COLUMNS)
        self.assertNotIn("id", model["pipeline"].seen_features)
        self.assertNotIn("Will_Buy_EV", model["pipeline"].seen_features)
        self.assertEqual(full.columns.tolist(), ["rank", "id", "yes_probability"])
        self.assertEqual(full["id"].tolist(), ["customer-3", "0020", "0001", "0040"])
        self.assertEqual(full["yes_probability"].tolist(), [0.55, 0.4, 0.25, 0.15])
        assert_frame_equal(shortlist, full.head(2))
        self.assertTrue(any("Will_Buy_EV" in warning and "Ignored" in warning for warning in warnings))

    def test_exact_k_and_invalid_capacity(self):
        for k in [1, 2, 4]:
            full, shortlist, _ = score_profiles(profiles(), bundle(), k)
            self.assertEqual(len(full), 4)
            self.assertEqual(len(shortlist), k)
        for k in [0, -1, 5, 1.5, True, "2"]:
            with self.subTest(k=k), self.assertRaises(ProfileValidationError):
                score_profiles(profiles(), bundle(), k)

    def test_ties_are_stable_under_input_shuffle(self):
        ids = [f"{value:04d}" for value in range(20)]
        probabilities = np.ones(20) * 0.2
        expected = rank_scores(ids, probabilities)
        permutation = np.random.default_rng(123).permutation(20)
        actual = rank_scores(np.asarray(ids)[permutation], probabilities[permutation])
        assert_frame_equal(expected, actual)
        self.assertNotEqual(expected["id"].tolist(), ids)
        self.assertEqual(expected["rank"].tolist(), list(range(1, 21)))

    def test_nonprobabilities_and_empty_rankings_fail(self):
        for scores in [[np.nan], [np.inf], [-0.01], [1.01], [[0.5]], [0.1, 0.2]]:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                rank_scores(["001"], scores)
        with self.assertRaises(ProfileValidationError):
            rank_scores([], [])
        with self.assertRaises(ProfileValidationError):
            rank_scores(["x", "x"], [0.1, 0.2])

    def test_saved_pipeline_roundtrip_has_probability_parity(self):
        normalized, _ = validate_profiles(profiles())
        preprocessing = ColumnTransformer(
            [("numeric", StandardScaler(), NUMERIC_FEATURES),
             ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)]
        )
        pipeline = Pipeline([("preprocess", preprocessing), ("model", LogisticRegression(random_state=42))])
        pipeline.fit(normalized[FEATURE_COLUMNS], [0, 1, 1, 0])
        original = bundle(pipeline)
        before, _, _ = score_profiles(profiles(), original, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trusted_model.joblib"
            joblib.dump(original, path)
            restored = load_bundle(path)
            after, _, _ = score_profiles(profiles(), restored, 4)
            assert_frame_equal(before, after)
            np.testing.assert_allclose(pipeline.predict_proba(normalized[FEATURE_COLUMNS]), restored["pipeline"].predict_proba(normalized[FEATURE_COLUMNS]), rtol=0, atol=0)

    def test_bundle_schema_and_positive_class_are_checked(self):
        model = bundle()
        model["schema_version"] = "unknown"
        with self.assertRaisesRegex(ValueError, "schema version"):
            score_profiles(profiles(), model, 1)
        model = bundle()
        model["metadata"]["feature_columns"] = ["id"] + FEATURE_COLUMNS
        with self.assertRaisesRegex(ValueError, "feature schema"):
            score_profiles(profiles(), model, 1)
        model = bundle()
        model["pipeline"].classes_ = np.array([0, 2])
        with self.assertRaisesRegex(ValueError, "positive class 1"):
            score_profiles(profiles(), model, 1)


if __name__ == "__main__":
    unittest.main()
