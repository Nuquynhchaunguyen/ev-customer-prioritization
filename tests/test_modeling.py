"""Small-data checks of evaluation isolation and saved-model consistency."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from ev_project import modeling
from ev_project.schema import FEATURE_COLUMNS, TARGET_COLUMN, validate_profiles
from ev_project.scoring import load_bundle, rank_scores, score_profiles


def labeled_profiles(n=1000):
    row = np.arange(n)
    raw = pd.DataFrame({
        "id": [f"{index:06d}" for index in row],
        "Age": 25 + row % 45,
        "Annual_Income_USD": 30000 + row * 100,
        "Daily_Commute_km": 5 + (row % 90),
        "Number_of_Cars_Owned": 1 + row % 4,
        "Charging_Stations_Near_Home": row % 15,
        "Charging_Stations_Near_Work": row % 20,
        "Environmental_Concern_Level": 1 + row % 5,
        "Gender": np.take(["Female", "Male", "Other"], row % 3),
        "City_Type": np.take(["Urban", "Suburban", "Rural"], row % 3),
        "Current_Car_Type": np.take(["Sedan", "SUV", "Truck", "Hatchback"], row % 4),
        "Home_Charging_Possible": np.where(row % 2, "Yes", "No"),
        "Subsidy_Available": np.where(row % 5 == 0, "Yes", "No"),
        "Range_Anxiety_Level": np.take(["Low", "Medium", "High"], row % 3),
    })
    normalized, _ = validate_profiles(raw)
    normalized[TARGET_COLUMN] = (row % 5 == 0).astype(np.int8)
    return normalized


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """A cloneable estimator records the exact rows passed to fitting."""

    def fit(self, features, target):
        self.seen_features_ = features.copy()
        self.seen_target_ = np.asarray(target).copy()
        self.feature_names_in_ = np.asarray(features.columns)
        self.classes_ = np.array([1, 0])
        return self

    def predict_proba(self, features):
        positive = features["Age"].to_numpy(dtype=float) / 100
        return np.column_stack([positive, 1 - positive])


class ModelingIsolationTests(unittest.TestCase):
    def test_stratified_splits_are_disjoint_complete_and_deterministic(self):
        data = labeled_profiles()
        first = modeling.make_splits(data)
        second = modeling.make_splits(data)
        expected_counts = {"train": 700, "validation": 150, "holdout": 150}
        sets = {name: set(frame["id"]) for name, frame in first.items()}
        for name, frame in first.items():
            assert_frame_equal(frame, second[name])
            self.assertEqual(len(frame), expected_counts[name])
            self.assertAlmostEqual(frame[TARGET_COLUMN].mean(), 0.2)
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["holdout"])
        self.assertFalse(sets["validation"] & sets["holdout"])
        self.assertEqual(set.union(*sets.values()), set(data["id"]))
        changed_seed = modeling.make_splits(data, seed=123)
        self.assertNotEqual(set(changed_seed["holdout"]["id"]), sets["holdout"])

    def test_final_fit_clones_and_uses_train_plus_validation_only(self):
        splits = modeling.make_splits(labeled_profiles())
        initial = Pipeline([("classifier", RecordingClassifier())])
        initial.fit(splits["train"][FEATURE_COLUMNS], splits["train"][TARGET_COLUMN])
        final = modeling.fit_final_model(initial, splits)
        original_rows = initial.named_steps["classifier"].seen_features_
        final_rows = final.named_steps["classifier"].seen_features_
        self.assertIsNot(initial, final)
        self.assertEqual(len(original_rows), 700)
        self.assertEqual(len(final_rows), 850)
        self.assertEqual(final_rows.columns.tolist(), FEATURE_COLUMNS)
        self.assertNotIn("id", final_rows)
        self.assertNotIn(TARGET_COLUMN, final_rows)
        # Unique income values identify rows without passing customer IDs to a model.
        expected = set(pd.concat([splits["train"], splits["validation"]])["Annual_Income_USD"])
        self.assertEqual(set(final_rows["Annual_Income_USD"]), expected)
        self.assertFalse(set(final_rows["Annual_Income_USD"]) & set(splits["holdout"]["Annual_Income_USD"]))

    def test_selection_rejects_holdout_metrics_and_uses_prespecified_ties(self):
        metrics = pd.DataFrame({
            "model": ["Decision Tree", "Logistic Regression", "Dummy baseline"],
            "split": ["validation"] * 3,
            "lift_at_k": [2.0, 2.0, 1.0],
            "roc_auc": [0.8, 0.8, 0.5],
        })
        self.assertEqual(modeling.select_model(metrics), "Logistic Regression")
        tied = metrics.assign(lift_at_k=1.0, roc_auc=0.5)
        self.assertEqual(modeling.select_model(tied), "Dummy baseline")
        held_out = metrics.copy()
        held_out.loc[0, "split"] = "holdout"
        with self.assertRaisesRegex(ValueError, "validation results only"):
            modeling.select_model(held_out)

    def test_positive_class_order_and_ranking_labels_remain_aligned(self):
        data = labeled_profiles(200)
        pipeline = Pipeline([("classifier", RecordingClassifier())]).fit(data[FEATURE_COLUMNS], data[TARGET_COLUMN])
        probabilities = modeling._positive_probability(pipeline, data)
        np.testing.assert_array_equal(probabilities, data["Age"].to_numpy(dtype=float) / 100)
        summary, ranking_metrics, predictions = modeling.evaluate_final(pipeline, data, "Recording test")
        expected_ranking = rank_scores(data["id"], probabilities)
        by_id = data.set_index("id")[TARGET_COLUMN]
        expected_k = int(np.ceil(0.10 * len(data)))
        yes_in_top_k = int(expected_ranking.head(expected_k)["id"].map(by_id).sum())
        self.assertEqual(summary.iloc[0]["precision_at_k"], yes_in_top_k / expected_k)
        self.assertEqual(summary.iloc[0]["lift_at_k"], (yes_in_top_k / expected_k) / data[TARGET_COLUMN].mean())
        self.assertEqual(predictions["y_true"].tolist(), data[TARGET_COLUMN].tolist())
        shuffled = data.sample(frac=1, random_state=123).reset_index(drop=True)
        shuffled_summary, shuffled_ranking, _ = modeling.evaluate_final(pipeline, shuffled, "Recording test")
        assert_frame_equal(summary, shuffled_summary)
        assert_frame_equal(ranking_metrics, shuffled_ranking)

    def test_scaler_and_category_vocabulary_fit_training_only(self):
        data = labeled_profiles(1000)
        splits = modeling.make_splits(data)
        # A category can be valid in the schema yet absent from the fitting subset.
        train = splits["train"].copy()
        train["Gender"] = "Female"
        candidate = modeling.make_candidates()["Logistic Regression"]
        modeling._fit_checked(candidate, train)
        preprocessing = candidate.named_steps["preprocessor"]
        scaler = preprocessing.named_transformers_["numeric"]
        expected_means = train[modeling.NUMERIC_FEATURES].mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected_means, rtol=0, atol=1e-12)
        encoder = preprocessing.named_transformers_["categorical"]
        gender_index = modeling.CATEGORICAL_FEATURES.index("Gender")
        self.assertEqual(encoder.categories_[gender_index].tolist(), ["Female"])
        # Prediction can encode a known-but-unseen category without refitting it.
        validation = splits["validation"].copy()
        validation["Gender"] = "Other"
        modeling._positive_probability(candidate, validation)
        self.assertEqual(encoder.categories_[gender_index].tolist(), ["Female"])

    def test_saved_application_bundle_is_the_holdout_evaluated_pipeline(self):
        data = labeled_profiles(1000)
        splits = modeling.make_splits(data)
        candidate = Pipeline([("classifier", RecordingClassifier())])
        modeling._fit_checked(candidate, splits["train"])
        candidates = {"Logistic Regression": candidate}
        validation_metrics, validation_ranking, validation_predictions = modeling.evaluate_candidates(candidates, splits["validation"])
        final = modeling.fit_final_model(candidate, splits)
        holdout_metrics, holdout_ranking, holdout_predictions = modeling.evaluate_final(final, splits["holdout"], "Logistic Regression")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/processed").mkdir(parents=True)
            data.to_csv(root / "data/processed/train_clean.csv", index=False)
            data.to_csv(root / "train.csv", index=False)
            test = labeled_profiles(10).drop(columns=TARGET_COLUMN)
            test["id"] = [f"test-{index:03d}" for index in range(10)]
            test.to_csv(root / "test.csv", index=False)
            metadata = modeling.save_training_artifacts(
                root, splits, candidates, validation_metrics, validation_ranking,
                validation_predictions, "Logistic Regression", final,
                holdout_metrics, holdout_ranking, holdout_predictions,
            )
            loaded = load_bundle(root / "models/model_bundle.joblib")
            scored, _, _ = score_profiles(splits["holdout"].drop(columns=TARGET_COLUMN), loaded, 10)
            actual_by_id = scored.set_index("id")["yes_probability"]
            expected_by_id = holdout_predictions.set_index("id")["yes_probability"]
            np.testing.assert_array_equal(actual_by_id.loc[expected_by_id.index], expected_by_id)
            fitted_rows = loaded["pipeline"].named_steps["classifier"].seen_features_
            self.assertEqual(len(fitted_rows), 850)
            self.assertEqual(metadata["split_counts"], {"train": 700, "validation": 150, "holdout": 150})
            demo = pd.read_csv(root / "data/demo/customer_profiles_demo.csv", dtype={"id": "string"})
            self.assertNotIn(TARGET_COLUMN, demo)
            self.assertFalse(set(demo["id"]) & set(data["id"]))


if __name__ == "__main__":
    unittest.main()
