"""Exercise the customer workflow without starting a web server."""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ev_project.schema import FEATURE_COLUMNS, SCHEMA_VERSION, validate_profiles
from ev_project import scoring


def example_profiles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ["001", "002", "003", "004"],
            "Age": [30, 45, 55, 65],
            "Annual_Income_USD": [40000, 65000, 90000, 120000],
            "Daily_Commute_km": [10.0, 20.0, 30.0, 40.0],
            "Number_of_Cars_Owned": [1, 2, 1, 2],
            "Charging_Stations_Near_Home": [2, 3, 4, 5],
            "Charging_Stations_Near_Work": [3, 4, 5, 6],
            "Environmental_Concern_Level": [1, 2, 3, 4],
            "Gender": ["Female", "Male", "Other", "Female"],
            "City_Type": ["Urban", "Suburban", "Rural", "Urban"],
            "Current_Car_Type": ["Sedan", "SUV", "Hatchback", "Truck"],
            "Home_Charging_Possible": ["No", "No", "Yes", "Yes"],
            "Subsidy_Available": ["No", "No", "Yes", "Yes"],
            "Range_Anxiety_Level": ["High", "Medium", "Low", "Low"],
        }
    )


class AppWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        shutil.copy(PROJECT_ROOT / "app.py", self.root / "app.py")
        (self.root / "models").mkdir()
        (self.root / "data" / "demo").mkdir(parents=True)
        self.profiles = example_profiles()
        self.profiles.to_csv(self.root / "data" / "demo" / "customer_profiles_demo.csv", index=False)
        normalized, _ = validate_profiles(self.profiles)
        pipeline = Pipeline(
            [
                ("preprocess", ColumnTransformer([("features", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLUMNS)])),
                ("classifier", DecisionTreeClassifier(max_depth=2, random_state=42)),
            ]
        )
        pipeline.fit(normalized[FEATURE_COLUMNS], [0, 0, 1, 1])
        self.bundle = {
            "schema_version": SCHEMA_VERSION,
            "pipeline": pipeline,
            "metadata": {
                "schema_version": SCHEMA_VERSION,
                "feature_columns": FEATURE_COLUMNS,
                "selected_model": "Test classifier",
                "trained_at_utc": "2026-01-01T00:00:00Z",
                "holdout_metrics": {},
            },
        }
        joblib.dump(self.bundle, self.root / "models" / "model_bundle.joblib")

    def tearDown(self) -> None:
        st.cache_resource.clear()
        self.directory.cleanup()

    def app(self) -> AppTest:
        return AppTest.from_file(self.root / "app.py", default_timeout=15).run()

    @staticmethod
    def create_shortlist(app: AppTest) -> AppTest:
        return app.button[0].click().run()

    def test_demo_ranking_capacity_and_export_use_same_scores(self) -> None:
        with patch("ev_project.scoring.score_profiles", wraps=scoring.score_profiles) as predict:
            with patch("streamlit.download_button", wraps=st.download_button) as download:
                app = self.app()
                self.assertFalse(app.exception)
                app.number_input[0].set_value(2)
                self.create_shortlist(app)
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                ranked = app.session_state["ranked_profiles"]["full_ranked"]
                self.assertEqual(app.session_state["ranked_profiles"]["capacity"], 2)
                self.assertTrue(ranked["yes_probability"].is_monotonic_decreasing)
                self.assertEqual(set(ranked["id"]), {"001", "002", "003", "004"})
                self.assertEqual(len(app.dataframe[0].value), 2)
                self.assertTrue(app.dataframe[0].value["yes_probability"].between(0, 1).all())
                exports = {call.args[0]: call.args[1] for call in download.call_args_list}
                shortlist = pd.read_csv(io.BytesIO(exports["Download contact shortlist"]), dtype={"id": str})
                full = pd.read_csv(io.BytesIO(exports["Download all ranked profiles"]), dtype={"id": str})
                self.assertEqual(len(shortlist), 2)
                self.assertEqual(len(full), 4)
                self.assertEqual(shortlist.columns.tolist(), ["rank", "id", "yes_probability"] + FEATURE_COLUMNS)
                source_by_id = self.profiles.set_index("id")
                for _, row in full.iterrows():
                    expected = source_by_id.loc[row["id"]]
                    for column in ["Gender", "Age", "Annual_Income_USD", "Home_Charging_Possible"]:
                        self.assertEqual(row[column], expected[column])
                self.assertEqual(app.dataframe[0].value.columns.tolist(), ["rank", "id", "yes_probability", "Gender", "Age", "Annual_Income_USD"])
                app.number_input[1].set_value(2).run()
                second = ranked.iloc[1]
                self.assertEqual(app.metric[0].value, str(second["Gender"]))
                self.assertEqual(app.metric[1].value, str(int(second["Age"])))
                self.assertEqual(app.metric[2].value, f"${int(second['Annual_Income_USD']):,}")
                app.multiselect[0].set_value(["Home_Charging_Possible"]).run()
                self.assertEqual(app.dataframe[0].value.columns.tolist(), ["rank", "id", "yes_probability", "Home_Charging_Possible"])
                self.assertTrue(shortlist["yes_probability"].between(0, 1).all())
                pd.testing.assert_frame_equal(shortlist, full.head(2))
                app.number_input[0].set_value(1)
                self.create_shortlist(app)
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state["ranked_profiles"]["capacity"], 1)
                self.assertEqual(predict.call_count, 1, "Changing capacity, visible columns, or customer detail must not recompute scores.")
                self.assertEqual(app.number_input[1].value, 1, "A selected detail rank outside the smaller shortlist must reset.")

    def test_uploaded_outcomes_are_ignored_and_never_exported(self) -> None:
        upload = self.profiles.assign(Will_Buy_EV=["Yes", "Yes", "No", "No"], campaign_note="not a predictor")
        app = self.app()
        app.radio[0].set_value("Upload a CSV").run()
        app.file_uploader[0].set_value(("profiles.csv", upload.to_csv(index=False).encode(), "text/csv")).run()
        self.create_shortlist(app)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any("Will_Buy_EV" in warning.value for warning in app.warning))
        ranked = app.session_state["ranked_profiles"]["full_ranked"]
        self.assertEqual(ranked.columns.tolist(), ["rank", "id", "yes_probability"] + FEATURE_COLUMNS)
        self.assertNotIn("Will_Buy_EV", ranked.columns)
        self.assertNotIn("campaign_note", ranked.columns)

    def test_customer_edits_follow_ids_across_selection_and_capacity_without_rescoring(self) -> None:
        with patch("ev_project.scoring.score_profiles", wraps=scoring.score_profiles) as predict:
            app = self.app()
            self.create_shortlist(app)
            original_ranking = app.session_state["ranked_profiles"]["full_ranked"].copy(deep=True)
            first_id = original_ranking.iloc[0]["id"]
            first_template = app.text_area[0].value
            draft = "My customer-specific questions.\nConfirm home charging before offering a test drive."
            app.text_area[0].set_value(draft).run()
            app.number_input[1].set_value(2).run()
            self.assertNotEqual(app.text_area[0].value, draft)
            app.text_area[0].set_value("Different questions for the second customer.").run()
            app.number_input[1].set_value(1).run()
            self.assertEqual(app.text_area[0].value, draft)
            app.number_input[0].set_value(1)
            self.create_shortlist(app)
            self.assertFalse(app.exception)
            self.assertEqual(app.text_area[0].value, draft)
            self.assertEqual(app.session_state["conversation_workspace"]["edits"][first_id], draft)
            self.assertNotEqual(first_template, draft)
            pd.testing.assert_frame_equal(app.session_state["ranked_profiles"]["full_ranked"], original_ranking)
            self.assertEqual(predict.call_count, 1)

    def test_guide_downloads_match_shortlist_and_refresh_after_edits(self) -> None:
        with patch("streamlit.download_button", wraps=st.download_button) as download:
            app = self.app()
            app.number_input[0].set_value(2)
            self.create_shortlist(app)
            ranked = app.session_state["ranked_profiles"]["full_ranked"]
            first_id = ranked.iloc[0]["id"]
            draft = "Xin chào.\nPlease confirm your charging plans."
            app.text_area[0].set_value(draft).run()
            next(button for button in app.button if button.label == "Prepare guides for the shortlist").click().run()
            self.assertFalse(app.exception)
            exports = {call.args[0]: call.args[1] for call in download.call_args_list}
            guides = pd.read_csv(io.BytesIO(exports["Download conversation guides (CSV)"]), dtype={"id": str, "Environmental_Concern_Level": str})
            self.assertEqual(guides["id"].tolist(), ranked.head(2)["id"].tolist())
            self.assertEqual(guides["rank"].tolist(), [1, 2])
            self.assertEqual(guides.iloc[0]["conversation_guide"], draft)
            self.assertEqual(guides.iloc[0]["guide_status"], "Edited by staff")
            self.assertEqual(guides.iloc[1]["guide_status"], "Template draft")
            self.assertIn(f"Customer ID: {first_id}", exports["Download this guide (TXT)"].decode())
            self.assertIn(draft, exports["Download this guide (TXT)"].decode())
            # CSV stores values, not pandas extension dtype metadata.
            pd.testing.assert_frame_equal(guides.loc[:, ranked.columns], ranked.head(2).reset_index(drop=True), check_dtype=False)
            app.text_area[0].set_value("A revised script.").run()
            self.assertNotIn("export", app.session_state["conversation_workspace"])
            self.assertFalse(any(button.label == "Download conversation guides (CSV)" for button in app.download_button))
            next(button for button in app.button if button.label == "Prepare guides for the shortlist").click().run()
            latest = {call.args[0]: call.args[1] for call in download.call_args_list}
            revised = pd.read_csv(io.BytesIO(latest["Download conversation guides (CSV)"]), dtype={"id": str})
            self.assertEqual(revised.iloc[0]["conversation_guide"], "A revised script.")
            app.number_input[0].set_value(1)
            self.create_shortlist(app)
            self.assertFalse(any(button.label == "Download conversation guides (CSV)" for button in app.download_button))

    def test_reset_restores_only_selected_customers_template_and_invalidates_export(self) -> None:
        app = self.app()
        self.create_shortlist(app)
        original = app.text_area[0].value
        app.text_area[0].set_value("First customer's edited script").run()
        app.number_input[1].set_value(2).run()
        second_id = app.session_state["ranked_profiles"]["full_ranked"].iloc[1]["id"]
        app.text_area[0].set_value("Keep this second draft").run()
        app.number_input[1].set_value(1).run()
        next(button for button in app.button if button.label == "Prepare guides for the shortlist").click().run()
        next(button for button in app.button if button.label == "Reset this guide").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.text_area[0].value, original)
        self.assertEqual(app.session_state["conversation_workspace"]["edits"], {second_id: "Keep this second draft"})
        self.assertNotIn("export", app.session_state["conversation_workspace"])

    def test_different_upload_does_not_inherit_edits_for_the_same_customer_id(self) -> None:
        app = self.app()
        self.create_shortlist(app)
        app.text_area[0].set_value("Private draft for the original file").run()
        previous_key = app.session_state["conversation_workspace"]["key"]
        app.radio[0].set_value("Upload a CSV").run()
        upload = self.profiles.assign(source_note="a different file")
        app.file_uploader[0].set_value(("new.csv", upload.to_csv(index=False).encode(), "text/csv")).run()
        self.assertNotIn("conversation_workspace", app.session_state, "A new file clears previous drafts before a new shortlist is scored.")
        self.create_shortlist(app)
        self.assertFalse(app.exception)
        self.assertNotEqual(app.session_state["conversation_workspace"]["key"], previous_key)
        self.assertEqual(app.session_state["conversation_workspace"]["edits"], {})
        self.assertNotIn("Private draft", app.text_area[0].value)

    def test_large_shortlist_keeps_individual_guide_but_bounds_bulk_export(self) -> None:
        large = pd.concat([self.profiles] * 251, ignore_index=True).head(1001)
        large["id"] = [f"customer-{i:04d}" for i in range(len(large))]
        large.to_csv(self.root / "data" / "demo" / "customer_profiles_demo.csv", index=False)
        app = self.app()
        app.number_input[0].set_value(1001)
        self.create_shortlist(app)
        app.number_input[1].set_value(1001).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.text_area[0].value)
        self.assertTrue(any(button.label == "Download this guide (TXT)" for button in app.download_button))
        self.assertFalse(any(button.label == "Prepare guides for the shortlist" for button in app.button))
        self.assertTrue(any("1,000 or fewer" in notice.value for notice in app.info))

    def test_invalid_uploaded_profiles_do_not_produce_a_shortlist(self) -> None:
        upload = self.profiles.drop(columns="Age")
        app = self.app()
        app.radio[0].set_value("Upload a CSV").run()
        app.file_uploader[0].set_value(("profiles.csv", upload.to_csv(index=False).encode(), "text/csv")).run()
        self.create_shortlist(app)
        self.assertFalse(app.exception)
        self.assertTrue(any("Missing required columns: Age" in error.value for error in app.error))
        self.assertNotIn("ranked_profiles", app.session_state)
        self.assertFalse(any(button.label == "Download contact shortlist" for button in app.download_button))

    def test_malformed_csv_reports_error_before_scoring(self) -> None:
        app = self.app()
        app.radio[0].set_value("Upload a CSV").run()
        app.file_uploader[0].set_value(("profiles.csv", b"id,Age\n001,30,extra\n", "text/csv")).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("header has 2" in error.value for error in app.error))
        self.assertFalse(app.button)

    def test_missing_model_has_clear_setup_message(self) -> None:
        (self.root / "models" / "model_bundle.joblib").unlink()
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(any("model is not ready" in notice.value for notice in app.info))
        self.assertFalse(app.button)


if __name__ == "__main__":
    unittest.main()
