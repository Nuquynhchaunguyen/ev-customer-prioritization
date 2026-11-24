"""HTTP contract, shared-artifact parity, validation, and startup integrity."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import shutil
import unittest

import numpy as np
from fastapi.testclient import TestClient

from ev_project.api import MAX_BATCH_SIZE, create_app
from ev_project.schema import read_profile_csv, validate_profiles
from ev_project.scoring import load_bundle, score_profiles

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "model_bundle.joblib"
REFERENCE = ROOT / "models" / "monitoring_reference.json"


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = read_profile_csv((ROOT / "data/demo/customer_profiles_demo.csv").read_bytes())
        normalized, _ = validate_profiles(cls.demo)
        cls.rows = normalized.to_dict(orient="records")
        cls.bundle = load_bundle(MODEL)
        cls.client = TestClient(create_app(model_path=MODEL, reference_path=REFERENCE))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def request(self, rows=None, k=5):
        return self.client.post("/rank", json={"profiles": self.rows[:10] if rows is None else rows, "k": k})

    def test_valid_http_ranking_matches_saved_pipeline_and_monitoring(self):
        response = self.request(self.rows, 50)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        ranked, shortlisted, _ = score_profiles(self.demo, self.bundle, 50)
        self.assertEqual([r["id"] for r in result["ranking"]], ranked["id"].tolist())
        self.assertEqual([r["id"] for r in result["shortlist"]], shortlisted["id"].tolist())
        np.testing.assert_allclose([r["yes_probability"] for r in result["ranking"]], ranked["yes_probability"], rtol=0, atol=1e-12)
        self.assertEqual(result["count"], 500)
        self.assertIn(result["monitoring"]["status"], ["stable", "review", "investigate", "insufficient_data"])

    def test_health_schema_and_openapi_document_input(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertEqual(len(self.client.get("/schema").json()["required_columns"]), 14)
        spec = self.client.get("/openapi.json").json()
        self.assertIn("/rank", spec["paths"])
        self.assertIn("Age", spec["components"]["schemas"]["CustomerProfile"]["required"])

    def test_string_ids_preserved_and_ties_independent_of_input_order(self):
        rows = deepcopy(self.rows[:2])
        rows[0]["id"], rows[1]["id"] = "0001", "0002"
        first = self.request(rows, 2).json()["ranking"]
        second = self.request(rows[::-1], 2).json()["ranking"]
        self.assertEqual(first, second)
        self.assertEqual({r["id"] for r in first}, {"0001", "0002"})

    def test_missing_predictor_and_unknown_category_rejected_without_echo(self):
        row = deepcopy(self.rows[0])
        del row["Age"]
        response = self.request([row], 1)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Age", str(response.json()))
        row = {**self.rows[0], "Gender": "PRIVATE_INPUT_VALUE"}
        response = self.request([row], 1)
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("PRIVATE_INPUT_VALUE", response.text)

    def test_duplicate_and_blank_ids_rejected(self):
        for rows in ([self.rows[0], self.rows[0]], [{**self.rows[0], "id": "   "}]):
            self.assertEqual(self.request(rows, 1).status_code, 422)

    def test_capacity_whole_number_and_within_batch(self):
        for value in [0, -1, 11, True, 1.5, "2"]:
            with self.subTest(k=value):
                self.assertEqual(self.request(k=value).status_code, 422)

    def test_invalid_numeric_values_rejected(self):
        for column, value in [("Age", True), ("Age", 121), ("Number_of_Cars_Owned", 1.5),
                              ("Annual_Income_USD", -1), ("Daily_Commute_km", "NaN"),
                              ("Environmental_Concern_Level", True), ("Environmental_Concern_Level", 8)]:
            with self.subTest(column=column, value=value):
                self.assertEqual(self.request([{**self.rows[0], column: value}], 1).status_code, 422)

    def test_outcomes_and_extra_columns_never_change_predictions(self):
        clean = self.request().json()
        rows = [{**row, "Will_Buy_EV": "Yes", "note": "ignored"} for row in self.rows[:10]]
        result = self.request(rows).json()
        self.assertEqual(clean["ranking"], result["ranking"])
        self.assertTrue(any("Ignored extra columns" in message for message in result["warnings"]))

    def test_empty_over_limit_and_malformed_requests_rejected(self):
        self.assertEqual(self.request([], 1).status_code, 422)
        self.assertEqual(self.request([self.rows[0]] * (MAX_BATCH_SIZE + 1), 1).status_code, 422)
        self.assertEqual(self.client.post("/rank", content="{broken", headers={"content-type": "application/json"}).status_code, 422)

    def test_artifact_change_after_startup_returns_clear_unavailability(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model.joblib"
            reference = Path(temporary) / "reference.json"
            shutil.copyfile(MODEL, model)
            shutil.copyfile(REFERENCE, reference)
            with TestClient(create_app(model_path=model, reference_path=reference)) as client:
                reference.write_text("{}", encoding="utf-8")
                self.assertEqual(client.get("/health").status_code, 503)
                response = client.post("/rank", json={"profiles": self.rows[:1], "k": 1})
                self.assertEqual(response.status_code, 503)
                self.assertIn("restart", response.json()["detail"])

    def test_missing_model_or_reference_prevents_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            with self.assertRaises(FileNotFoundError):
                with TestClient(create_app(model_path=missing, reference_path=REFERENCE)):
                    pass
            with self.assertRaises(FileNotFoundError):
                with TestClient(create_app(model_path=MODEL, reference_path=missing)):
                    pass


if __name__ == "__main__":
    unittest.main()
