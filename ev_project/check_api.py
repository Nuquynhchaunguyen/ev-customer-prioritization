"""Portable HTTP demonstration. Run against a local or Docker API."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from .schema import read_profile_csv, validate_profiles
from .scoring import load_bundle, score_profiles

ROOT = Path(__file__).resolve().parents[1]


def http_json(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = Request(base_url.rstrip("/") + path, data=data,
                      headers={"Content-Type": "application/json"} if data else {})
    try:
        with urlopen(request, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def check_api(base_url):
    model_path = ROOT / "models" / "model_bundle.joblib"
    demo = read_profile_csv((ROOT / "data" / "demo" / "customer_profiles_demo.csv").read_bytes())
    normalized, _ = validate_profiles(demo)
    payload = {"profiles": normalized.to_dict(orient="records"), "k": 50}
    health_status, health = http_json(base_url, "/health")
    if health_status != 200 or health.get("status") != "ok":
        raise AssertionError("The API is not ready.")
    expected_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if health.get("model_sha256") != expected_hash:
        raise AssertionError("API artifact differs from the local saved model.")
    status, result = http_json(base_url, "/rank", payload)
    if status != 200 or result.get("count") != len(demo) or result.get("k") != 50:
        raise AssertionError("Valid request did not return the expected ranking.")
    ranked, shortlisted, _ = score_profiles(demo, load_bundle(model_path), 50)
    if [row["id"] for row in result["ranking"]] != ranked["id"].tolist():
        raise AssertionError("HTTP ranking differs from local saved-pipeline ranking.")
    if [row["id"] for row in result["shortlist"]] != shortlisted["id"].tolist():
        raise AssertionError("HTTP shortlist differs from local saved-pipeline shortlist.")
    difference = float(np.max(np.abs(np.array([row["yes_probability"] for row in result["ranking"]])
                                     - ranked["yes_probability"].to_numpy())))
    if difference > 1e-12:
        raise AssertionError(f"HTTP probability difference {difference} exceeds tolerance 1e-12.")
    invalid_payloads = {
        "missing_column": {"profiles": [{k: v for k, v in payload["profiles"][0].items() if k != "Age"}], "k": 1},
        "duplicate_ids": {"profiles": [payload["profiles"][0], payload["profiles"][0]], "k": 1},
        "capacity_exceeds_batch": {"profiles": payload["profiles"][:1], "k": 2},
        "invalid_category": {"profiles": [{**payload["profiles"][0], "Gender": "Unknown"}], "k": 1},
    }
    invalid_statuses = {}
    for name, invalid in invalid_payloads.items():
        code, error = http_json(base_url, "/rank", invalid)
        if code != 422 or not error.get("detail"):
            raise AssertionError(f"{name} did not receive a clear HTTP 422 response.")
        invalid_statuses[name] = code
    evidence = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "base_url": base_url,
                "python": platform.python_version(), "platform": platform.platform(),
                "model": result["model"], "model_sha256": expected_hash,
                "ranked_profiles": result["count"], "shortlisted_profiles": len(result["shortlist"]),
                "maximum_probability_difference": difference, "ranking_matches_saved_pipeline": True,
                "invalid_request_statuses": invalid_statuses,
                "monitoring_status": result["monitoring"].get("status"), "passed": True}
    print(f"API verified: ranked {len(demo)} profiles and shortlisted 50.")
    print(f"Ranking matches saved pipeline. Maximum score difference: {difference:g}.")
    print("Invalid inputs return HTTP 422. OK")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL, including Docker host port")
    parser.add_argument("--evidence", type=Path, help="Optional aggregate JSON evidence destination")
    arguments = parser.parse_args()
    try:
        evidence = check_api(arguments.url)
    except (URLError, TimeoutError) as error:
        raise SystemExit("Cannot reach the API. Start it in another terminal and check the --url address.") from error
    if arguments.evidence:
        arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
        arguments.evidence.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
