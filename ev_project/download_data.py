"""Retrieve this competition's raw CSVs with Kaggle API authentication."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import requests

COMPETITION = "playground-series-s6e9"
FILES = {"train.csv", "test.csv", "sample_submission.csv"}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, help="Path to a legacy kaggle.json file.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    existing = {name for name in FILES if (root / name).exists()}
    if existing == FILES:
        print("All three raw CSVs already exist; nothing was overwritten.")
        return
    if existing:
        raise SystemExit("Some raw CSVs already exist. Review the folder before downloading.")

    headers, auth = {"User-Agent": "kaggle-api-client"}, None
    if os.environ.get("KAGGLE_API_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["KAGGLE_API_TOKEN"]
    elif os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        auth = (os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"])
    else:
        candidates = [args.credentials] if args.credentials else [
            Path.home() / ".kaggle" / "kaggle.json",
            Path.home() / "Downloads" / "kaggle.json",
        ]
        credential_file = next((p for p in candidates if p and p.is_file()), None)
        if credential_file is None:
            raise SystemExit(
                "Kaggle API credentials are required. Use --credentials PATH or "
                "set KAGGLE_API_TOKEN / KAGGLE_USERNAME and KAGGLE_KEY. "
                "Do not put credentials in a notebook or Git repository."
            )
        credentials = json.loads(credential_file.read_text())
        if not credentials.get("username") or not credentials.get("key"):
            raise SystemExit("The credential file does not contain a username and key.")
        auth = (credentials["username"], credentials["key"])

    endpoint = "https://www.kaggle.com/api/v1/competitions/data/download-all/" + COMPETITION
    with tempfile.TemporaryDirectory(prefix="ev-kaggle-download-") as directory:
        archive = Path(directory) / "competition.zip"
        # Requests strips Authorization on redirects to a different host.
        with requests.get(endpoint, auth=auth, headers=headers, stream=True,
                          timeout=(30, 120)) as response:
            if not response.ok:
                raise SystemExit(f"Kaggle returned HTTP {response.status_code}; check credentials/access.")
            with archive.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        output.write(chunk)
        if not zipfile.is_zipfile(archive):
            raise SystemExit("The response was not a valid dataset ZIP.")
        with zipfile.ZipFile(archive) as data:
            if set(data.namelist()) != FILES or data.testzip() is not None:
                raise SystemExit("The ZIP contents or integrity did not match the expected dataset.")
            for name in sorted(FILES):
                with data.open(name) as source, (root / name).open("xb") as output:
                    shutil.copyfileobj(source, output)
    print("Downloaded train.csv, test.csv, and sample_submission.csv.")

if __name__ == "__main__":
    main()
