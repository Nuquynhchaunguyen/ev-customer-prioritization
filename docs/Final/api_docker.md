# Reproduce the ranking API and container

The API makes the same saved ranking pipeline available to another local program.
It returns the complete ranking and top K, identifies the exact model file with a
SHA-256 digest, and evaluates aggregate input drift against the training reference.
It does not train a model, contact customers, or retain submitted profiles.

## Install and start the local API

Open the project folder and activate the Python 3.12 or 3.13 environment using the
main README. If the complete dependencies are already installed, skip installation.
These commands work in macOS Terminal, Windows PowerShell, and Command Prompt
once the environment is active.

```bash
python -m pip install --only-binary=:all: -r requirements-service.txt
python -m pip check
python -m uvicorn ev_project.api:app --host 127.0.0.1 --port 8000 --no-access-log
```

Keep that terminal running. Open <http://127.0.0.1:8000/docs> to see the request
schema and try requests. <http://127.0.0.1:8000/schema> lists columns and accepted
categories. <http://127.0.0.1:8000/health> reports readiness and model digest.

In a second terminal, open the same project folder, activate the same environment,
and run the portable HTTP demonstration.

```bash
python -m ev_project.check_api
```

Expected messages are **API verified: ranked 500 profiles and shortlisted 50.**,
**Ranking matches saved pipeline. Maximum score difference: 0.**, and
**Invalid inputs return HTTP 422. OK**. Very small platform differences up to
1e-12 are accepted. The test compares every score and ranking, not just the first
customer. Press **Ctrl+C** in the server terminal when finished.

## Request and response

`POST /rank` accepts a JSON object with `profiles` and `k`. `profiles` contains 1 to
5,000 records with one unique, nonblank text `id` and all 13 predictors listed in
`/schema`. Use JSON numbers for numeric predictors. Environmental concern accepts
a whole number or string code 1 to 5. The demo client shows how CSV values are
normalized before sending them. Each request is ranked independently.

For a small manual request in **POST /rank > Try it out**, use this synthetic
example from the included demo. This is request data, not a terminal command.

```json
{
  "profiles": [
    {
      "id": "668665",
      "Age": 61,
      "Annual_Income_USD": 67725,
      "Daily_Commute_km": 16.9,
      "Number_of_Cars_Owned": 2,
      "Charging_Stations_Near_Home": 7,
      "Charging_Stations_Near_Work": 4,
      "Environmental_Concern_Level": "4",
      "Gender": "Male",
      "City_Type": "Suburban",
      "Current_Car_Type": "Sedan",
      "Home_Charging_Possible": "Yes",
      "Subsidy_Available": "No",
      "Range_Anxiety_Level": "Low"
    }
  ],
  "k": 1
}
```

| Field | Meaning |
|---|---|
| `k` | Whole number from 1 to the number of submitted profiles |
| `ranking` | Every submitted ID, descending Yes score, and rank |
| `shortlist` | The first K rows of the same ranking |
| `model_sha256` | Digest of the exact saved model used |
| `warnings` | Extra columns ignored, or values outside observed training ranges |
| `monitoring` | Aggregate drift results, batch-size limitation, and investigation guidance |

Scores estimate the synthetic `Will_Buy_EV = Yes` label. They do not establish a
real customer's purchase probability or prove a sales increase. Equal scores use
the same deterministic seeded ID tie rule as the app. Extra fields, including the
target, are excluded from prediction and reported in warnings.

Invalid input returns **HTTP 422** with a clear error location/message. Examples
include missing columns, unknown categories, negative or nonfinite numeric values,
duplicate IDs, invalid K, empty batches, oversized batches, and malformed JSON.
The service avoids echoing submitted values in schema validation errors. A missing
or mismatched model/reference prevents startup rather than claiming readiness.
If either artifact changes after startup, health and ranking return **HTTP 503**
until matching artifacts are restored and the service is restarted. Stop the API
before retraining. Rebuild the monitoring reference after retraining and restart
with the matched files. Rebuild the Docker image when its artifacts change.

Run the HTTP contract tests from the project folder.

```bash
python -m unittest discover -s tests -p "test_api.py" -v
```

## Build and run Docker locally

Install and open [Docker Desktop](https://docs.docker.com/desktop/). On Windows,
use **Linux containers** with the WSL 2 backend. Wait until the engine is running.
The same commands below work on Mac and Windows. Run them from the project folder.
The build downloads a Python base image and packages. No Kaggle credentials or full
dataset are needed because the fitted model, training reference, and demo are included.

```bash
docker build -t ev-prioritization:final .
docker run --rm -d --name ev-prioritization-final -p 127.0.0.1:8001:8000 ev-prioritization:final
```

Open <http://127.0.0.1:8001/docs>. In the active project environment run the same
HTTP test against the container port.

```bash
python -m ev_project.check_api --url http://127.0.0.1:8001
```

Expect the same **API verified** and **Invalid inputs return HTTP 422. OK** messages.
Inspect health and run the tests inside the image.

```bash
docker inspect --format "{{.State.Health.Status}}" ev-prioritization-final
docker exec ev-prioritization-final python -m unittest discover -s tests -p "test_api.py" -v
docker exec ev-prioritization-final python -m unittest discover -s tests -p "test_monitoring.py" -v
```

Health may briefly say **starting**. After startup, expect **healthy** and a final
**OK** from each test suite. Then stop the disposable container.

```bash
docker stop ev-prioritization-final
```

If port 8001 is occupied, use `-p 127.0.0.1:8002:8000` and test with
`--url http://127.0.0.1:8002`. If a previous container has the same name, stop that
project container first. If Docker cannot connect, open Docker Desktop and wait for
its engine. Build failure is not a passed test.

The image runs as a user without administrator privileges, binds its published
port to the local computer, and copies only the code, model/reference, synthetic
demo, and extension tests. The explicit `.dockerignore` allowlist excludes full
CSVs, `kaggle.json`, local environments, and Git history. Access logging is disabled;
raw customer payloads are not saved. Internet deployment would require a separate
access-control and transport-security design.

## Evidence and limits

The local evidence files identify the verification time, artifact digest, image
identity, platform, results, and invalid-request tests.

| Saved evidence | Observed result |
|---|---|
| [Local HTTP API](evidence/api_local.json) | 500 ranked, 50 shortlisted, all saved-pipeline scores/ranks matched; invalid requests returned 422 |
| [Linux arm64 Docker](evidence/docker_arm64.json) | Clean image built and ran; healthy; 21 extension tests passed; score difference 0 |
| [Linux amd64 Docker](evidence/docker_amd64.json) | Clean image built and ran under Apple Silicon emulation; healthy; 21 extension tests passed; maximum score difference 2.22e-16 |
| [Windows service wheels](evidence/windows_service_wheels.json) | All 28 pinned service packages resolved as prebuilt packages for Windows x64 Python 3.13 |
| [Windows complete dependencies](evidence/windows_full_requirements.json) | The full requirements resolved as 98 prebuilt packages, including explicit Windows-specific requirements; native Windows execution not tested |

Both container runs passed **No broken requirements found.** and ran as UID 10001.
The Linux x64 run checks that container architecture, rather than native Windows. Recreate evidence after any
code, model, reference, or dependency change. A Windows wheel-availability check is
recorded separately from actual operating-system execution. Independent peer
reproduction and a signed validation form must be supplied by the reviewer.

## Technical references

- [FastAPI request bodies](https://fastapi.tiangolo.com/tutorial/body/)
- [FastAPI validation and HTTP errors](https://fastapi.tiangolo.com/tutorial/handling-errors/)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Docker Python guide](https://docs.docker.com/guides/python/)
- [Dockerfile and health checks](https://docs.docker.com/reference/dockerfile/)
