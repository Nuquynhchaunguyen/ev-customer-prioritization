# EV Customer Prioritization

**Nu Quynh Chau Nguyen | MSBA265 | Fall 2026**

**Which customers should an EV dealership prioritize for contact when its outreach capacity is limited?**

An illustrative EV dealership has **1,000 customer profiles** but staff can contact
only **50 per campaign**. Its sales manager needs a tested rule for choosing which
profiles to review first. An analyst would develop and evaluate that rule.

**Illustrative contact labor budget:** At **10 minutes per contact** and **$30 per
staff hour**, each contact represents **$5** in staff time, or **$250 for 50 contacts**
(8 hours 20 minutes). These planning assumptions exclude analyst work and other costs.

This project uses Logistic Regression and the approved
[Kaggle competition dataset](https://www.kaggle.com/competitions/playground-series-s6e9)
to evaluate whether a ranked shortlist identifies more profiles labeled
`Will_Buy_EV = Yes` than random selection within the same contact capacity and labor
budget. The Streamlit app lets staff choose K, review customer facts, and download
a shortlist with editable conversation guides.

The data are synthetic. Model scores and the illustrative costs do not establish
actual sales, cost savings, or return on investment.

## Submission materials

- **Final:** [Business report PDF](docs/Final/EV_Customer_Prioritization_Final_Report.pdf),
  [readable HTML](docs/Final/EV_Customer_Prioritization_Final_Report.html),
  [stakeholder PowerPoint](docs/Final/EV_Customer_Prioritization_Final_Presentation.pptx),
  [product demonstration](docs/Final/demo_walkthrough.md), and
  [revision log](docs/Final/revision_log.md).
  The report follows the ten required sections and includes the complete
  data-to-action pipeline. [API and Docker](docs/Final/api_docker.md) and
  [input monitoring](docs/Final/monitoring.md) include reproducible checks.
  [Final handoff checklist](docs/Final/submission_checklist.md) distinguishes
  prepared local materials from publication, independent peer replication,
  and the separately submitted signed validation form.
- **P1:** [Combined proposal PDF](docs/P1/EV_Customer_Prioritization_P1.pdf).
  Pages 1-2: topic examples and one-page infographic; page 3: project proposal;
  page 4: dataset and initial exploration; page 5: baseline, feasibility, and next steps;
  page 6: references.
- **P2:**

  1. [Technical paper](docs/P2/EV_Customer_Prioritization_Technical_Report.pdf) follows the Part 5 outline.
  2. [Python source](ev_project/), [Streamlit app](app.py), and [Kaggle data access instructions](#5-download-the-dataset-through-kaggle-api).
  3. [EDA report](docs/reports/01_data_understanding_and_cleaning.html) and [modeling report](docs/reports/02_modeling_and_prioritization.html). The technical paper interprets the results and visuals on pages 3 and 5–7. Open the HTML reports locally in a browser.
  4. **Data-to-action pipeline** on page 4 of the technical paper. Solid arrows show completed steps. Dashed arrows show planned outreach, outcome tracking, and feedback.
  5. [Saved fitted pipeline](models/model_bundle.joblib) and [load-and-test command](#7-check-the-saved-model-and-tests). See page 8 of the technical paper for the result.
  6. [Stakeholder PowerPoint](docs/P2/EV_Customer_Prioritization_Stakeholder_Presentation.pptx).
  7. [Setup](#setup), [run instructions](#run-the-project), and [Python dependencies](requirements.txt).
  8. **Revision summary** in Appendix B on page 10 of the technical paper. It records improvements since P1.

## Project map

```text
ev-customer-prioritization/
├── README.md           Setup and run instructions
├── requirements.txt    Python dependencies
├── requirements-service.txt  Pinned API and container dependencies
├── Dockerfile          Local ranking service image
├── .gitignore          Excludes full data, credentials, and local environments
├── docs/               P1, P2, Final report and PowerPoint, and analysis reports
├── notebooks/          01: data understanding; 02: modeling and evaluation
├── ev_project/         Download, training, scoring, API, and input monitoring
├── data/               Included demo and locally generated data
├── models/             Saved model pipeline and metadata
├── app.py              Streamlit interface
└── tests/              Automated checks
```

## Setup

Install [Python 3.13.9](https://www.python.org/downloads/release/python-3139/),
[Git](https://git-scm.com/downloads), and [VS Code](https://code.visualstudio.com/download).
Python 3.12 is also supported. On Windows, choose the standard **Windows installer
(64-bit)** for x64 Python. Enable the launcher and **Add Python to PATH** during
installation, then reopen VS Code. Git is optional for ZIP.

### 1. Download the project

**Option A - Git clone**

Open a terminal in the folder where you want to keep the project, then run:

```bash
git clone https://github.com/Nuquynhchaunguyen/ev-customer-prioritization.git
cd ev-customer-prioritization
```

**Option B - Download ZIP**

Open the [GitHub repository](https://github.com/Nuquynhchaunguyen/ev-customer-prioritization),
choose **Code > Download ZIP**, and extract it. Both options download the project
code; the full dataset is downloaded through Kaggle API in step 5.

### 2. Open the folder in VS Code

In VS Code, choose **File > Open Folder** and open the project folder you
cloned or extracted. Then choose **Terminal > New Terminal**.

### 3. Create and activate the virtual environment

Use **Python 3.13** for a new setup. Python **3.12** also works with the pinned
packages. If a version check shows **3.14** or any other version, follow
[Troubleshooting](#troubleshooting-python-setup) before creating the venv.

If this project already has a venv, start with the **Activate** command for
your system, then check its version below.

#### Mac

1. **Check Python.** Use the version-specific command so you can select 3.13
   even if your Mac defaults to a newer Python.

   ```bash
   python3.13 --version
   ```

   Continue when it prints **Python 3.13.x**. If the command is unavailable, see
   [Troubleshooting](#troubleshooting-python-setup).

2. **Create the venv.**

   ```bash
   python3.13 -m venv .venv-msba265-ev-purchases
   ```

3. **Activate the venv.**

   ```bash
   source .venv-msba265-ev-purchases/bin/activate
   ```

#### Windows (PowerShell)

1. **Check Python.** The first command should show **3.13.x** or **3.12.x**.
   The second should show **win-amd64**. If either check fails, see
   [Troubleshooting](#troubleshooting-python-setup).

   ```powershell
   python --version
   python -c "import sysconfig; print(sysconfig.get_platform())"
   ```

2. **Create the venv.**

   ```powershell
   python -m venv .venv-msba265-ev-purchases
   ```

3. **Activate the venv.**

   ```powershell
   .\.venv-msba265-ev-purchases\Scripts\Activate.ps1
   ```

#### Check the active venv on Mac or Windows

```bash
python --version
python -c "import sys; print(sys.executable)"
```

The version should be **3.13.x** or **3.12.x**, and the path should include
this project's `.venv-msba265-ev-purchases` folder.

**Windows only.** In the same terminal, check the active environment.

```powershell
python -c "import sysconfig; print(sysconfig.get_platform())"
```

The expected Windows result is `win-amd64`. If a check fails, see the
troubleshooting steps below. When your checks pass, continue to step 4.
Activate the venv again when you open a new terminal.

#### Troubleshooting Python setup

- **Mac cannot find `python3.13`.** Try `python3 --version`. If that prints
  **3.13.x** or **3.12.x**, use `python3 -m venv .venv-msba265-ev-purchases` in
  Mac step 2. If you already have `python3.12`, you can use it the same way.
  Otherwise, install Python 3.13 from the [setup link](#setup), reopen VS Code,
  and try `python3.13 --version` again.
- **Windows `python` is missing or fails the version or platform check.** Try `py -3.13`
  or `python3` in place of `python` in Windows steps 1 and 2. For an existing
  3.12 installation with the Python launcher, `py -3.12` also works. Use the
  first option that reports a supported version and `win-amd64`. If needed,
  install the [Windows 64-bit Python listed under Setup](#setup).
- **PowerShell blocks activation.** Open **Command Prompt** using the arrow
  next to the VS Code Terminal **+** button. From the project folder, run
  `.venv-msba265-ev-purchases\Scripts\activate.bat`, then use that terminal for
  the remaining steps.
- **The active venv shows an unsupported Python version or Windows platform.** Close
  any app or notebook using it and run `deactivate`. In Finder or File Explorer,
  rename `.venv-msba265-ev-purchases` to `.venv-msba265-ev-purchases-backup`
  (choose an unused name). Then create a fresh venv using the verified Python
  command for your system. Activate it and repeat the checks before step 4.
  Running the create command again with the same Python keeps that version.

### 4. Install or update the packages

With the verified venv active, run the first command and wait for installation
to succeed. Then run the second command.

```bash
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip check
```

**Expected message:** **No broken requirements found.** To update an existing compatible venv,
activate it and repeat this step.

If pip reports `No matching distribution found for pandas==2.2.3`, check the
active Python version again. This release has no prebuilt package for Python
3.14. Follow [Troubleshooting](#troubleshooting-python-setup) to replace a
3.14 venv with a supported version before retrying. On Windows, also confirm `win-amd64`.

## Run the project

Keep the venv active for all commands below. Run each command separately and
wait for it to finish before starting the next one.

### 5. Download the dataset through Kaggle API

`kaggle.json` is a small file containing your Kaggle API login key. It lets the
script download data using your own Kaggle account. The dataset CSVs are
obtained by running the command below.

First, sign in to your own [Kaggle account](https://www.kaggle.com/) and open the
[approved competition](https://www.kaggle.com/competitions/playground-series-s6e9).
If Kaggle asks you to join or accept its rules, review and complete those steps.

**Already have a working `kaggle.json` in Downloads?** Keep using it and go to
**Run the download** below. Opening a new VS Code window or cloning the project
again does not require a new key.

**First time setting up Kaggle access**

1. Open [Kaggle API settings](https://www.kaggle.com/settings/api) and select
   **API Tokens**. Scroll down to **Legacy API Credentials**.
2. Click **Create Legacy API Key**. If a confirmation appears, review it and
   choose **Continue**. Kaggle then downloads a file named `kaggle.json`.
   Creating a new legacy key invalidates the previous legacy key.
3. Open **Finder > Downloads** on Mac or **File Explorer > Downloads** on Windows.
   Confirm that `kaggle.json` is there. If your browser saved it elsewhere, move it
   into Downloads. If the browser added a number to the new filename, rename
   that new file to `kaggle.json`, replacing the outdated key file.
   Keep this file outside the project; it is private and should stay off GitHub.

**Run the download - both Mac and Windows**

Return to the project's VS Code terminal with the venv active, then run:

```bash
python ev_project/download_data.py
```

Expected message: **Downloaded train.csv, test.csv, and sample_submission.csv.**
The three files appear in the main project folder beside `app.py`.
If they already exist, the script reports **All three raw CSVs already exist;
nothing was overwritten.** Continue to step 6 in either case.

If you previously configured other Kaggle credentials, the script uses those;
otherwise it looks for `kaggle.json` in Downloads. Each person uses their own credentials and follows
the competition's license and rules; full CSVs and keys are excluded from GitHub.

| If downloading fails | What to check |
|---|---|
| Credentials are required | Confirm `kaggle.json` is in Downloads, with that exact name rather than `kaggle.json.txt` |
| HTTP 401/403 | Check your key and the same account's access to the competition |
| HTTP 404 | Check that your project copy includes the latest downloader fix |

If the key is in another folder and you are using a legacy key file, pass its
path with `--credentials`. For example, on Windows PowerShell:

```powershell
python ev_project/download_data.py --credentials "C:\path\to\kaggle.json"
```

| File | Purpose |
|---|---|
| `train.csv` | 668,665 labeled profiles with 13 predictors and the Yes/No target |
| `test.csv` | 286,571 unlabeled profiles for scoring |
| `sample_submission.csv` | Kaggle output template; not customer input |

Field definitions and source limits are in the
[data dictionary](docs/reports/tables/data_dictionary.csv). The data contain no
customer names, phone numbers, or observed outreach outcomes.

### 6. Run the analysis

Run these commands in order. **Executing notebook...** or
**Executing modeling notebook...** means processing has started and may take a few
minutes. Wait for the **Saved** message and terminal prompt before continuing.
Stop any running API first; rerunning replaces generated reports and model files.

**1. Explore and clean the data.**

```bash
python ev_project/run_eda.py
```

**Expected message:**
**Saved executed notebook and docs/reports/01_data_understanding_and_cleaning.html.**

**2. Train and evaluate the models.**

```bash
python ev_project/run_modeling.py
```

**Expected message:**
**Saved executed notebook and docs/reports/02_modeling_and_prioritization.html.**

If **Parent appears to have exited, shutting down** appears **after this saved
message**, press **Enter** on an empty line to show the prompt, then continue below.

**3. Update the monitoring reference to match the retrained model.**

```bash
python -m ev_project.monitoring --build-reference
```

**Expected message:**
**Saved training-only reference for 468,065 rows to .../models/monitoring_reference.json.**
The path varies by computer.

Continue to step 7. If you already use the API or Docker, restart the API and
rebuild the image before testing the updated model.

**Other messages:** **Kernel is running over TCP without encryption** describes
an unencrypted kernel connection. A traceback or missing saved report needs investigation.

Open the [EDA report](docs/reports/01_data_understanding_and_cleaning.html) and
[modeling report](docs/reports/02_modeling_and_prioritization.html) locally in a
browser. Cleaned CSVs are in `data/processed/`; fitted model files are in `models/`.

**Expected report results:** **468,065 training / 100,300 validation / 100,300
holdout** profiles. The 5% shortlist contains **4,246 Yes labels in 5,015 profiles
(84.67%)**; the 10% shortlist contains **7,902 in 10,030 (78.78%)**.
These are synthetic-label results, not measured sales.

To run interactively instead, install VS Code's **Python** and **Jupyter**
extensions, select this project's venv with **Select Kernel**, and **Run All** in
notebook 01 followed by notebook 02.

### 7. Check the saved model and tests

**Check the saved model.** This works directly after step 4 using the included
model and demo profiles.

```bash
python -m ev_project.check_saved_model
```

**Expected message (included model):**

**Model: Logistic Regression**<br>
**Ranked 500 demo profiles and shortlisted 50.**<br>
**First profile: 668721 (Yes score 0.930806).**<br>
**Ranking matches after reload. Maximum score difference: 0.**

**Run the automated tests.**

```bash
python -m unittest discover -s tests -v
```

**Expected message:** **Ran 74 tests** followed by **OK**. The elapsed time varies.

Warnings such as **missing ScriptRunContext** or **StarletteDeprecationWarning**
may appear; check the final summary to confirm success.

The included [demo CSV](data/demo/customer_profiles_demo.csv) shows the input
format. Use one unique `id` and the 13 predictor columns per row, with headers
matching the [template](data/demo/customer_profiles_template.csv). The pipeline
selects predictors by header and estimates the supplied `Will_Buy_EV = Yes`
label. These scores support ranking.

### 8. Try the Streamlit app (optional)

The saved model and 500-profile demo are included, so trying the app does not
require the full-data download or retraining. After setup, run:

```bash
python -m streamlit run app.py
```

**Expected message:** **You can now view your Streamlit app in your browser.**
Open the **Local URL** shown, usually **http://localhost:8501**. Keep the terminal
running; press **Ctrl+C** there to stop the app.

1. Choose **Try the demo**, set K to **50**, and click **Create shortlist**.
2. Review the 50 ranked profiles and their customer information.
3. Select a customer to review/edit their conversation guide, then download
   the shortlist or prepare and download the guide CSV.

For your own upload, use `test.csv` or a CSV matching the demo's columns.
Use unique IDs; `sample_submission.csv` is not a valid customer upload.
Download edits before closing the session. The app does not contact customers.

If a module is missing, reactivate the venv and repeat step 4. If a file cannot
be found, check that your terminal is in the folder containing `README.md`.

### 9. Try the ranking API

Complete [setup steps 1–4](#setup), including the full `requirements.txt`.
Steps 9–11 use the included model, monitoring reference, and demo; no Kaggle
download or retraining is needed. Full analysis reproduction is in steps 5–7.

**Terminal A** is your current terminal, where you start the API.
**Terminal B** is a new terminal you open to run the check.

**Terminal A — start the API in your current terminal**, with the project environment active:

```bash
python -m uvicorn ev_project.api:app --host 127.0.0.1 --port 8002 --no-access-log
```

**Expected message:** **Uvicorn running on http://127.0.0.1:8002**.
Open **http://127.0.0.1:8002/health** and confirm **"status": "ok"**.
Interactive documentation is at **http://127.0.0.1:8002/docs**.

**Open Terminal B in the same VS Code window:**

1. Keep **Terminal A running the API** open. **Do not press Ctrl+C yet.**
2. Choose **Terminal → New Terminal**, or click **+** in the Terminal panel.
3. This new terminal is **Terminal B**. Stay in the same project folder and run the commands for your system:

**Mac**

```bash
source .venv-msba265-ev-purchases/bin/activate
python -m ev_project.check_api --url http://127.0.0.1:8002
```

**Windows PowerShell**

```powershell
.\.venv-msba265-ev-purchases\Scripts\Activate.ps1
python -m ev_project.check_api --url http://127.0.0.1:8002
```

**Windows Command Prompt** — if PowerShell blocks activation, select
**Command Prompt** from the terminal dropdown, then run:

```bat
.venv-msba265-ev-purchases\Scripts\activate.bat
python -m ev_project.check_api --url http://127.0.0.1:8002
```

**Expected message:**

**API verified: ranked 500 profiles and shortlisted 50.**<br>
**Ranking matches saved pipeline. Maximum score difference: 0.**<br>
**Invalid inputs return HTTP 422. OK**

Score differences up to **1e-12** are accepted across platforms; ranks must match.
**HTTP 422** is expected for the intentionally invalid requests; the final **OK**
confirms the checks passed.

**After OK:** Return to **Terminal A (running the API)** and press **Ctrl+C**
to stop it. Continue with Docker and monitoring in **Terminal B**, keeping its
environment active.

| If you see | What to do |
|---|---|
| **address already in use** | Choose another free port, such as 8003. Replace 8002 in the start command, browser URLs, and checker command. |
| **Cannot reach the API** | Check that the API terminal is still running and the checker uses the same port. |

The API accepts up to 5,000 profiles per batch and returns the ranking, top K,
validation warnings, and aggregate input monitoring. It uses the same saved
pipeline as Streamlit and does not save customer requests or send messages.
See [API details](docs/Final/api_docker.md).


### 10. Run the service with Docker

Install and open [Docker Desktop](https://docs.docker.com/desktop/), then wait
for its engine to run. Windows uses **Linux containers** with the **WSL 2**
backend. In **Terminal B (the new terminal from step 9)**, confirm the engine is available.

```bash
docker version
```

**Expected result:** Both **Client** and **Server** information. If the server cannot be
reached, open Docker Desktop and wait for startup before continuing.

**Build the image.** The first build may take several minutes. These commands
work on Mac and Windows.

```bash
docker build -t ev-prioritization:final .
```

**Expected result:** The build completes without **ERROR** and the prompt returns;
Docker may show **FINISHED**. **CACHED** means earlier steps were reused, so a
repeat build can finish quickly.
The image is named **ev-prioritization:final**.

**Start the container.**

```bash
docker run --rm -d --name ev-prioritization-final -p 127.0.0.1:8001:8000 ev-prioritization:final
```

**Expected result:** A long container ID is printed; it differs each run. The
container runs in the background, so the prompt returns. Continue in this same terminal.

**Check that the container is ready before testing.**

```bash
docker inspect --format "{{.State.Health.Status}}" ev-prioritization-final
```

**Expected message:** **healthy**. If it says **starting**, wait a few seconds and
repeat the same health command. If it says **unhealthy**, inspect
`docker logs ev-prioritization-final` and resolve the startup error first.

After **healthy**, open **http://127.0.0.1:8001/docs**. With the full Python project
environment still active, run the HTTP check against the container's port.

```bash
python -m ev_project.check_api --url http://127.0.0.1:8001
```

**Expected message:**

**API verified: ranked 500 profiles and shortlisted 50.**<br>
**Ranking matches saved pipeline. Maximum score difference: 0.**<br>
**Invalid inputs return HTTP 422. OK**

The same **1e-12** score tolerance applies. A small floating-point difference
can appear on another architecture; the ranking and shortlist must match.

**Run the tests inside the container.**

```bash
docker exec ev-prioritization-final python -m unittest discover -s tests -v
```

**Expected message:** **Ran 21 tests** followed by **OK**; elapsed time varies.
The image contains the 11 API and 10 monitoring checks.

Stop the container after the HTTP check and all 21 tests pass.

```bash
docker stop ev-prioritization-final
```

**Expected message:** **ev-prioritization-final**.

The image includes the service, fitted model, training reference, and synthetic
demo. A Kaggle key and full CSV download are unnecessary for this service.
If port 8001 is occupied, use `-p 127.0.0.1:8002:8000` in the run command and
`--url http://127.0.0.1:8002` in the HTTP check; open **http://127.0.0.1:8002/docs**
in the browser. If the container name is already in use, stop that previous
project container before starting it again.
[Container verification](docs/Final/api_docker.md#build-and-run-docker-locally)
provides further troubleshooting and the recorded run evidence.

### 11. MLOps: Run the input-monitoring example

In **Terminal B**, use the **full Python environment from step 4**. The plotted
demonstration requires the complete `requirements.txt`; the smaller Docker
service environment does not include the plotting packages. No running server
is required. Run the same command on Mac or Windows.

```bash
python -m ev_project.monitoring --demo
```

**Expected messages:**

**Baseline: stable; shifted batch: investigate.**<br>
**Investigate: Age, Annual_Income_USD, Daily_Commute_km, Home_Charging_Possible.**<br>
**Saved aggregate monitoring evidence to .../docs/reports/monitoring.**

The path varies by computer. **investigate** is expected: the demo deliberately
changes four features to test the alert.

Open `docs/reports/monitoring/` in VS Code and inspect these outputs.

| Output | Expected evidence |
|---|---|
| `demo_baseline.json` | **"status": "stable"** for the included 500-profile batch |
| `demo_shifted.json` | **"status": "investigate"** and the four named features |
| `demo_summary.json` | Both statuses, **500** batch rows, **468,065** reference rows, and **"model_unchanged": true** |
| `drift_demo.png` | A chart comparing the original batch with the deliberately shifted batch |

The reference uses **468,065 training profiles**
and excludes validation/holdout features and all target labels. The deliberate
shift tests the alert mechanism. It does not measure real purchase accuracy.
See [monitoring rules, investigation steps, and limitations](docs/Final/monitoring.md).

### Classmate verification summary

| Component | Evidence the classmate should observe |
|---|---|
| FastAPI | Local health is **ok**; HTTP check ranks **500**, selects **50**, matches the saved model, and ends with **HTTP 422. OK** |
| Docker | Image builds; container becomes **healthy**; the same HTTP check passes on port 8001; **21 container tests** end with **OK** |
| MLOps | The original batch is **stable**, the deliberately shifted batch is **investigate**, and the JSON/chart outputs are created |

Record your operating system, Python version, results, and any errors.
Use [step 7](#7-check-the-saved-model-and-tests) for the complete test suite.
Complete the validation form after independent replication has been performed.

## Verification scope

The [local verification records](docs/Final/evidence/) record the tested platform,
dependency versions, artifact identity, API parity, container results, and invalid
request checks. Windows x64 Python 3.13 package availability is checked separately
from native Windows execution. Commands use Python modules and single-line Docker
instructions to work across the documented terminals. A native Windows peer run
and the signed validation form remain part of the final handoff.
