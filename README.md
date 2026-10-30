# EV Customer Prioritization

**Nu Quynh Chau Nguyen | MSBA265 | Fall 2026**

**Which customers should an EV dealership prioritize for contact when its outreach capacity is limited?**

This project uses Logistic Regression to rank profiles from the approved
[Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e9).
The Streamlit app lets staff choose a contact capacity, review the top K profiles,
and download a shortlist and editable conversation guides. The data are synthetic;
model scores do not establish real purchases or sales impact.

## Submission materials

- **P1:** [Combined proposal PDF](docs/P1/EV_Customer_Prioritization_P1.pdf).
  Page 1: proposal; page 2: six topic examples; page 3: one-page infographic;
  page 4: dataset, initial exploration, and baseline; page 5: next steps and references.
- **P2 drafts:** [Technical paper](docs/P2/EV_Customer_Prioritization_Technical_Report.pdf)
  and [stakeholder slides](docs/P2/EV_Customer_Prioritization_Stakeholder_Presentation.pptx).
  These will be updated after P1 feedback. Final delivery will be prepared later.

The Assignment Validation form stays local and is submitted separately on Canvas.
P1 can be reviewed from its PDF; the runnable analysis and app support P2.

## Project map

```text
ev-customer-prioritization/
├── README.md           Setup and run instructions
├── requirements.txt    Python dependencies
├── .gitignore          Excludes full data, credentials, and local environments
├── docs/               P1 proposal, P2 drafts, and generated reports
├── notebooks/          01: data understanding; 02: modeling and evaluation
├── ev_project/         Download, preparation, training, and scoring code
├── data/               Included demo and locally generated data
├── models/             Saved model pipeline and metadata
├── app.py              Streamlit interface
└── tests/              Automated checks
```

## Setup

Install [Python 3.13.9](https://www.python.org/downloads/release/python-3139/),
[Git](https://git-scm.com/downloads), and [VS Code](https://code.visualstudio.com/download).
Python 3.12 is also supported. On Windows, enable the **Python launcher** and
**Add Python to PATH** during installation. Reopen VS Code afterward. Git is optional for ZIP.

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

Choose **File > Open Folder**, select the folder containing `README.md`, then
**Terminal > New Terminal**. Run every command below from this main project folder.

### 3. Create and activate the virtual environment

If this venv already exists, run only your system's activation line below, then
verify it. For first setup, follow the Mac or Windows steps.

**Mac**

Check `python3 --version`. If it shows **3.12.x or 3.13.x**, run:

```bash
python3 -m venv .venv-msba265-ev-purchases
source .venv-msba265-ev-purchases/bin/activate
```

**Windows - PowerShell**

Check `python --version`. If it shows **3.12.x or 3.13.x**, run:

```powershell
python -m venv .venv-msba265-ev-purchases
.\.venv-msba265-ev-purchases\Scripts\Activate.ps1
```

**Other Python commands, if needed**

If the main version check fails or shows another version, choose one working
alternative below that reports **3.12.x or 3.13.x**. Run its creation command,
then the activation line from your system's block above.

| Computer | Check the version | Alternative creation command |
|---|---|---|
| Mac | `python3.13 --version` | `python3.13 -m venv .venv-msba265-ev-purchases` |
| Mac | `python --version` | `python -m venv .venv-msba265-ev-purchases` |
| Windows | `py -3.13 --version` | `py -3.13 -m venv .venv-msba265-ev-purchases` |
| Windows | `python3 --version` | `python3 -m venv .venv-msba265-ev-purchases` |

For version-specific Python 3.12 commands, use `python3.12` on Mac or `py -3.12`
on Windows. If none finds a supported version, install Python 3.13 from the link
above and reopen VS Code.

**If PowerShell blocks activation:** open **Command Prompt** from the arrow
beside the Terminal **+** button, stay in the project folder, and run:

```bat
.venv-msba265-ev-purchases\Scripts\activate.bat
```

**Verify - both systems**

```bash
python -c "import sys; print(sys.version.split()[0]); print(sys.executable)"
```

The version should be **3.12.x or 3.13.x**, and the executable path should be
inside this project's `.venv-msba265-ev-purchases` folder. Continue to step 4.
Activate the venv again in each new terminal.

If an existing venv uses another version, close its app/notebook kernels,
run `deactivate`, rename the venv folder to a backup, and recreate it with a
supported Python above. Keep the backup until the replacement works.
Installing packages does not change the venv's Python version.

### 4. Install or update the packages

With the venv active, the following commands work on **both Mac and Windows**:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

Expected: **No broken requirements found**. To update an existing venv after
receiving new project files, activate it and repeat this step.

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

| File | Purpose |
|---|---|
| `train.csv` | 668,665 labeled profiles with 13 predictors and the Yes/No target |
| `test.csv` | 286,571 unlabeled profiles for scoring |
| `sample_submission.csv` | Kaggle output template; not customer input |

Field definitions and source limits are in the
[data dictionary](docs/reports/tables/data_dictionary.csv). The data contain no
customer names, phone numbers, or observed outreach outcomes.

### 6. Run the analysis

Run EDA first, then modeling:

```bash
python ev_project/run_eda.py
python ev_project/run_modeling.py
```

Each command finishes with **Saved executed notebook** and an HTML report path.
Modeling uses the cleaned data from EDA and compares a Dummy baseline,
Logistic Regression, and a Decision Tree. Rerunning replaces generated outputs
and model files while keeping the raw CSVs unchanged.

**Jupyter kernel messages during local runs**

| Message | Meaning |
|---|---|
| `Kernel is running over TCP without encryption` | Jupyter reports that kernel traffic is unencrypted. This is a connection warning, separate from an error in an analysis cell. |
| `Parent appears to have exited, shutting down` | The kernel is closing. When this follows the saved-output message, it can be normal cleanup after the run. |

Confirm completion by checking for **Saved executed notebook**, no Python
traceback, and an HTML report that opens with tables and charts. If the command
stops before saving, shows a traceback, or the report is missing, check the error
before continuing. These warnings alone do not confirm success or failure.

| Output | Location |
|---|---|
| Cleaned data | `data/processed/` |
| EDA report, figures, and tables | `docs/reports/01_data_understanding_and_cleaning.html`, `docs/reports/figures/`, `docs/reports/tables/` |
| Modeling report and evaluation results | `docs/reports/02_modeling_and_prioritization.html`, `docs/reports/modeling/` |
| Fitted model and metadata | `models/model_bundle.joblib`, `models/model_metadata.json` |

Open the HTML reports locally in a browser to see the tables and charts.
To run interactively instead, install VS Code's **Python** and **Jupyter**
extensions, select this project's venv with **Select Kernel**, and **Run All** in
notebook 01 followed by notebook 02.

### 7. Check the saved model and tests

This loads the included model in a fresh Python process:

```bash
python -c "from ev_project.scoring import load_bundle; print(load_bundle('models/model_bundle.joblib')['metadata']['selected_model'])"
python -m unittest discover -s tests -v
```

Expected: **Logistic Regression**, then **Ran 53 tests** and **OK**.
The app demo in step 8 checks scoring with the saved model on new profiles.
Mac execution has been verified; independent Windows reproduction is still pending.

### 8. Try the Streamlit app (optional)

The saved model and 500-profile demo are included, so trying the app does not
require the full-data download or retraining. After setup, run:

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser. Keep the terminal running;
press **Ctrl+C** there to stop the app.

1. Choose **Try the demo**, set K to **50**, and click **Create shortlist**.
2. Review the 50 ranked profiles and their customer information.
3. Select a customer to review/edit their conversation guide, then download
   the shortlist or prepare and download the guide CSV.

For your own upload, use `test.csv` or a CSV matching the demo's columns.
Use unique IDs; `sample_submission.csv` is not a valid customer upload.
Download edits before closing the session. The app does not contact customers.

If a module is missing, reactivate the venv and repeat step 4. If a file cannot
be found, check that your terminal is in the folder containing `README.md`.
