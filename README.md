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

Install [Python 3.13](https://www.python.org/downloads/),
[Git](https://git-scm.com/downloads), and [VS Code](https://code.visualstudio.com/download).
Python 3.12 is also supported. On Windows, enable **Add Python to PATH** during
installation. Reopen VS Code after installing Python. Git is optional for ZIP.

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

First check Python: `python3 --version` on Mac or `python --version` on Windows.
Continue when it reports **3.12.x or 3.13.x**; otherwise install a supported version.

**Mac**

```bash
python3 -m venv .venv-msba265-ev-purchases
source .venv-msba265-ev-purchases/bin/activate
```

**Windows - PowerShell**

```powershell
python -m venv .venv-msba265-ev-purchases
.\.venv-msba265-ev-purchases\Scripts\Activate.ps1
```

The terminal should now show the environment name in parentheses. If this venv
already exists with Python 3.12/3.13, skip its creation line and activate it.
Activate it again whenever you open a new terminal.

If PowerShell blocks activation, run the following in that terminal, then retry
`Activate.ps1`. This setting applies only to the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

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

1. Sign in to [Kaggle](https://www.kaggle.com/) and complete any access/rule steps
   for the [competition](https://www.kaggle.com/competitions/playground-series-s6e9).
2. Create a legacy API key in [Kaggle API settings](https://www.kaggle.com/settings/api).
   Leave the downloaded `kaggle.json` in **Downloads**, outside this project.
3. Run:

```bash
python ev_project/download_data.py
```

Expected: `train.csv`, `test.csv`, and `sample_submission.csv` appear beside
`app.py`. If all three already exist, the script skips downloading without
overwriting them. Each reviewer uses their own credentials; follow Kaggle's
license and rules. Full CSVs and credentials are excluded from GitHub.

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
