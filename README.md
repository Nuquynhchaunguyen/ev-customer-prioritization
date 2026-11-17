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
  8. **P1 feedback update** in Appendix B on page 10 of the technical paper. It records improvements since P1 and the status of instructor feedback.

## Project map

```text
ev-customer-prioritization/
├── README.md           Setup and run instructions
├── requirements.txt    Python dependencies
├── .gitignore          Excludes full data, credentials, and local environments
├── docs/               P1 proposal, P2 report and PowerPoint slides, and analysis reports
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

First check that `python` points to Python 3.12 or 3.13 and prints `win-amd64`.

```powershell
python -c "import sys, sysconfig; print(sys.version.split()[0]); print(sysconfig.get_platform())"
```

If both values match, create and activate the environment:

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

For an installed Python 3.12, you can also use `python3.12` on Mac or
`py -3.12` on Windows. On Windows, check the chosen interpreter reports
`win-amd64` by substituting it for `python` in the platform check above.
If none finds a supported version, install Python 3.13 from the link
above and reopen VS Code.

**If PowerShell blocks activation:** open **Command Prompt** from the arrow
beside the Terminal **+** button, stay in the project folder, and run:

```bat
.venv-msba265-ev-purchases\Scripts\activate.bat
```

**Verify - both systems**

```bash
python -c "import sys, sysconfig; print(sys.version.split()[0]); print(sys.executable); print(sysconfig.get_platform())"
```

The version should be **3.12.x or 3.13.x**, and the executable path should be
inside this project's `.venv-msba265-ev-purchases` folder. On Windows the last
line should read `win-amd64`. Continue to step 4.
Activate the venv again in each new terminal.

If an existing venv uses another version, close its app/notebook kernels,
run `deactivate`, rename the venv folder to a backup, and recreate it with a
supported Python above. Keep the backup until the replacement works.
Installing packages does not change the venv's Python version.

### 4. Install or update the packages

With the venv active, use pip to read the dependency list and install prebuilt packages:

```bash
python -m pip install --only-binary=:all: -r requirements.txt
python -m pip check
```

Expected: **No broken requirements found**. To update an existing venv after
receiving new project files, activate it and repeat this step.

If the install cannot find a compatible pandas build on Windows, repeat the
version and platform check in step 3. If needed, rename the existing venv and
recreate it with x64 Python 3.13. If the error persists, record the output of
step 3, `python -m pip --version`, and the full install error for the P2 review.

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

Run EDA first, then modeling:

```bash
python ev_project/run_eda.py
python ev_project/run_modeling.py
```

**Allow time to finish:** Each command may take a few minutes, depending on your
computer. After `Executing notebook...` or `Executing modeling notebook...`, the
terminal may stay quiet while the notebook runs; this alone does not mean an
error. Keep the terminal open and wait for **Saved executed notebook** and the
command prompt to return before running the next command.

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
The expected split sizes are **468,065 training**, **100,300 validation**, and
**100,300 holdout**. The holdout ranking should find **4,246 Yes labels in the
top 5,015 profiles** (84.67%, 4.85 times the expected share for random selection).
At the primary 10% capacity, expect **7,902 Yes labels in 10,030 profiles**
(78.78%, 4.51 times the random share). These are results on synthetic labels.

To run interactively instead, install VS Code's **Python** and **Jupyter**
extensions, select this project's venv with **Select Kernel**, and **Run All** in
notebook 01 followed by notebook 02.

### 7. Check the saved model and tests

The first command loads the bundled model and scores the included demo in a
fresh Python process. It also works directly after step 4:

```bash
python -m ev_project.check_saved_model
python -m unittest discover -s tests -v
```

Expect **Logistic Regression**, **500 ranked profiles**, **50 shortlisted**, and
first profile **668721** (Yes score about **0.930806**). The command compares two
loads of the same artifact and should report a maximum score difference of **0**.
The tests should report **Ran 53 tests** and **OK**.

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
