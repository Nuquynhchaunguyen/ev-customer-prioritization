# EV Customer Prioritization

**Nu Quynh Chau Nguyen | MSBA265 | Fall 2026**

**Which customers should an EV dealership prioritize for contact when its outreach capacity is limited?**

This project uses the professor-approved [Kaggle competition dataset](https://www.kaggle.com/competitions/playground-series-s6e9)
to rank customer profiles for staff review. If a dealership can contact only
50 people, the tool produces a shortlist of 50 profiles and editable conversation
guides based on their recorded driving and charging information.

## P1 submission

**Read the [combined P1 PDF](docs/P1/EV_Customer_Prioritization_P1.pdf).**
All P1 content is together in this five-page document:

| Required content | Where to find it |
|---|---|
| Topic examples | Page 2: six analytics topics, a realistic example for each, and the decision it supports |
| Single-page infographic with credible citations | Page 3: history, industry use, value/costs, and connection to this project; full references on page 5 |
| Project proposal | Page 1: stakeholder problem, focused question, intended use, proposed method, success measure, and repository link; dataset details on page 4 |
| Initial exploration and interpretation | Page 4: file sizes, key fields, quality checks, class-balance chart, interpretation, and majority-class baseline |
| Feasibility and next steps | Page 5: feasibility, remaining unknowns, planned evaluation, and next steps |

The Assignment Validation form stays local and will be completed after independent
peer review, then submitted separately on Canvas.

P1 is the upcoming submission. P2 follows two weeks later. The P2 paper and slides
are drafts and will be updated after the professor's P1 feedback. Final delivery
will be prepared later.

## Project map

```text
ev-customer-prioritization/
├── README.md           Project overview and Mac/Windows setup and run instructions
├── docs/
│   ├── P1/             Combined P1 proposal PDF
│   ├── P2/             Draft technical paper, stakeholder slides, and extension plan
│   └── reports/        Notebook HTML reports, charts, tables, and checks
├── notebooks/          01: data understanding; 02: modeling and evaluation
├── ev_project/         Dataset download, preparation, training, and scoring code
├── data/               Included demo; generated cleaned data and splits stay local
├── models/             Saved preprocessing/model bundle and readable metadata
├── app.py              Streamlit ranking and customer conversation interface
├── requirements.txt    Pinned Python package dependencies
├── tests/              Input, model, ranking, and app checks
└── .gitignore          Excludes full data, credentials, caches, and local environments
```

The code and saved model are shared working materials for P2. P1 can be reviewed
by reading its combined PDF; a Streamlit demonstration is not needed for P1.
The full raw CSVs are downloaded beside `app.py` during setup and stay off GitHub.

## Method and current working materials

The main method is **Logistic Regression**, following the professor's
[classification materials](https://github.com/shylasolis/machine-learning-pathway/blob/main/Project_02_Classification/assignment_instructions.md).
It calculates a weighted score and applies the sigmoid function:

```text
z = beta_0 + beta_1*x_1 + ... + beta_p*x_p
P(Will_Buy_EV = Yes) = 1 / (1 + exp(-z))
```

Numeric inputs are standardized and categories are one-hot encoded inside the
saved pipeline. The implementation fits the weights with L-BFGS and L2
regularization. Ranking probabilities selects exactly K profiles; a 0.5 cutoff
for assigning Yes/No is a separate decision. Notebook 02 compares a Dummy
baseline, Logistic Regression, and a Decision Tree using separate evaluation data.

- [Notebook 01: data understanding and cleaning](notebooks/01_data_understanding_and_cleaning.ipynb).
- [Notebook 02: modeling and prioritization](notebooks/02_modeling_and_prioritization.ipynb).
- [Draft P2 technical paper](docs/P2/EV_Customer_Prioritization_Technical_Report.pdf).
- [Draft P2 stakeholder PowerPoint](docs/P2/EV_Customer_Prioritization_Stakeholder_Presentation.pptx).
- [Data dictionary](docs/reports/tables/data_dictionary.csv) and [model metadata](models/model_metadata.json).

FastAPI, Docker, and MLOps are described in the [P2 extension plan](docs/P2/extension_plan.json).
They are planned and are not completed features. Individual P1 feedback and
independent peer reproduction are still needed before the P2 submission.

## Start here: run the project on Mac or Windows

| Your goal | Steps |
|---|---|
| Read P1 and existing results | Open the P1 PDF or saved notebooks; Python is not needed |
| Reproduce the analysis | Steps 1-9 in order |
| Try the included app demo | Steps 1-4, then 9-10; full data and retraining are not needed |

You need Python **3.13** (recommended) or **3.12** and internet access for package
installation. The supplied fitted model was produced with Python 3.13.9.
VS Code is used below to open the folder and type commands. Anaconda is optional.
A Kaggle account is needed only for the full-data download in step 6.

**There are two downloads:** GitHub supplies the project code, saved model,
reports, and small demo. Kaggle API supplies the full competition CSVs. Downloading
the GitHub ZIP does not download the full dataset.

### 1. Download the project code: choose one of two options

- **Option A - Git clone:** download the files with their version history.
- **Option B - Download ZIP:** download and extract the files in your browser.

Both options are shown below. Complete one option, then continue to step 2.

#### Option A - Git clone

Install [Git](https://git-scm.com/downloads) if needed, then reopen your terminal.
On Mac, press **Command+Space**, type **Terminal**, and press Enter. On Windows,
open **PowerShell** from the Start menu. Run one line at a time:

**Mac**

```bash
git --version
cd "$HOME/Documents"
git clone https://github.com/Nuquynhchaunguyen/ev-customer-prioritization.git
cd ev-customer-prioritization
git rev-parse HEAD
```

**Windows - PowerShell**

```powershell
git --version
cd ([Environment]::GetFolderPath('MyDocuments'))
git clone https://github.com/Nuquynhchaunguyen/ev-customer-prioritization.git
cd ev-customer-prioritization
git rev-parse HEAD
```

**Check:** a new `ev-customer-prioritization` folder appears in Documents. The last
command prints the downloaded version's identifier, called a commit. Record it
for reproduction. If that folder already exists, use another empty location.

#### Option B - Download ZIP

1. Open the [repository home page](https://github.com/Nuquynhchaunguyen/ev-customer-prioritization).
2. Click the green **Code** button, then **Download ZIP**, or use the
   [direct ZIP download](https://github.com/Nuquynhchaunguyen/ev-customer-prioritization/archive/refs/heads/main.zip).
3. Open Downloads. **Mac:** double-click the ZIP. **Windows:** right-click it,
   choose **Extract All**, then **Extract**.
4. Move the extracted folder, usually `ev-customer-prioritization-main`, to a
   permanent location such as Documents before creating its environment.

**Check:** open the extracted folder and find `README.md`, `app.py`,
`requirements.txt`, and `models`. On Windows, work in the extracted folder,
not inside the ZIP. You do not need to download files one by one.

### 2. Open the whole project folder and its terminal

1. Install [Visual Studio Code](https://code.visualstudio.com/download) if needed.
2. In VS Code choose **File > Open Folder** (or **File > Open** on Mac).
3. Select the project folder containing `README.md` and `app.py`.
4. Choose **Terminal > New Terminal**. Use **zsh/bash** on Mac or **PowerShell**
   on Windows. The terminal is the panel where you type commands.
5. Run `ls` on Mac or `dir` on Windows. Confirm that `README.md`, `app.py`,
   `requirements.txt`, and `ev_project` are listed.

All following commands run from this main folder. Copy only the commands inside
each box, one line at a time, and wait for each to finish. Do not enter them in a
notebook cell or at Python's `>>>` prompt. You can also use a system terminal:
on Mac, type `cd` followed by a space and drag the folder into Terminal; on
Windows, open the folder in File Explorer, type `powershell` in its address bar,
and press Enter.

### 3. Install or check Python

Install a Python 3.13 release from [Python for Mac](https://www.python.org/downloads/macos/)
or [Python for Windows](https://www.python.org/downloads/windows/) if needed.
On Windows, enable the Python launcher and **Add Python to PATH** if offered.
After installing, close and reopen VS Code and its terminal.

**Mac**

```bash
python3.13 --version
```

**Windows - PowerShell**

```powershell
py -3.13 --version
```

**Check:** the result is `Python 3.13.x`. If using Python 3.12, substitute
`python3.12` on Mac or `py -3.12` on Windows in this check and the environment
creation line below. If Windows cannot find `py`, check `python --version`;
use `python` for the creation line only if it reports 3.12 or 3.13.

### 4. Create or update the named environment

A **venv** keeps this project's packages separate from other class projects.
Every computer creates its own `.venv-msba265-ev-purchases` folder. A copied or
moved venv should be recreated in its new location.

**First setup - Mac**

```bash
python3.13 -m venv --prompt msba265-ev-purchases .venv-msba265-ev-purchases
./.venv-msba265-ev-purchases/bin/python -m pip install -r requirements.txt
./.venv-msba265-ev-purchases/bin/python -m pip check
```

**First setup - Windows PowerShell**

```powershell
py -3.13 -m venv --prompt msba265-ev-purchases .venv-msba265-ev-purchases
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m pip check
```

**Check:** the last command reports **No broken requirements found**.
Installation can take several minutes. These commands call the venv directly,
so activation is optional; you do not need to see its name in the terminal prompt.

**Already have this environment?** Stop the app with Ctrl+C and use these update
commands instead of creating it again. They match the pinned package versions
in `requirements.txt`; they do not update Python or retrain the model.

**Update - Mac**

```bash
./.venv-msba265-ev-purchases/bin/python --version
./.venv-msba265-ev-purchases/bin/python -m pip install --upgrade -r requirements.txt
./.venv-msba265-ev-purchases/bin/python -m pip check
```

**Update - Windows PowerShell**

```powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" --version
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m pip install --upgrade -r requirements.txt
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m pip check
```

Restart open notebook kernels after updating packages. If the venv is missing,
will not start, or uses another Python minor version, close its app/kernels,
rename only that venv folder to a backup, and repeat first setup. Keep the backup
until the replacement works. See [Python's venv guidance](https://docs.python.org/3.13/library/venv.html).

For shorter commands you may optionally activate it: Mac uses
`source .venv-msba265-ev-purchases/bin/activate`; PowerShell uses
`.\.venv-msba265-ev-purchases\Scripts\Activate.ps1`. If PowerShell blocks
activation, use the full paths above. Anaconda's additional `(base)` label is fine.

### 5. Understand the data source and file roles

Source: [Predicting Electric Vehicle Purchases, Kaggle Competitions](https://www.kaggle.com/competitions/playground-series-s6e9).
One row represents one synthetic profile, with `id` and 13 predictors covering
age, income, commuting, car ownership, charging access, and related characteristics.
The target is `Will_Buy_EV`, mapped as No=0 and Yes=1.

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `train.csv` | 668,665 | 15 | Labeled development and internal evaluation data |
| `test.csv` | 286,571 | 14 | Unlabeled profiles to score; no evaluation labels |
| `sample_submission.csv` | 286,571 | 2 | Kaggle prediction-format template; not customer input |
| `data/demo/customer_profiles_demo.csv` | 500 | 14 | Included unlabeled profiles for the app/load check |

The full three Kaggle CSVs are downloaded to the main project folder beside
`app.py`. The demo is included in GitHub. Read the
[data dictionary](docs/reports/tables/data_dictionary.csv) for exact names and values.
Use your own Kaggle account and follow the competition's access rules and license.
Credentials and full raw files are excluded from GitHub.

The profiles have no customer names, phone numbers, purchase dates, or observed
outreach responses. Country, income definition, charging-station radius, and
purchase timeframe are not supplied. Scores estimate the synthetic dataset label;
they do not demonstrate real purchases or sales caused by contacting someone.

### 6. Download the full dataset through Kaggle API

If all three raw CSVs already exist beside `app.py`, continue to step 7.
Otherwise:

1. Sign in and open the [competition](https://www.kaggle.com/competitions/playground-series-s6e9).
   Complete any account access or rule-acceptance steps Kaggle requires.
2. Open [Kaggle API settings](https://www.kaggle.com/settings/api) and create a
   legacy API key (`kaggle.json`, containing your username and key).
3. Leave `kaggle.json` in your Downloads folder, outside the project.
4. Run the command for your computer:

**Mac**

```bash
./.venv-msba265-ev-purchases/bin/python ev_project/download_data.py
```

**Windows - PowerShell**

```powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" ev_project/download_data.py
```

**Check:** the terminal prints **Downloaded train.csv, test.csv, and
sample_submission.csv.** The files appear beside `app.py`. The downloader calls
the Kaggle API; notebooks then read these downloaded local CSVs.

If all three already exist, it reports **All three raw CSVs already exist;
nothing was overwritten.** If only some exist, it stops for you to review the
folder. It also searches `~/.kaggle/kaggle.json`. For another credential location,
append `--credentials` and the quoted full path to the command. Already configured
`KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` with `KAGGLE_KEY`, are also supported.
Never include a credential file or key in the repository.

### 7. Run the analysis in order

Run notebook 01 (EDA) first and wait for it to finish. Then run notebook 02
(modeling), which uses the cleaned training data produced by EDA.

**Mac**

```bash
./.venv-msba265-ev-purchases/bin/python ev_project/run_eda.py
./.venv-msba265-ev-purchases/bin/python ev_project/run_modeling.py
```

**Windows - PowerShell**

```powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" ev_project/run_eda.py
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" ev_project/run_modeling.py
```

**Check:** each finishes with **Saved executed notebook** and its HTML report
path. A traceback or interruption means that run is incomplete. Runs can take
several minutes. The runners use the venv you called and create a temporary
notebook kernel; a separate Jupyter server is not needed.

Rerunning replaces generated notebook outputs, HTML reports, cleaned files,
evaluation results, and model artifacts. Raw CSVs stay unchanged.

To work interactively, install VS Code's Microsoft **Python** and **Jupyter**
extensions. Open notebook 01, choose **Select Kernel > Python Environments** and
select this project's venv, then **Run All**. Repeat for notebook 02. The kernel's
executable is `.venv-msba265-ev-purchases/bin/python` on Mac or
`.venv-msba265-ev-purchases\Scripts\python.exe` on Windows.

### 8. Check the expected outputs

| Run | Expected files and checks |
|---|---|
| EDA | Executed notebook 01; `docs/reports/01_data_understanding_and_cleaning.html`; cleaned CSVs in `data/processed/`; tables and eight charts in `docs/reports/` |
| Modeling | Executed notebook 02; `docs/reports/02_modeling_and_prioritization.html`; `models/model_bundle.joblib`; `models/model_metadata.json`; evaluation tables in `docs/reports/modeling/`; split membership in `data/splits/` |

EDA retains all 668,665 labeled and 286,571 unlabeled rows, with zero source rows
or predictor values removed or changed. It maps the training target to 0/1.
The modeling split has 468,065 training, 100,300 validation, and 100,300 holdout
profiles, with seed 42. Preprocessing learns from training rows during comparison;
selection uses validation Lift@10%, then ROC-AUC to break a tie. The selected
pipeline is refitted on training plus validation before holdout evaluation.

The saved run selects Logistic Regression. At holdout K=10,030 (10%),
Precision@K is approximately **78.78%** and Lift@K **4.51**. Precision is the Yes-label
share among the selected profiles; lift compares it with the holdout Yes-label
share. These are synthetic-data ranking results, not sales conversion rates.
Full-data EDA was viewed before splitting, so the holdout is an internal check,
not an entirely unseen external dataset. Small numerical differences across
computers are possible.

Open the HTML files locally in a browser through Finder/File Explorer. If an
editor shows HTML code, use **Open With > your browser**. GitHub may show HTML
source instead of the rendered report. Saved notebook outputs can also be read
on GitHub without rerunning them.

### 9. Load the saved model in a fresh run and test it

The fitted bundle is included. This check loads it in a new Python process,
scores the included 500-profile demo, and confirms a 50-profile shortlist.
It does not train a model or need Kaggle credentials.

**Mac**

```bash
./.venv-msba265-ev-purchases/bin/python -c "from pathlib import Path; from ev_project.schema import read_profile_csv; from ev_project.scoring import load_bundle, score_profiles; profiles = read_profile_csv(Path('data/demo/customer_profiles_demo.csv').read_bytes()); ranked, shortlist, warnings = score_profiles(profiles, load_bundle('models/model_bundle.joblib'), 50); assert len(ranked) == 500 and len(shortlist) == 50; print('Profiles:', len(ranked), 'Shortlist:', len(shortlist), 'Warnings:', len(warnings))"
```

**Windows - PowerShell**

```powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -c "from pathlib import Path; from ev_project.schema import read_profile_csv; from ev_project.scoring import load_bundle, score_profiles; profiles = read_profile_csv(Path('data/demo/customer_profiles_demo.csv').read_bytes()); ranked, shortlist, warnings = score_profiles(profiles, load_bundle('models/model_bundle.joblib'), 50); assert len(ranked) == 500 and len(shortlist) == 50; print('Profiles:', len(ranked), 'Shortlist:', len(shortlist), 'Warnings:', len(warnings))"
```

**Expected:** `Profiles: 500 Shortlist: 50 Warnings: 0`.
The bundle includes the fitted preprocessing steps and classifier. Load only the
trusted model supplied with this project. Uploaded CSVs are customer inputs.

Then run the automated checks:

**Mac**

```bash
./.venv-msba265-ev-purchases/bin/python -m unittest discover -s tests -v
```

**Windows - PowerShell**

```powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m unittest discover -s tests -v
```

**Expected:** **Ran 53 tests** and **OK**. The checks cover data separation,
input rules, ranking, model reload, and conversation editing/export. Record the
commit, operating system, Python version, commands, and actual outcomes for
independent reproduction.

### 10. Try the Streamlit app

After installing packages, you can use the supplied model and demo without
running the full analysis. Start the local app from the main folder:

**Mac**

```bash
./.venv-msba265-ev-purchases/bin/python -m streamlit run app.py
```

**Windows - PowerShell**

```powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m streamlit run app.py
```

Open the printed **Local URL**, usually `http://localhost:8501`, in your browser.
Keep that terminal running. Press **Ctrl+C** inside it to stop the app; repeat
this command to start it again. VS Code's normal Run Python File button does not
start the Streamlit interface.

1. Select **Try the demo**; it loads 500 profiles. Set contact capacity K to **50**.
2. Click **Create shortlist**. Check that the shortlist has 50 profiles.
3. Review ID, estimated Yes probability, gender, age, and income. Use **Customer
   information to show** to add driving and charging fields.
4. Under **Prepare a customer conversation**, select a priority rank. Review and
   edit the five-part guide: opening, needs questions, discussion points,
   concerns to explore, and next step. Click outside a text box to apply edits.
5. Download the selected guide as TXT, or choose **Prepare guides for the
   shortlist** and download the guide CSV. K=50 produces 50 customer rows.
   After changing K or editing a guide, prepare the CSV again.
6. Download the ranked shortlist. It includes all 13 profile fields even when
   fewer columns are visible in the app.

To use your own CSV, select **Upload a CSV**. Use `test.csv` or the demo file,
with a unique `id` and the same 13 predictor columns/categories as the dictionary.
`sample_submission.csv` is an output template and cannot be used here. Upload
limits are 50 MB and 300,000 profiles; bulk guide exports support up to 1,000.

Uploads and edits stay in session memory; download them before closing the
session or loading another file. Editing a guide does not change scores or ranks.
Guides use recorded driving/charging facts, without inferring preferences from
gender or age or affordability from income. They are editable conversation aids,
not explanations of the model score or proof of sales impact. Staff review the
wording and confirm needs, vehicle details, prices, and eligibility. The app does
not contact customers or book appointments automatically.

## Get later updates

For a Git copy, stop the app, run `git status`, and if the working tree is clean,
run `git pull --ff-only`. Keep and resolve local changes before pulling if Git
reports modifications. Then use the environment update commands in step 4.
For ZIP, download/extract a new copy into a new folder and create its own venv;
keep the old copy until the new one works. Do not copy the old environment.

## Troubleshooting and verification

| What you see | What to check |
|---|---|
| `requirements.txt` or `ev_project` not found | Open the main folder containing `README.md` and `app.py`, then its terminal |
| `python3.13` / `py` not found | Complete step 3 and reopen VS Code; use the documented 3.12/3.13 fallback |
| `ModuleNotFoundError` | Install `requirements.txt` using the named venv in step 4 |
| PowerShell blocks activation | Use the full `Scripts\python.exe` commands; activation is optional |
| Kaggle HTTP 401/403 | Check your own credentials, competition access, and accepted rules |
| Some raw CSVs already exist | Review and back up the partial set before downloading; the script does not overwrite it |
| Cleaned training file not found | Run EDA before modeling |
| HTML appears as code | Open the local HTML in a browser |
| Notebook uses the wrong Python | Select this project's venv as the notebook kernel |

The analysis and app have been executed on macOS; automated workflow checks pass.
Windows PowerShell instructions are provided but have not been executed on a
Windows host. Independent peer reproduction remains pending. The runners and
checks are shared for the reviewer to verify the submitted version.
