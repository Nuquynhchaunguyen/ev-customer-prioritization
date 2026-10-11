# EV Customer Prioritization

**Nu Quynh Chau Nguyen | MSBA265 | Fall 2026**

Which customer profiles should a dealership prioritize for EV outreach when it
can contact only **K** people? This project uses the professor-approved Kaggle
competition dataset for data understanding, model comparison, customer ranking,
and a local Streamlit application.

The data contain 668,665 synthetic labeled profiles; about 17.46% have a Yes
label. The app ranks uploaded profiles using the saved preprocessing-and-model
pipeline. Its scores estimate the supplied Yes-label probability in this dataset,
not a guaranteed real purchase, sales uplift, or revenue forecast.

## P1 submission and peer review

Repository: [https://github.com/Nuquynhchaunguyen/ev-customer-prioritization](https://github.com/Nuquynhchaunguyen/ev-customer-prioritization)

- [Combined P1 proposal PDF](submissions/P1/EV_Customer_Prioritization_P1.pdf): business question, six topic examples, one-page infographic with credible sources, data profile, interpreted EDA, baseline and feasibility, and next steps.
- Assignment Validation is kept locally and submitted separately to the professor. Completed or prefilled forms are excluded from this repository; peer checks, dates, and signatures remain pending until independent replication.
- [P1 handoff checklist](submissions/P1/README.md): required files, peer workflow, and remaining human verification.
- [Optional extension plan](submissions/P1/bonus_plan.json): FastAPI, Docker, and MLOps are planned for P2/final delivery. None is claimed as completed or awarded.

P1 is the first of three separate graded submissions (20/30/50 points).
The technical prototype already extends beyond P1, so all current runnable work
is included. Independent peer verification is still pending. Automated checks
and a working app do not substitute for the course form and real peer signatures.

## Start reading

- [Data understanding and cleaning notebook](notebooks/01_data_understanding_and_cleaning.ipynb): dataset definitions, quality checks, and EDA.
- [Modeling and prioritization notebook](notebooks/02_modeling_and_prioritization.ipynb): fixed splits, three candidates, validation selection, holdout results, and model artifacts.
- [EDA HTML reading copy](reports/01_data_understanding_and_cleaning.html): read the first notebook without Python.
- [Modeling HTML reading copy](reports/02_modeling_and_prioritization.html): read the modeling notebook without Python.
- [Data dictionary](reports/tables/data_dictionary.csv): fields, observed values, and interpretation limits.
- [Model metadata](models/model_metadata.json): selected model, split counts, package versions, and evaluation records.

The executed notebooks contain saved outputs. Open the HTML reading copies in a
browser rather than as source code in a text editor. To reproduce results or run
the app, follow the instructions for your operating system below.

## Project map

~~~text
project-folder/
├── train.csv                                  Raw labeled profiles
├── test.csv                                   Raw unlabeled profiles
├── sample_submission.csv                      Kaggle submission template
├── app.py                                     Streamlit customer-ranking app
├── ev_project/                                Shared schema, ranking, and modeling logic
├── notebooks/
│   ├── 01_data_understanding_and_cleaning.ipynb EDA and cleaning notebook
│   └── 02_modeling_and_prioritization.ipynb     Modeling and prioritization notebook
├── data/
│   ├── processed/                             Cleaned EDA data
│   ├── splits/split_membership.csv             Frozen ID-to-split membership
│   └── demo/customer_profiles_demo.csv         Small unlabeled app demonstration
├── models/
│   ├── model_bundle.joblib                    Pipeline and metadata used by the app
│   ├── selected_pipeline.joblib               Exact evaluated final pipeline
│   └── model_metadata.json                    Readable training and evaluation records
├── reports/
│   ├── 01_data_understanding_and_cleaning.html EDA reading copy
│   ├── 02_modeling_and_prioritization.html     Modeling reading copy
│   ├── figures/                              EDA charts
│   ├── tables/                               Dictionary, audits, and EDA tables
│   ├── modeling/                             Model metrics, ranking tables, and charts
│   └── data_manifest.json                    EDA file hashes and run information
├── scripts/
│   ├── download_data.py                      Kaggle API download when needed
│   ├── run_eda.py                            Execute EDA and export results
│   ├── run_modeling.py                       Execute modeling and export results
│   └── build_modeling_notebook.py             Rebuild the explanatory notebook source
├── tests/                                    Input-validation and ranking checks
├── .vscode/settings.json                     Portable environment settings
├── .venv-msba265-ev-purchases/                 Your local Python environment
├── requirements_eda.txt                       Pinned EDA dependencies
├── requirements_project.txt                   EDA plus modeling and app dependencies
└── README.md                                 Setup and execution instructions
~~~

**Reading sequence:** business question → data dictionary → quality checks →
EDA findings → split roles → validation comparison → final holdout evaluation →
app workflow and limitations.

## Start here: run the complete project on Mac or Windows

You need Python **3.13** (recommended) or **3.12**, and internet access to install
packages. Kaggle authentication is needed only if you must download the raw CSVs.
If the CSVs are included, notebooks and local app scoring run offline after
package installation. The app needs saved model artifacts; step 6 creates them.
Anaconda and Git are optional; VS Code is optional for the terminal workflow.

**On the author's Mac, the named environment is already updated and the saved
model is ready.** To use the current app, open a terminal in this project folder
and go directly to step 7. Retraining is optional when using the supplied model.

A terminal is a window where you type commands. Copy only the commands inside
each box, run one line at a time, and wait for it to finish. Use **Terminal with
bash/zsh on macOS** or **PowerShell on Windows** for these examples.

### 1. Obtain the complete project

Clone [the project repository](https://github.com/Nuquynhchaunguyen/ev-customer-prioritization), extract a complete project ZIP, or use an existing checkout. Keep the
notebook, scripts, requirements file, and raw CSVs together as shown above.
If the CSVs are absent, step 5 explains how to download them through Kaggle API.

Each computer must create its own environment. Exclude the venv when sharing a ZIP;
copying a Mac venv to Windows, or moving an existing venv to a different folder,
does not create a portable environment. Recreate it using step 4.

### 2. Open a terminal in the project folder

- **Mac:** open Terminal. Type cd followed by a space, drag the project folder
  from Finder into Terminal, and press Enter. Run ls and confirm README.md,
  requirements_project.txt, and scripts are listed.
- **Windows:** open the project folder in File Explorer. Click the address bar,
  type powershell, and press Enter. Run dir and confirm the same files are listed.
- **VS Code on either system:** choose File > Open Folder and open the folder
  containing README.md, then Terminal > New Terminal. On Windows, select PowerShell.

All subsequent commands run from this project folder, not from notebooks or
scripts. A folder name containing spaces is fine.

### 3. Install or check Python

Install a Python 3.13 release from [Python for macOS](https://www.python.org/downloads/macos/)
or [Python for Windows](https://www.python.org/downloads/windows/), or use an
existing Python 3.13/3.12 installation. On Windows, enable the Python launcher
and Add Python to PATH if those options appear. Reopen the terminal after installation.

**Mac**

~~~bash
python3.13 --version
~~~

**Windows — PowerShell**

~~~powershell
py -3.13 --version
~~~

Expect Python 3.13.x. For an installed Python 3.12, replace python3.13 with
python3.12 on Mac, or py -3.13 with py -3.12 on Windows in the environment-creation
commands. Other Python minor versions have not been verified for this project.

### 4. Create the named venv and install packages

The environment folder is **.venv-msba265-ev-purchases** and its activated prompt
is **msba265-ev-purchases**. For first setup, use the commands for your computer.
If this environment already exists and works with Python 3.12/3.13, skip creation
and follow **Update an existing environment** below.

These commands call the environment's Python directly. Activation is optional,
so Windows users can run them even when PowerShell blocks activation scripts.

**Mac**

~~~bash
python3.13 -m venv --prompt msba265-ev-purchases .venv-msba265-ev-purchases
./.venv-msba265-ev-purchases/bin/python -m pip install --upgrade pip
./.venv-msba265-ev-purchases/bin/python -m pip install -r requirements_project.txt
./.venv-msba265-ev-purchases/bin/python -m pip check
~~~

**Windows — PowerShell**

~~~powershell
py -3.13 -m venv --prompt msba265-ev-purchases .venv-msba265-ev-purchases
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m pip install -r requirements_project.txt
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m pip check
~~~

The last command should report **No broken requirements found**. If py is not
available on Windows, first check python --version. If it reports 3.13.x or
3.12.x, use python -m venv --prompt msba265-ev-purchases .venv-msba265-ev-purchases
for the creation line; keep the remaining commands unchanged.

### 5. Obtain the raw CSVs if needed

If train.csv, test.csv, and sample_submission.csv already exist at the project
root, proceed to step 6. They were originally downloaded through Kaggle API;
the notebook reads these local files during analysis.

To download on another computer, obtain your own API credentials from
[Kaggle API settings](https://www.kaggle.com/settings/api) and satisfy the
competition's access requirements. For a legacy key, save kaggle.json in your
user's Downloads folder or .kaggle folder, then run:

**Mac**

~~~bash
./.venv-msba265-ev-purchases/bin/python scripts/download_data.py
~~~

**Windows — PowerShell**

~~~powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" scripts/download_data.py
~~~

For a legacy credential file elsewhere, append --credentials and its quoted
full path. KAGGLE_API_TOKEN or the pair KAGGLE_USERNAME and KAGGLE_KEY environment
variables are also supported. The downloader returns without overwriting when
all three raw files exist; if only some exist, it stops for you to review the folder.

Credentials belong to each user's account and are never included in the shared
project or ZIP. The API endpoint is defined in scripts/download_data.py.

### 6. Reproduce EDA and modeling from the terminal

**Mac**

~~~bash
./.venv-msba265-ev-purchases/bin/python scripts/run_eda.py
./.venv-msba265-ev-purchases/bin/python scripts/run_modeling.py
~~~

**Windows — PowerShell**

~~~powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" scripts/run_eda.py
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" scripts/run_modeling.py
~~~

Run one command at a time. Wait for each **Saved executed notebook** message.
The modeling run trains the candidates, selects with validation data, refits the
selected pipeline, evaluates the holdout, and saves model artifacts. It takes
longer than opening a notebook with saved results. A traceback means that run did
not complete successfully.

Each runner uses the Python that launched it to create a temporary notebook
kernel and execute all cells. No notebook server or global kernel registration
is required. EDA regenerates its processed CSVs, tables, figures, and manifest;
modeling regenerates model artifacts and evaluation records. Rerunning replaces
generated results while keeping raw CSVs unchanged.

The EDA-only installation remains available through requirements_eda.txt. Use
requirements_project.txt for the complete workflow, including modeling and the app.

Check these expected outputs:

- Training file: 668,665 rows and 15 columns; test file: 286,571 rows and 14 columns.
- Thirteen candidate predictors; ID is excluded from model predictors.
- Zero rows removed; training target No=0, Yes=1; test remains unlabeled.
- Eight charts under reports/figures and saved notebook outputs.
- The HTML reading copies open in a browser with tables and charts available.
- Modeling saves models/model_bundle.joblib, models/selected_pipeline.joblib,
  models/model_metadata.json, split membership, and validation/holdout reports.
- data/demo/customer_profiles_demo.csv contains unlabeled example app inputs.

### 7. Start the Streamlit application

After model artifacts exist, run the following from the project root. This starts
a local browser application. No Kaggle API credential is needed to score an
uploaded file.

**Mac**

~~~bash
./.venv-msba265-ev-purchases/bin/python -m streamlit run app.py
~~~

**Windows — PowerShell**

~~~powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m streamlit run app.py
~~~

Use the local URL printed in the terminal if a browser does not open. Keep that
terminal running while using the app; press Ctrl+C there to stop the local server.
This command does not publish the app to the internet.

1. Start with data/demo/customer_profiles_demo.csv or your own compatible,
   unlabeled CSV. The demo has 500 Kaggle test profiles and no outcome labels.
2. Upload the CSV. Required names are `id` plus the 13 fields listed in the
   notebook data dictionary. Keep exact field names and supported categories;
   identifiers must be present and unique.
3. Resolve any reported input errors before scoring. The app checks its schema,
   category membership, numeric domains, and missing or malformed values.
4. Choose **K**, the number of profiles you can contact, up to the batch size.
5. Review the highest scores. The table shows Gender, Age, and Annual income by
   default; use **Customer information to show** to add charging, commute, or other
   recorded profile fields. Scores estimate the synthetic Yes label, not an
   established real-customer purchase probability.
6. Under **Prepare a customer conversation**, choose a priority rank to inspect
   that customer. The view shows demographics, driving/charging context, and
   suggested questions about needs and budget. Questions use recorded transport
   facts, without inferring preferences from gender or age, or affordability from
   income. They are conversation aids, not explanations of the model score.
7. Download the contact shortlist or all ranked profiles. Both exports contain
   rank, ID, Yes probability (0–1), and all 13 normalized profile fields, regardless
   of which columns are visible. Profile facts are matched by ID after ranking;
   target labels and unrelated extra upload columns are excluded.

These synthetic profiles have no names or phone numbers. A real contact workflow
would need a separate authorized customer contact source keyed by customer ID.
The app does not invent contact details or initiate customer calls.

The app loads the trusted project model bundle. Do not replace it with a joblib
or pickle file obtained from an unknown source. Uploaded CSVs are inputs, not
model files.

### 8. Read or run interactively in VS Code

1. Install the Microsoft **Python** and **Jupyter** extensions if needed.
2. Open the whole project folder. Run Python: Select Interpreter from the Command
   Palette (Command+Shift+P on Mac; Ctrl+Shift+P on Windows).
3. Select .venv-msba265-ev-purchases. If asked for its executable, use
   .venv-msba265-ev-purchases/bin/python on Mac or
   .venv-msba265-ev-purchases\Scripts\python.exe on Windows.
4. Open either notebook in notebooks/. Choose Select Kernel > Python
   Environments, select that same environment, and choose Run All. Run Notebook
   01 before Notebook 02 when reproducing the complete project.

The saved kernel display name is MSBA265 | EV Purchases. Select your own local
environment explicitly; selecting the editor interpreter does not always select
the notebook kernel. Workspace settings point to the environment folder, which
allows the same configuration on both systems. If an older interpreter was
already selected, use Select Interpreter to change it.

## Update an existing environment

Use this after obtaining a revised requirements_project.txt, or when the venv exists
but project packages have not been installed or refreshed. These commands keep
the pinned direct package versions; they **do not upgrade Python**. Run them
from the project root using your own environment.

**Mac**

~~~bash
./.venv-msba265-ev-purchases/bin/python --version
./.venv-msba265-ev-purchases/bin/python -m pip install --upgrade pip
./.venv-msba265-ev-purchases/bin/python -m pip install --upgrade -r requirements_project.txt
./.venv-msba265-ev-purchases/bin/python -m pip check
./.venv-msba265-ev-purchases/bin/python scripts/run_eda.py
./.venv-msba265-ev-purchases/bin/python scripts/run_modeling.py
~~~

**Windows — PowerShell**

~~~powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" --version
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m pip install --upgrade -r requirements_project.txt
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m pip check
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" scripts/run_eda.py
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" scripts/run_modeling.py
~~~

If CSVs are absent, complete step 5 before the final run command. After updating,
restart an already running notebook kernel before Run All so it uses the refreshed
packages. requirements_project.txt includes requirements_eda.txt and pins direct
modeling/app dependencies; it is not a complete lockfile for every transitive
dependency. Installing packages does not retrain a model: the notebook runner
commands do that. If you only need to refresh packages and launch the saved app,
stop after pip check and use step 7.

### Recreate a missing, moved, or broken venv

Updating packages cannot repair a missing base Python or change the venv's
Python minor version. Recreate the environment when changing from another
Python version, after moving the project, or when its interpreter will not start.

1. Close notebook kernels and terminals using the old venv.
2. In Finder or File Explorer, rename only .venv-msba265-ev-purchases to a backup
   name that is not already used. Keep the notebook, raw data, scripts, and reports.
   If the environment folder is absent, skip this step.
3. Open a new terminal at the project root. Check Python 3.13 or 3.12 as in step 3.
4. Repeat step 4 to create the correctly named venv and reinstall packages.
5. Run step 6, then select the new interpreter and notebook kernel in VS Code.

Keep the backup until the new environment runs successfully. Recreate each
machine's venv from the requirements file instead of copying someone else's
venv. See the [Python venv documentation](https://docs.python.org/3.13/library/venv.html).

## Optional terminal activation

Activation makes the shorter python command use this environment. It is not
required for the direct-environment commands above.

**Mac — bash/zsh**

~~~bash
source .venv-msba265-ev-purchases/bin/activate
~~~

**Windows — PowerShell**

~~~powershell
.\.venv-msba265-ev-purchases\Scripts\Activate.ps1
~~~

If PowerShell blocks Activate.ps1, use the direct python.exe commands above;
no execution-policy change is needed. An activated terminal normally shows
(msba265-ev-purchases). An additional (base) label can appear with Anaconda.
To check which interpreter the shorter command uses, run:

~~~bash
python -c "import sys; print(sys.executable)"
~~~

The path should be inside .venv-msba265-ev-purchases. Type deactivate to leave the
venv. Opening a new terminal may require activation again.

## Data, cleaning, and interpretation

Source: [Predicting Electric Vehicle Purchases — Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e9).
This approved source is in Kaggle Competitions. Playground data are synthetic;
Will_Buy_EV is a supplied dataset label, not documented as a verified transaction
with a known purchase horizon.

The data contain seven numeric and six categorical candidate predictors covering
demographics, income, commuting, vehicle ownership, charging access, subsidies,
environmental concern, and range anxiety. ID is retained for traceability.
sample_submission.csv defines the submission format; its probabilities are not
test labels.

No missing values, duplicate IDs, or checked category/domain defects were found.
Cleaning preserves all rows and predictor values, converts income and
environmental-concern codes losslessly to integer storage, and explicitly maps
the training target to 0/1. No target is added to test. Rare groups and statistical
outliers are retained and discussed. In the modeling pipeline, environmental
concern codes are encoded as categories, so modeling uses six numeric and seven
categorical fields; the storage types described in the EDA remain unchanged.

Approximately 17.46% of training labels are Yes. Income and commute distances
have concentrations at USD 30,000 and 5 km; their origins are not established.
Subgroup label shares and correlations are descriptive associations, not causal
effects or established model feature importance. Definitions such as charging
radius and the outcome horizon are not supplied, and are documented as unknown.

## Modeling decisions and interpretation

The fixed seed-42 stratified split assigns approximately 70% of labeled rows to
training, 15% to validation, and 15% to holdout. Full-data EDA was already viewed
before this split; the holdout is reserved from fitting and selection, but it is
not an entirely unseen external dataset. Kaggle test.csv is unlabeled and is not
used as the internal evaluation holdout.

The candidates are DummyClassifier(strategy="prior"), Logistic Regression
(C=1, lbfgs, max_iter=1000), and Decision Tree (max_depth=6,
min_samples_leaf=200, random_state=42). Categorical fields use one-hot encoding;
Logistic Regression scales numeric fields. Pipeline preprocessing learns only
from training rows during candidate comparison. Initial probability baselines
use no SMOTE or class weighting.

Selection uses highest validation **Lift@10%**, then ROC-AUC to resolve a tie.
If both tie, the fixed simplicity order is Dummy, Logistic Regression, Decision Tree.
We also compare 1%, 5%, and 20% contact capacities. Tied profile scores have a
fixed seed-and-ID hash order that does not use outcome labels. Dummy scores are
constant, so its realized ranking can differ slightly from the expected random
baseline, whose expected precision is the evaluation group's Yes-label share and
whose expected lift is 1.

After selection, the chosen pipeline is freshly fitted on training plus
validation (85%) and evaluated on the holdout (15%). The saved app pipeline is
that exact final fit. **Precision@K** is the Yes-label share among the top K;
**Recall@K** is the share of all Yes labels captured there; **Lift@K** compares
precision with the evaluation group's prevalence. Additional reports include
ROC-AUC, average precision, Brier score, and threshold-0.50 classification
metrics. The score threshold is a diagnostic and does not define the app's top K.

### Current executed results

The split contains **468,065 training**, **100,300 validation**, and **100,300
holdout** profiles. Logistic Regression was selected using validation results:

| Validation candidate | Precision@10% | Lift@10% | ROC-AUC |
|---|---:|---:|---:|
| Dummy baseline | 17.69% | 1.013 | 0.500 |
| Logistic Regression | 79.10% | 4.529 | 0.938 |
| Decision Tree | 78.76% | 4.510 | 0.937 |

The two fitted classifiers are close on this comparison; selection follows the
predefined rule and does not establish a universally superior algorithm.

After refitting Logistic Regression on 568,365 train-plus-validation profiles,
the final holdout results are:

| Contact capacity | K | Precision@K | Lift@K |
|---|---:|---:|---:|
| 1% | 1,003 | 90.13% | 5.161 |
| 5% | 5,015 | 84.67% | 4.848 |
| 10% (primary) | 10,030 | 78.78% | 4.511 |
| 20% | 20,060 | 66.19% | 3.790 |

Holdout ROC-AUC is **0.939**; average precision is **0.741**. At the primary K,
the shortlist captures **45.11%** of the holdout's Yes labels. These are internal
synthetic-data results, not verified dealership conversion rates. The app lets
users choose any integer K within their uploaded batch; unlabeled uploads cannot
provide an observed Precision@K or Lift@K.

Subgroup associations and model scores do not establish causal effects, extra
sales, or ROI. Real deployment needs external labeled validation, review of
whether demographic fields belong in the outreach process, and subgroup checks.
Calibration and drift for real customers remain unestablished. The local app
is a class-project prototype and does not contact customers automatically.

## Reproducibility and verification status

The EDA has been executed on macOS with Python 3.12 and 3.13. The modeling
notebook was executed on macOS with Python 3.13.9: all 12 code cells completed
without errors. The generated model metadata records the package versions,
input hashes, split counts, and evaluation results.

All **30 automated checks** passed, covering disjoint stratified splits,
training-only preprocessing, validation-only selection, exact saved-pipeline
probabilities, CSV errors, outcome-label exclusion, repeatable tied scores,
exact-K output, the Streamlit workflow, and correct ID-based attachment of
customer profile information to ranked scores. The running browser app was also
checked with an uploaded demo CSV; its downloaded 20-row shortlist matched the
saved model's ranks, IDs, and probabilities. The richer profile output was
verified against all 286,571 Kaggle test profiles: rankings and probabilities
were preserved, and all top-50 customer facts matched their source IDs.

To run the automated checks on your computer:

**Mac**

~~~bash
./.venv-msba265-ev-purchases/bin/python -m unittest discover -s tests -v
~~~

**Windows — PowerShell**

~~~powershell
& ".\.venv-msba265-ev-purchases\Scripts\python.exe" -m unittest discover -s tests -v
~~~

Windows PowerShell instructions use the same code with platform-specific paths,
but have not been executed on a Windows host; independent peer verification is
still needed. Each computer creates its own venv from requirements_project.txt.

AI assistance was used to draft code and analysis. The student should review,
explain, and refine the target interpretation, split roles, selection rule,
ranking metrics, and deployment limits before submission.

## Troubleshooting

| Problem | Action |
|---|---|
| Python command not found | Install Python 3.13 or 3.12 and reopen the terminal; use step 4's fallback if the Windows launcher is absent. |
| requirements_project.txt or scripts not found | Run ls on Mac or dir on Windows and move to the project root described in step 2. |
| Environment's Python path does not exist | Create the venv using step 4; use bin/python on Mac and Scripts/python.exe on Windows. |
| Missing package or incompatible installed version | Follow Update an existing environment, then restart the notebook kernel. |
| PowerShell activation blocked | Call the environment's python.exe directly; activation is optional. |
| Notebook cannot import packages | Select the same environment used to install requirements as the notebook kernel. |
| Environment not listed in VS Code | Refresh environment discovery with Python Environments: Refresh All Environment Managers, or select the interpreter path explicitly. |
| Missing CSV or Kaggle access error | Use step 5, your own API credentials, and the competition's access instructions. |
| Command entered at the Python prompt (>>>) gives SyntaxError | Type exit() first, then run the command in Terminal or PowerShell. |
| Installation fails | Check internet access and Python version, retry installation, and retain the full error if it persists. |
| HTML opens as code | Open either reports/*.html reading copy with a browser. |
| App reports missing model artifacts | Install requirements_project.txt and run scripts/run_modeling.py before starting the app. |
| Model cannot load after changing package versions | Restore the pinned requirements; if deliberately changing versions, retrain and evaluate before using the app. |
| Uploaded CSV fails validation | Use the included demo and dictionary to check field names, missing values, categories, numeric domains, and duplicate IDs. |
| Streamlit command not found | Use the environment Python with -m streamlit, as in step 7. |
| Notebook run times out | Use --timeout 600 for run_eda.py or --timeout 1800 for run_modeling.py; the limit is per cell. |

Setup references: [Python virtual environments](https://docs.python.org/3.13/library/venv.html),
[Python on Windows](https://docs.python.org/3.13/using/windows.html),
and [VS Code Python settings](https://code.visualstudio.com/docs/python/settings-reference).
