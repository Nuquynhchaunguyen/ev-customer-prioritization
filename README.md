# EV Customer Prioritization

**Nu Quynh Chau Nguyen | MSBA265 | Fall 2026**

**Which customers should an EV dealership prioritize for contact when its outreach capacity is limited?**

I use the professor-approved [Kaggle competition dataset](https://www.kaggle.com/competitions/playground-series-s6e9)
to study this question with supervised binary classification.

## P1 submission

**Start with the [combined P1 PDF](docs/P1/EV_Customer_Prioritization_P1.pdf).**
It is the single P1 submission document and contains:

- The dealership problem, intended user, project question, proposed method, and success measure.
- Six topic examples and a cited one-page infographic.
- The dataset profile, initial quality checks, an interpreted chart, a simple baseline, and next steps.

P1 is the upcoming submission. P2 is the later review, two weeks after P1.
The Assignment Validation form is completed with an independent peer and submitted
separately to the professor. It stays local and is excluded from GitHub.
Peer verification and signatures are still pending.

## Submission stages

| Stage | Main materials | Location and status |
|---|---|---|
| P1 | One combined proposal PDF with topic background and initial data exploration | `docs/P1/` — current submission |
| P2 | Draft technical paper, stakeholder slides, runnable analysis, results, and a saved model | `docs/P2/` and the shared code below — work in progress; update after P1 feedback |
| Final | Polished P2 work after instructor feedback, including the final report, analysis output, presentation, and reproducible handoff | Later; no duplicate final package has been created |

## Project map

```text
ev-customer-prioritization/
├── README.md           Project overview and where to start
├── docs/
│   ├── P1/             One combined P1 submission PDF
│   ├── P2/             Draft paper, slides, run guide, and extension plan
│   └── reports/        Supporting notebook outputs, charts, and tables
├── notebooks/          Data understanding and modeling analysis
├── ev_project/         Dataset download, preparation, training, and scoring
├── data/               Included demo; full and generated data stay local
├── models/             Saved preprocessing and classification model
├── app.py              Streamlit customer ranking and conversation guides
├── requirements.txt    Python packages needed to run the project
├── tests/              Checks for data, model, ranking, and app behavior
└── .gitignore          Excludes credentials, full data, and local environments
```

The source code and saved model are shared working materials for P2. They are
kept in one place so later submissions can reuse them. The combined P1 PDF is
self-contained; its review does not require a Streamlit demonstration.

## Proposed classification method

The main method is **Logistic Regression**, as taught in the professor's
[classification materials](https://github.com/shylasolis/machine-learning-pathway/blob/main/Project_02_Classification/assignment_instructions.md).
It calculates a weighted score and then uses the sigmoid function:

```text
z = beta_0 + beta_1*x_1 + ... + beta_p*x_p
P(Will_Buy_EV = Yes) = 1 / (1 + exp(-z))
```

Here the inputs are the processed numeric and encoded categorical features.
Rank the resulting probabilities to choose the top K profiles. A threshold for
assigning Yes/No is a separate decision from choosing exactly K people.
The baseline is a Dummy classifier with no feature-based ranking. P1 explains
the planned use of Precision@K and Lift@K to compare ranking with that baseline.

The existing prototype already uses this weighted-score-and-sigmoid model.
Its coefficients are fitted with scikit-learn's L-BFGS optimizer and L2
regularization; it does not use the separate class exercise's NumPy training loop.

## P2 working materials

- [Draft technical paper](docs/P2/EV_Customer_Prioritization_Technical_Report.pdf).
- [Draft stakeholder PowerPoint](docs/P2/EV_Customer_Prioritization_Stakeholder_Presentation.pptx).
- [Mac and Windows run guide](docs/P2/RUN_GUIDE.md), including Git clone or ZIP,
  environment setup, Kaggle API download, notebook execution, app use, and tests.

The notebooks are the primary analysis source. The `ev_project` modules support
their execution and the app. Saved outputs are in `docs/reports/`; the fitted
pipeline and its metadata are in `models/`. The small demo is included; full
competition files are downloaded with the reviewer's own Kaggle API credentials.

The P2 drafts need the professor's individual P1 feedback when it becomes
available, a documented response to that feedback, and independent peer
reproduction before submission. The [optional extension plan](docs/P2/extension_plan.json)
covers FastAPI, Docker, and MLOps; those extensions are planned, not completed.

## Data interpretation

The data contain **668,665 synthetic labeled profiles**, with **17.46% Yes**
labels. Each row is a profile, with 13 predictors and a supplied EV purchase label.
The dataset does not supply customer names, phone numbers, purchase dates, or
verified responses to outreach. Scores estimate that dataset label; they do not
establish real sales, campaign impact, or revenue.

For field definitions and exploration, read the
[data understanding notebook](notebooks/01_data_understanding_and_cleaning.ipynb)
and [data dictionary](docs/reports/tables/data_dictionary.csv).
