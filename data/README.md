# Data provenance and input schema

The project uses synthetic profiles from [Kaggle Predicting Electric Vehicle Purchases, Playground Series Season 6 Episode 9](https://www.kaggle.com/competitions/playground-series-s6e9). Each row is a synthetic customer profile. The labeled outcome is `Will_Buy_EV`, mapped reversibly from No/Yes to 0/1. It does not supply a verified purchase transaction or prediction timeframe.

## Obtain the source data

Follow the unchanged Kaggle API instructions in the root README using your own authorized Kaggle account and `kaggle.json`. Join the competition or accept its terms if requested. Run the following from the project root with the documented environment active.

```bash
python ev_project/download_data.py
```

The script obtains `train.csv`, `test.csv` and `sample_submission.csv` at the project root. Full source CSVs are excluded from version control. Do not commit credentials. Synthetic data remain subject to their source access terms.

| File | Rows | Columns | Role |
|---|---:|---:|---|
| train.csv | 668,665 | 15 | Labeled source for development and evaluation |
| test.csv | 286,571 | 14 | Unlabeled source for scoring examples |
| sample_submission.csv | 286,571 | 2 | Competition output template |

The labeled file contains 116,779 Yes labels, or 17.4645%, and 551,886 No labels. The unlabeled test source cannot measure accuracy. File hashes and processed-data row counts are recorded in `docs/reports/data_manifest.json`.

## Included demonstration files

`data/demo/customer_profiles_demo.csv` contains the first 500 profiles from the unlabeled competition test source. It provides synthetic scoring inputs, not real customer records or outcome evidence. `customer_profiles_template.csv` contains the upload headers only. The saved model and included demo let a reviewer test scoring without a fresh competition download.

## Required input schema

Scoring requires a unique, nonblank `id` and the 13 predictors below. CSV columns are selected by name rather than position. Extra columns, including an outcome, are ignored with a warning and never enter prediction. CSV should be UTF-8. IDs are preserved as text, including leading zeros.

| Field | Input type or accepted values | Interpretation or limitation |
|---|---|---|
| id | Unique nonblank text | Traceability only; excluded from prediction |
| Age | Nonnegative whole number, at most 120 | Recorded age in years; observed source range 25–69 is not an eligibility rule |
| Annual_Income_USD | Nonnegative whole number | Annual USD amount; individual/household definition and country unspecified |
| Daily_Commute_km | Finite nonnegative number | Daily distance; one-way/round-trip basis unspecified |
| Number_of_Cars_Owned | Nonnegative whole number | Ownership count; person/household unit unspecified |
| Charging_Stations_Near_Home | Nonnegative whole number | Charging radius and station/connector unit unspecified |
| Charging_Stations_Near_Work | Nonnegative whole number | Same qualification as home charging count |
| Environmental_Concern_Level | 1, 2, 3, 4, 5 | Nominal code; survey anchors and spacing unverified |
| Gender | Female, Male, Other | Recorded category; business necessity and fairness require review |
| City_Type | Rural, Suburban, Urban | Definitions and country unspecified |
| Current_Car_Type | Hatchback, SUV, Sedan, Truck | Fuel type and vehicle age not provided |
| Home_Charging_Possible | No, Yes | Recorded possibility; installed equipment and cost not established |
| Subsidy_Available | No, Yes | Recorded flag; program eligibility and amount not established |
| Range_Anxiety_Level | High, Low, Medium | Recorded category; survey cutoffs unspecified |

The shared validator rejects missing inputs, duplicate IDs, unsupported categories, non-finite numbers, negative values and fractional counts. Whitespace normalization is visible. Plausible numeric values outside observed model-training ranges are preserved and warned about rather than rejected only for being unusual. Boolean values are not valid numbers. Exact application rules are defined in `ev_project/schema.py`.

## Preparation and evaluation provenance

Run EDA before modeling as documented in the root README. EDA produces `data/processed/train_clean.csv` and `test_clean.csv`, audits under `docs/reports/tables/`, and the first HTML report. No source rows or predictor values were removed or changed in the recorded analysis. Integer storage and the target mapping are lossless. IQR-flagged observations were retained.

Stratified development membership is 468,065 training, 100,300 validation and 100,300 holdout profiles. Membership is saved by ID at `data/splits/split_membership.csv` and excluded from version control with the generated data. Candidate preprocessing is fitted on training only. Model selection uses validation lift at 10% contact capacity. The selected pipeline is refitted on training plus validation, with holdout excluded. Descriptive full-source exploration preceded splitting, so reported performance is internal synthetic evaluation.

The monitoring reference is an aggregate of training-only inputs, linked to a specific saved model and source/membership hashes. It contains input distributions rather than raw profiles or verified outcome labels. Batch drift monitoring does not establish prediction quality or fairness.

The full dictionary and interpretation qualifications are in `docs/reports/tables/data_dictionary.csv`. Preparation provenance is in `docs/reports/data_manifest.json`; experiment rules and results are in `docs/reports/modeling/`; saved artifact information is in `models/model_metadata.json`.
