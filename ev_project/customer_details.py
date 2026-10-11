"""Attach supplied profile facts to scores and suggest consultation questions.

These questions are discussion aids. They do not explain why the model assigned
a score, infer customer preferences from demographics, or establish EV suitability.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .schema import FEATURE_COLUMNS, ID_COLUMN, ProfileValidationError, validate_profiles
from .scoring import rank_scores


PROFILE_DISPLAY_COLUMNS = [
    "Gender",
    "Age",
    "Annual_Income_USD",
    "City_Type",
    "Current_Car_Type",
    "Daily_Commute_km",
    "Home_Charging_Possible",
    "Range_Anxiety_Level",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Subsidy_Available",
    "Number_of_Cars_Owned",
    "Environmental_Concern_Level",
]

PROFILE_LABELS = {
    "Gender": "Gender",
    "Age": "Age (years)",
    "Annual_Income_USD": "Annual income (USD)",
    "City_Type": "Area type",
    "Current_Car_Type": "Current vehicle type",
    "Daily_Commute_km": "Daily commute (km)",
    "Home_Charging_Possible": "Home charging possible",
    "Range_Anxiety_Level": "Range anxiety (recorded level)",
    "Charging_Stations_Near_Home": "Charging stations near home",
    "Charging_Stations_Near_Work": "Charging stations near work",
    "Subsidy_Available": "Subsidy available (recorded)",
    "Number_of_Cars_Owned": "Cars owned",
    "Environmental_Concern_Level": "Environmental concern (raw code 1–5)",
}

RANKING_COLUMNS = ["rank", ID_COLUMN, "yes_probability"]
ENRICHED_RANKING_COLUMNS = RANKING_COLUMNS + FEATURE_COLUMNS


def enrich_ranking(ranked: pd.DataFrame, uploaded_profiles: pd.DataFrame) -> pd.DataFrame:
    """Return the complete ranking with each profile joined by normalized ID.

    The complete ranking must match the uploaded table one to one and follow the
    shared scorer's rank and tie rules. Targets and extra source/ranking fields are
    excluded. All original rank/ID/probability values and row order are retained.
    Choose top K after this function rather than passing a partial shortlist.
    """
    profiles, _ = validate_profiles(uploaded_profiles)
    if not isinstance(ranked, pd.DataFrame) or ranked.empty:
        raise ProfileValidationError("Provide a nonempty complete customer ranking.")
    if ranked.columns.duplicated().any():
        raise ProfileValidationError("Ranked column names must be unique.")
    missing = [column for column in RANKING_COLUMNS if column not in ranked.columns]
    if missing:
        raise ProfileValidationError("Missing ranked columns: " + ", ".join(missing))

    ranking = ranked.loc[:, RANKING_COLUMNS].copy().reset_index(drop=True)
    ids = ranking[ID_COLUMN]
    # Internal rankings use exact normalized strings. Refuse an ambiguous join
    # rather than silently replacing or converting their customer identifiers.
    if ids.isna().any() or not ids.map(lambda value: isinstance(value, str)).all():
        raise ProfileValidationError("Ranked customer IDs must be nonblank normalized strings.")
    normalized_ids = ids.astype("string").str.strip()
    if normalized_ids.eq("").any() or ids.tolist() != normalized_ids.tolist() or normalized_ids.duplicated().any():
        raise ProfileValidationError("Ranked customer IDs must be nonblank, normalized, and unique.")
    if len(ranking) != len(profiles) or set(normalized_ids) != set(profiles[ID_COLUMN]):
        raise ProfileValidationError("The complete ranking and uploaded profiles must contain exactly the same customer IDs.")

    ranks = pd.to_numeric(ranking["rank"], errors="coerce")
    if ranking["rank"].map(lambda value: isinstance(value, (bool, np.bool_))).any() or not np.array_equal(
        ranks.to_numpy(), np.arange(1, len(ranking) + 1)
    ):
        raise ProfileValidationError("Ranks must be consecutive whole numbers from 1 in displayed order.")
    if ranking["yes_probability"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ProfileValidationError("Ranked scores must be numeric probabilities, not True/False values.")
    try:
        expected = rank_scores(normalized_ids, ranking["yes_probability"])
    except (TypeError, ValueError) as error:
        raise ProfileValidationError("Ranked scores must be finite probabilities between 0 and 1.") from error
    if expected[ID_COLUMN].tolist() != normalized_ids.tolist():
        raise ProfileValidationError("The ranking must follow descending probabilities and the saved ID tie rule.")

    # Reindex by ID rather than row position: uploaded profiles may arrive in
    # any order while the returned priority order must remain untouched.
    attached = profiles.set_index(ID_COLUMN).loc[normalized_ids.tolist(), PROFILE_DISPLAY_COLUMNS]
    attached = attached.reset_index(drop=True)
    return pd.concat([ranking, attached], axis=1).loc[:, ENRICHED_RANKING_COLUMNS]


def conversation_prompts(profile: Mapping | pd.Series) -> list[str]:
    """Suggested questions using recorded transport context, not demographics.

    The supplied record is normalized under the same schema as prediction inputs.
    Income does not establish affordability. Concern codes have unknown survey
    anchors and are deliberately not interpreted in these suggested questions.
    """
    if isinstance(profile, pd.Series):
        profile = profile.to_dict()
    if not isinstance(profile, Mapping):
        raise ProfileValidationError("Provide a single customer profile as a record.")
    normalized, _ = validate_profiles(pd.DataFrame([dict(profile)]))
    record = normalized.iloc[0]

    prompts = [
        f"Your profile records a daily commute of {record['Daily_Commute_km']:g} km. "
        "Is this still accurate, and what other regular or longer journeys should we consider?",
    ]
    if record["Home_Charging_Possible"] == "Yes":
        prompts.append(
            "Your profile records home charging as possible. What parking access, permissions, "
            "and installation options should we confirm?"
        )
    else:
        prompts.append(
            "Your profile records home charging as unavailable. Which charging locations could "
            "fit your routine, and would home charging be worth exploring?"
        )
    prompts.extend([
        f"Your profile records {record['Charging_Stations_Near_Home']} charging stations near home "
        f"and {record['Charging_Stations_Near_Work']} near work. Which are actually convenient, "
        "accessible, and compatible with a vehicle you would consider?",
        f"Your recorded range anxiety level is {record['Range_Anxiety_Level']}. Which journeys or "
        "charging situations concern you, and what information would help you assess them?",
        f"Your current vehicle type is recorded as {record['Current_Car_Type']}. What do you want "
        "to keep or change about your current vehicle?",
        "What purchase budget and ongoing costs are comfortable for you, and what financing "
        "options would you like to discuss?",
        f"Your profile records subsidy availability as {record['Subsidy_Available']}. Would you "
        "like us to verify current programs, your eligibility, and the applicable conditions?",
    ])
    return prompts
