"""Draft staff-reviewed EV conversations from supplied transport facts.

Templates do not change scores, explain the model, or establish sales impact.
They run locally without an external AI service.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .customer_details import ENRICHED_RANKING_COLUMNS, conversation_prompts
from .schema import ID_COLUMN, ProfileValidationError, validate_profiles
from .scoring import rank_scores

GUIDE_VERSION = "conversation-guide-v1"
MAX_GUIDE_EXPORT = 1_000
GUIDE_SECTIONS = (
    ("opening", "1. Opening (say)"),
    ("needs_questions", "2. Confirm needs (ask)"),
    ("discussion_points", "3. Topics to discuss (staff preparation)"),
    ("concerns_to_explore", "4. Explore concerns (ask, then respond)"),
    ("next_step", "5. Agree on a next step (say)"),
)
GUIDE_EXPORT_COLUMNS = ENRICHED_RANKING_COLUMNS + ["conversation_guide", "guide_status"]


def build_conversation_guide(profile: Mapping | pd.Series) -> dict[str, str]:
    """Normalize one profile and build five sections from recorded transport facts.

    Facts are prompts to confirm. Demographics and undocumented concern codes
    do not select wording, and recorded income does not establish a budget.
    """
    if isinstance(profile, pd.Series):
        profile = profile.to_dict()
    if not isinstance(profile, Mapping):
        raise ProfileValidationError("Provide a single customer profile as a record.")
    normalized, _ = validate_profiles(pd.DataFrame([dict(profile)]))
    record = normalized.iloc[0]
    questions = conversation_prompts(record)
    needs = "\n".join("- " + questions[index] for index in (0, 1, 4, 5))

    if record["Home_Charging_Possible"] == "Yes":
        charging_topic = (
            "If home charging is confirmed, discuss parking permission, installation "
            "requirements, and a suitable routine. Verify details before suggesting equipment."
        )
    else:
        charging_topic = (
            "If home charging remains unavailable, explore convenient public or workplace "
            "charging and confirm a workable routine before suggesting a vehicle."
        )
    station_topic = (
        f"The profile lists {record['Charging_Stations_Near_Home']} stations near home "
        f"and {record['Charging_Stations_Near_Work']} near work. Confirm which are usable; "
        "counts do not establish access, compatibility, availability, or charging speed."
    )
    if record["Subsidy_Available"] == "Yes":
        subsidy_topic = (
            "A subsidy is recorded as available. Ask whether the customer wants current "
            "programs and eligibility checked; do not promise a discount."
        )
    else:
        subsidy_topic = (
            "No subsidy is recorded. Ask whether the customer wants current programs "
            "checked; this record does not establish present eligibility."
        )
    topics = "\n".join("- " + text for text in (charging_topic, station_topic, subsidy_topic))

    level = record["Range_Anxiety_Level"]
    if level == "High":
        concern = (
            'Ask: "Your profile records high concern about driving range. Is that still '
            'true, and which trips or charging situations worry you most?"\n'
            "Staff response: If this concern is confirmed, discuss those routes and "
            "charging options using verified vehicle information. Do not promise a range."
        )
    elif level == "Medium":
        concern = (
            'Ask: "Your profile records some concern about driving range. Which journeys '
            'would you want to check before considering an EV?"\n'
            "Staff response: First confirm the concern, then review relevant routes and "
            "verified vehicle information together. Do not assume the concern is resolved."
        )
    else:
        concern = (
            'Ask: "Your profile records low concern about driving range. Is that still '
            'accurate, and are there any longer journeys we should check?"\n'
            "Staff response: Let the customer name any concerns. Discuss verified "
            "information relevant to those concerns rather than assuming there are none."
        )
    return {
        "opening": (
            "Hello, I'm calling from the dealership. Is now a good time for a short "
            "conversation about whether an EV could fit your driving needs?"
        ),
        "needs_questions": needs,
        "discussion_points": topics,
        "concerns_to_explore": concern,
        "next_step": (
            "If this sounds useful, would you prefer a further consultation or a test "
            "drive? What time would suit you? If you are not interested, we can end here."
        ),
    }


def format_conversation_guide(guide: Mapping[str, str]) -> str:
    """Render a sectioned draft as plain text for editing or export."""
    if not isinstance(guide, Mapping) or any(
        key not in guide or not isinstance(guide[key], str) for key, _ in GUIDE_SECTIONS
    ):
        raise ProfileValidationError("A conversation guide must contain all five text sections.")
    return "\n\n".join(f"{label}\n{guide[key]}" for key, label in GUIDE_SECTIONS)


def _validated_shortlist(shortlist: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(shortlist, pd.DataFrame) or shortlist.empty:
        raise ProfileValidationError("Provide a nonempty customer shortlist.")
    if len(shortlist) > MAX_GUIDE_EXPORT:
        raise ProfileValidationError(
            f"Conversation exports support at most {MAX_GUIDE_EXPORT:,} profiles per batch."
        )
    if shortlist.columns.duplicated().any():
        raise ProfileValidationError("Shortlist column names must be unique.")
    missing = [column for column in ENRICHED_RANKING_COLUMNS if column not in shortlist]
    if missing:
        raise ProfileValidationError("Missing shortlist columns: " + ", ".join(missing))
    frame = shortlist.loc[:, ENRICHED_RANKING_COLUMNS].copy().reset_index(drop=True)
    ids = frame[ID_COLUMN]
    if ids.isna().any() or not ids.map(lambda value: isinstance(value, str)).all():
        raise ProfileValidationError("Shortlist IDs must be normalized text.")
    if ids.tolist() != ids.astype("string").str.strip().tolist():
        raise ProfileValidationError("Shortlist IDs must be normalized text.")
    normalized, _ = validate_profiles(frame)
    ranks = pd.to_numeric(frame["rank"], errors="coerce")
    if frame["rank"].map(lambda value: isinstance(value, (bool, np.bool_))).any() or not np.array_equal(
        ranks.to_numpy(), np.arange(1, len(frame) + 1)
    ):
        raise ProfileValidationError("Shortlist ranks must run from 1 to K in displayed order.")
    if frame["yes_probability"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ProfileValidationError("Shortlist scores must be numeric probabilities.")
    try:
        expected = rank_scores(ids, frame["yes_probability"])
    except (TypeError, ValueError) as error:
        raise ProfileValidationError("Shortlist scores must be finite probabilities from 0 to 1.") from error
    if expected[ID_COLUMN].tolist() != ids.tolist():
        raise ProfileValidationError("Shortlist rows must follow the saved score and tie ordering.")
    # Each ID keeps its own normalized facts. No separate positional join is used.
    normalized.insert(0, "yes_probability", frame["yes_probability"].to_numpy())
    normalized.insert(0, "rank", frame["rank"].to_numpy())
    return normalized.loc[:, ENRICHED_RANKING_COLUMNS]


def export_conversation_guides(
    shortlist: pd.DataFrame, edits: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """Return exactly K guides and source facts, matching saved edits by exact ID.

    Edits for IDs outside the current shortlist are allowed so reducing K does
    not discard session drafts. Empty edited text is preserved deliberately.
    """
    frame = _validated_shortlist(shortlist)
    if edits is None:
        edits = {}
    if not isinstance(edits, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in edits.items()
    ):
        raise ProfileValidationError("Saved conversation edits must map text IDs to text scripts.")
    scripts, statuses = [], []
    for _, row in frame.iterrows():
        customer_id = row[ID_COLUMN]
        if customer_id in edits:
            scripts.append(edits[customer_id])
            statuses.append("Edited by staff")
        else:
            scripts.append(format_conversation_guide(build_conversation_guide(row)))
            statuses.append("Template draft")
    frame["conversation_guide"] = scripts
    frame["guide_status"] = statuses
    return frame.loc[:, GUIDE_EXPORT_COLUMNS]


def conversation_guides_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize with a BOM and neutralize potentially executable spreadsheet text.

    Prefix risky text cells with an apostrophe in the serialized copy only.
    The source DataFrame and its customer IDs remain unchanged.
    """
    if not isinstance(frame, pd.DataFrame) or frame.columns.tolist() != GUIDE_EXPORT_COLUMNS:
        raise ProfileValidationError("Provide the complete conversation-guide export table.")
    serialized = frame.copy(deep=True)

    def safe_cell(value):
        if isinstance(value, str) and (
            value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r", "\n"))
        ):
            return "'" + value
        return value

    for column in serialized.columns:
        serialized[column] = serialized[column].map(safe_cell)
    return serialized.to_csv(index=False).encode("utf-8-sig")
