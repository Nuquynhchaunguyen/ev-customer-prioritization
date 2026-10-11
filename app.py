"""Streamlit interface for the EV customer prioritization class prototype."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from ev_project.schema import FEATURE_COLUMNS, ProfileValidationError, read_profile_csv
from ev_project.customer_details import PROFILE_LABELS, conversation_prompts, enrich_ranking
from ev_project.scoring import load_bundle, score_profiles


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "model_bundle.joblib"
DEMO_PATH = PROJECT_ROOT / "data" / "demo" / "customer_profiles_demo.csv"
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_PROFILE_ROWS = 300_000
RESULT_VIEW_VERSION = "customer-details-v1"


@st.cache_resource(show_spinner=False)
def cached_bundle(path: str, modified_at: int) -> dict:
    """Reload the local model when its file changes; never accept uploaded models."""
    return load_bundle(Path(path))


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Use a BOM so exported CSV files also open cleanly in desktop Excel."""
    return frame.to_csv(index=False).encode("utf-8-sig")


def model_details(metadata: dict) -> None:
    with st.expander("About this model and its evaluation"):
        st.write(
            "This classroom prototype was trained on synthetic Kaggle profiles. "
            "Its score estimates the supplied Will_Buy_EV = Yes label. "
            "It does not establish a real purchase, a purchase date, or the effect "
            "of contacting a customer. Real-world performance needs new labeled data."
        )
        trained_at = metadata.get("trained_at_utc")
        if trained_at:
            st.caption(f"Model trained: {trained_at}")
        metrics = metadata.get("holdout_metrics", {})
        names = {
            "roc_auc": "ROC AUC",
            "average_precision": "Average precision",
            "brier_score": "Brier score",
            "accuracy_at_05": "Accuracy at score 0.50",
            "precision_at_05": "Precision at score 0.50",
            "recall_at_05": "Recall at score 0.50",
            "precision_at_k": "Precision@K",
            "recall_at_k": "Recall@K",
            "lift_at_k": "Lift@K",
        }
        rows = []
        if isinstance(metrics, dict):
            n = metrics.get("n")
            k = metrics.get("primary_k")
            if isinstance(n, int) and isinstance(k, int) and n > 0:
                st.caption(
                    f"Evaluation batch: {n:,} labeled holdout profiles. "
                    f"Ranking measures below use K = {k:,} ({100 * k / n:.0f}% of that batch). "
                    "This evaluation capacity is separate from the capacity you choose for your upload."
                )
            for name, title in names.items():
                value = metrics.get(name)
                if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
                    if name == "lift_at_k":
                        formatted = f"{value:.2f}×"
                    elif name in {"roc_auc", "average_precision", "brier_score"}:
                        formatted = f"{value:.3f}"
                    else:
                        formatted = f"{100 * value:.2f}%"
                    rows.append({"Holdout measure": title, "Value": formatted})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption(
                "These results use labeled internal holdout data. Your uploaded "
                "profiles do not have verified outcomes, so their actual "
                "Precision@K and Lift@K cannot be calculated here."
            )
        st.write(
            "Higher scores receive earlier ranks. Equal scores use a repeatable "
            "tie break; different ranks among tied profiles do not imply "
            "different estimated Yes probabilities. Downloads retain probabilities "
            "between 0 and 1."
        )


def main() -> None:
    st.set_page_config(page_title="EV Customer Prioritization", page_icon="⚡", layout="wide")
    st.title("EV Customer Prioritization")
    st.write("Build a contact shortlist from customer profile information.")
    st.caption(
        "MSBA265 class prototype • Synthetic Kaggle competition data • "
        "Scores estimate Will_Buy_EV = Yes, not confirmed sales."
    )

    if not MODEL_PATH.is_file():
        st.info("The model is not ready yet. Train it once before ranking profiles.")
        with st.expander("Project setup"):
            st.write("From this project folder, using the project Python environment:")
            st.code("python scripts/train_models.py", language="bash")
            st.write("Then refresh this page. Setup instructions are also in README.md.")
        return

    model_version = MODEL_PATH.stat().st_mtime_ns
    try:
        bundle = cached_bundle(str(MODEL_PATH), model_version)
    except Exception:
        st.error("The saved model could not be opened. Rebuild it using the project training instructions, then refresh this page.")
        return
    metadata = bundle.get("metadata", {})
    selected_model = str(metadata.get("selected_model", "Trained classifier")).replace("_", " ")
    st.caption(f"Current model: {selected_model}")

    st.subheader("1. Choose customer profiles")
    source = st.radio("Profile source", ["Try the demo", "Upload a CSV"], horizontal=True, key="profile_source")
    content: bytes | None = None
    if source == "Try the demo":
        if DEMO_PATH.is_file():
            content = DEMO_PATH.read_bytes()
            st.write(
                "The demo uses unlabeled synthetic customer profiles from the "
                "competition test file. It illustrates the workflow."
            )
        else:
            st.info("The demo file is not available yet. You can upload a customer profile CSV instead.")
    else:
        upload = st.file_uploader(
            "Customer profile CSV",
            type="csv",
            max_upload_size=50,
            help="Use the same column names and categories as the project dataset. Maximum 50 MB and 300,000 profiles.",
            key="customer_csv",
        )
        st.caption("Uploaded profiles are kept in memory for this session and are not saved to project files.")
        if upload is not None:
            content = upload.getvalue()

    with st.expander("CSV format and sample file"):
        st.write(
            "Use one profile per row, with the 13 predictor columns from the "
            "competition dataset. Include a unique id for every profile. "
            "The app checks missing values, categories, and numeric "
            "values before producing scores."
        )
        st.caption(
            "The target Will_Buy_EV and unrelated extra columns are ignored "
            "with a warning. No target label is used as a prediction."
        )
        if DEMO_PATH.is_file():
            st.download_button("Download example profiles", DEMO_PATH.read_bytes(), "customer_profiles_demo.csv", "text/csv", key="download_demo", on_click="ignore")

    if content is None:
        model_details(metadata)
        return
    if len(content) > MAX_FILE_BYTES:
        st.error("This file exceeds 50 MB. Upload a smaller CSV file.")
        return

    input_digest = hashlib.sha256(content).hexdigest()
    cache_key = (input_digest, model_version, RESULT_VIEW_VERSION)
    cached_result = st.session_state.get("ranked_profiles")
    if cached_result and cached_result["key"] != cache_key:
        st.session_state.pop("ranked_profiles", None)
        cached_result = None

    profiles = None
    try:
        # Reuse verified ranked profiles for detail/column/K changes within this session.
        if cached_result:
            profile_count = len(cached_result["full_ranked"])
        else:
            profiles = read_profile_csv(content)
            profile_count = len(profiles)
        if profile_count > MAX_PROFILE_ROWS:
            st.error("This file contains more than 300,000 profiles. Split it into smaller CSV files; no profiles have been scored.")
            return
        if profile_count == 0:
            st.error("This CSV has no customer rows. Add at least one profile and upload it again.")
            return
    except (ProfileValidationError, ValueError, UnicodeError) as exc:
        st.error(f"Please check the CSV: {exc}")
        return

    st.caption(f"{profile_count:,} customer profiles loaded. Limits: 50 MB and 300,000 profiles.")
    st.subheader("2. Set your contact capacity")
    capacity_key = f"contact_capacity_{input_digest[:16]}"
    default_capacity = min(50, profile_count)
    with st.form("rank_profiles_form"):
        capacity = st.number_input(
            "How many customers can you contact?",
            min_value=1,
            max_value=profile_count,
            value=default_capacity,
            step=1,
            key=capacity_key,
            help="The shortlist contains exactly this many profiles, starting with the highest scores.",
        )
        submitted = st.form_submit_button("Create shortlist", type="primary")

    if submitted:
        try:
            with st.spinner("Checking profiles and estimating Yes probabilities…"):
                if not cached_result:
                    full_ranked, _, warnings = score_profiles(profiles, bundle, int(capacity))
                    full_ranked = enrich_ranking(full_ranked, profiles)
                    cached_result = {"key": cache_key, "full_ranked": full_ranked, "warnings": warnings}
                cached_result["capacity"] = int(capacity)
                st.session_state["ranked_profiles"] = cached_result
        except (ProfileValidationError, ValueError) as exc:
            st.session_state.pop("ranked_profiles", None)
            st.error(f"Profiles could not be scored: {exc}")
            return
        except Exception:
            st.session_state.pop("ranked_profiles", None)
            st.error("The model could not score these profiles. Check the CSV format and the saved model, then try again.")
            return

    if cached_result:
        full_ranked = cached_result["full_ranked"]
        chosen_capacity = cached_result["capacity"]
        shortlist = full_ranked.head(chosen_capacity).copy()
        for message in cached_result["warnings"]:
            st.warning(message)
        st.subheader("3. Review your contact shortlist")
        st.write(
            f"The top **{chosen_capacity:,} of {len(full_ranked):,} profiles** "
            "are ranked below. Review the list before taking action."
        )
        st.caption(
            "The score estimates Will_Buy_EV = Yes. A score is not "
            "a confirmed purchase or a forecast of campaign results."
        )
        selected_columns = st.multiselect(
            "Customer information to show",
            options=FEATURE_COLUMNS,
            default=["Gender", "Age", "Annual_Income_USD"],
            format_func=lambda column: PROFILE_LABELS[column],
            key=f"visible_profile_columns_{input_digest[:16]}",
            help="Choose the information shown in this table. Both CSV downloads always include all 13 profile fields.",
        )
        table_columns = ["rank", "id", "yes_probability"] + selected_columns
        st.dataframe(
            shortlist.head(100).loc[:, table_columns],
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                **{column: st.column_config.Column(label) for column, label in PROFILE_LABELS.items()},
                "rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                "id": st.column_config.TextColumn("Customer ID", width="small"),
                "yes_probability": st.column_config.NumberColumn("Estimated Yes probability", format="percent", width="medium"),
                "Gender": st.column_config.TextColumn("Gender", width="small"),
                "Age": st.column_config.NumberColumn("Age (years)", format="%d", width="small"),
                "Annual_Income_USD": st.column_config.NumberColumn("Annual income (USD)", format="$%,d", width="medium"),
                "Daily_Commute_km": st.column_config.NumberColumn("Daily commute (km)", format="%.1f"),
            },
            key="shortlist_preview",
        )
        if len(shortlist) > 100:
            st.caption("Preview shows the first 100 profiles. The download contains the complete shortlist.")
        left, right = st.columns(2)
        with left:
            st.download_button(
                "Download contact shortlist",
                csv_bytes(shortlist),
                f"ev_contact_shortlist_top_{chosen_capacity}.csv",
                "text/csv",
                key="download_shortlist",
                type="primary",
                on_click="ignore",
            )
        with right:
            st.download_button(
                "Download all ranked profiles",
                csv_bytes(full_ranked),
                "ev_all_ranked_profiles.csv",
                "text/csv",
                key="download_full_ranking",
                on_click="ignore",
            )
        st.caption("Both downloads include rank, customer ID, Yes probability (0 to 1), and all 13 recorded profile fields.")

        st.subheader("4. Prepare a customer conversation")
        detail_key = f"selected_profile_rank_{input_digest[:16]}"
        if st.session_state.get(detail_key, 1) > chosen_capacity:
            st.session_state.pop(detail_key, None)
        chosen_rank = st.number_input(
            "Priority rank to view",
            min_value=1,
            max_value=chosen_capacity,
            value=1,
            step=1,
            key=detail_key,
            help="Enter a rank from the current shortlist to see that customer's recorded profile.",
        )
        customer = shortlist.iloc[int(chosen_rank) - 1]
        st.write(
            f"**Customer ID {customer['id']}** · Rank {int(customer['rank']):,} "
            f"· Estimated Yes probability {customer['yes_probability']:.2%}"
        )
        demographic_columns = st.columns(3)
        demographic_columns[0].metric("Gender", str(customer["Gender"]))
        demographic_columns[1].metric("Age (years)", str(int(customer["Age"])))
        demographic_columns[2].metric("Annual income (USD)", f"${int(customer['Annual_Income_USD']):,}")
        context_column, questions_column = st.columns(2)
        with context_column:
            st.markdown("**Driving and charging context**")
            context_fields = [
                "Current_Car_Type", "Daily_Commute_km", "Home_Charging_Possible",
                "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
                "Range_Anxiety_Level", "Subsidy_Available", "City_Type",
            ]
            st.dataframe(
                pd.DataFrame({"Profile field": [PROFILE_LABELS[c] for c in context_fields],
                              "Recorded value": [str(customer[c]) for c in context_fields]}),
                hide_index=True, width="stretch",
            )
        with questions_column:
            st.markdown("**Suggested questions for the conversation**")
            for question in conversation_prompts(customer):
                st.markdown(f"- {question}")
            st.caption("These questions use recorded profile information; they are not explanations of the model score.")
        with st.expander("All recorded profile fields"):
            st.dataframe(
                pd.DataFrame({"Profile field": [PROFILE_LABELS[c] for c in FEATURE_COLUMNS],
                              "Recorded value": [str(customer[c]) for c in FEATURE_COLUMNS]}),
                hide_index=True, width="stretch",
            )
            st.caption("Environmental concern is an undocumented code (1–5); the app does not infer what each code means.")
        st.caption("This dataset contains synthetic profile IDs, without customer names or phone numbers.")

    model_details(metadata)


if __name__ == "__main__":
    main()
