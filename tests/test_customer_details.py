"""Tests that ranked customer facts are attached to the correct ID."""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ev_project.customer_details import (
    ENRICHED_RANKING_COLUMNS,
    PROFILE_DISPLAY_COLUMNS,
    PROFILE_LABELS,
    conversation_prompts,
    enrich_ranking,
)
from ev_project.schema import FEATURE_COLUMNS, ProfileValidationError, validate_profiles
from ev_project.scoring import rank_scores


def profiles():
    return pd.DataFrame({
        "id": ["0001", "0020", "customer-3"],
        "Age": [35, 44, 60],
        "Annual_Income_USD": [50000, 80000, 110000],
        "Daily_Commute_km": [12.5, 40, 55.2],
        "Number_of_Cars_Owned": [1, 2, 3],
        "Charging_Stations_Near_Home": [0, 3, 5],
        "Charging_Stations_Near_Work": [1, 4, 8],
        "Environmental_Concern_Level": [1, 3.0, "5"],
        "Gender": ["Female", "Male", "Other"],
        "City_Type": ["Urban", "Suburban", "Rural"],
        "Current_Car_Type": ["Sedan", "SUV", "Truck"],
        "Home_Charging_Possible": ["No", "Yes", "Yes"],
        "Subsidy_Available": ["No", "Yes", "Yes"],
        "Range_Anxiety_Level": ["Medium", "Low", "High"],
    })


class CustomerDetailsTests(unittest.TestCase):
    def test_reordered_leading_zero_ids_attach_correct_facts_and_preserve_scores(self):
        source = profiles().iloc[[2, 0, 1]].reset_index(drop=True)
        ranked = rank_scores(["0001", "0020", "customer-3"], [0.95, 0.1, 0.75])
        before = ranked.copy(deep=True)
        enriched = enrich_ranking(ranked, source)
        assert_frame_equal(enriched.loc[:, before.columns], before)
        assert_frame_equal(ranked, before)
        self.assertEqual(enriched["id"].tolist(), ["0001", "customer-3", "0020"])
        self.assertEqual(enriched["Age"].tolist(), [35, 60, 44])
        self.assertEqual(enriched["Annual_Income_USD"].tolist(), [50000, 110000, 80000])
        self.assertEqual(enriched["Gender"].tolist(), ["Female", "Other", "Male"])
        self.assertEqual(enriched.columns.tolist(), ENRICHED_RANKING_COLUMNS)
        self.assertEqual(set(PROFILE_DISPLAY_COLUMNS), set(FEATURE_COLUMNS))
        self.assertEqual(set(PROFILE_LABELS), set(FEATURE_COLUMNS))

    def test_source_is_normalized_and_all_target_extra_columns_are_excluded(self):
        source = profiles().astype(object)
        source.columns = [f" {column} " for column in source.columns]
        source.loc[0, " id "] = " 0001 "
        source.loc[0, " Age "] = "35"
        source.loc[0, " Gender "] = " Female "
        source["Will_Buy_EV"] = "Yes"
        source["email"] = "private@example.com"
        ranked = rank_scores(["0001", "0020", "customer-3"], [0.95, 0.1, 0.75])
        ranked["Will_Buy_EV"] = "No"
        ranked["extra"] = "ignore"
        enriched = enrich_ranking(ranked, source)
        expected, _ = validate_profiles(source)
        for _, row in enriched.iterrows():
            expected_row = expected.set_index("id").loc[row["id"]]
            for field in FEATURE_COLUMNS:
                self.assertEqual(row[field], expected_row[field])
        self.assertEqual(enriched.loc[0, "Gender"], "Female")
        self.assertEqual(enriched.loc[0, "Age"], 35)
        self.assertNotIn("Will_Buy_EV", enriched)
        self.assertNotIn("email", enriched)
        self.assertNotIn("extra", enriched)

    def test_missing_mismatched_duplicate_or_unnormalized_ids_are_rejected(self):
        ranked = rank_scores(profiles()["id"], [0.3, 0.7, 0.5])
        variants = [ranked.iloc[:2].copy(), ranked.drop(columns="id")]
        for replacement in ["unknown", "0001", " 0020 ", None, 20]:
            variant = ranked.copy().astype({"id": object})
            variant.loc[0, "id"] = replacement
            variants.append(variant)
        for variant in variants:
            with self.subTest(ids=variant.get("id")), self.assertRaises(ProfileValidationError):
                enrich_ranking(variant, profiles())
        with self.assertRaises(ProfileValidationError):
            enrich_ranking(ranked, profiles().drop(columns="Gender"))
        duplicate_source = profiles()
        duplicate_source.loc[1, "id"] = "0001"
        with self.assertRaises(ProfileValidationError):
            enrich_ranking(ranked, duplicate_source)

    def test_corrupt_rank_order_scores_and_tie_order_are_rejected(self):
        ranked = rank_scores(profiles()["id"], [0.3, 0.7, 0.5])
        variants = []
        for invalid in [[1, 2, 4], [0, 1, 2], [1, 1, 3], [True, 2, 3]]:
            variant = ranked.copy()
            variant["rank"] = invalid
            variants.append(variant)
        reversed_rows = ranked.iloc[::-1].copy().reset_index(drop=True)
        reversed_rows["rank"] = [1, 2, 3]
        variants.append(reversed_rows)
        for score in [np.nan, np.inf, -0.1, 1.1, True, "broken"]:
            variant = ranked.astype({"yes_probability": object})
            variant.loc[0, "yes_probability"] = score
            variants.append(variant)
        ties = rank_scores(profiles()["id"], [0.5, 0.5, 0.5]).iloc[::-1].reset_index(drop=True)
        ties["rank"] = [1, 2, 3]
        variants.append(ties)
        for variant in variants:
            with self.subTest(ranking=variant.to_dict("list")), self.assertRaises(ProfileValidationError):
                enrich_ranking(variant, profiles())
        enriched_ties = enrich_ranking(rank_scores(profiles()["id"], [0.5, 0.5, 0.5]), profiles())
        self.assertEqual(len(enriched_ties), 3)

    def test_prompts_use_recorded_transport_context_but_not_demographics(self):
        profile = profiles().iloc[0].copy()
        prompts = conversation_prompts(profile)
        self.assertEqual(len(prompts), 7)
        self.assertTrue(all(prompt.endswith("?") for prompt in prompts))
        self.assertIn("12.5 km", prompts[0])
        self.assertIn("home charging as unavailable", prompts[1])
        self.assertIn("0 charging stations near home and 1 near work", prompts[2])
        self.assertIn("Medium", prompts[3])
        self.assertIn("Sedan", prompts[4])
        self.assertIn("budget", prompts[5])
        self.assertIn("verify current programs", prompts[6])
        changed = profile.copy()
        changed["Gender"] = "Other"
        changed["Age"] = 75
        changed["Annual_Income_USD"] = 0
        changed["Environmental_Concern_Level"] = "5"
        self.assertEqual(conversation_prompts(changed), prompts)
        changed["Home_Charging_Possible"] = "Yes"
        changed["Subsidy_Available"] = "Yes"
        yes_prompts = conversation_prompts(changed.to_dict())
        self.assertIn("home charging as possible", yes_prompts[1])
        self.assertIn("availability as Yes", yes_prompts[6])
        with self.assertRaises(ProfileValidationError):
            conversation_prompts({"id": "incomplete"})


if __name__ == "__main__":
    unittest.main()
