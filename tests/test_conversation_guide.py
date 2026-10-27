"""Check profile-grounded wording, ID-safe edits, and bounded guide exports."""

import io
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from ev_project.conversation_guide import (
    GUIDE_EXPORT_COLUMNS,
    GUIDE_SECTIONS,
    MAX_GUIDE_EXPORT,
    build_conversation_guide,
    conversation_guides_csv_bytes,
    export_conversation_guides,
    format_conversation_guide,
)
from ev_project.customer_details import enrich_ranking
from ev_project.schema import ProfileValidationError
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
        "Environmental_Concern_Level": [1, 3, 5],
        "Gender": ["Female", "Male", "Other"],
        "City_Type": ["Urban", "Suburban", "Rural"],
        "Current_Car_Type": ["Sedan", "SUV", "Truck"],
        "Home_Charging_Possible": ["No", "Yes", "Yes"],
        "Subsidy_Available": ["No", "Yes", "Yes"],
        "Range_Anxiety_Level": ["Medium", "Low", "High"],
    })


def shortlist(k=3):
    source = profiles().iloc[[2, 0, 1]].reset_index(drop=True)
    return enrich_ranking(rank_scores(["0001", "0020", "customer-3"], [0.95, 0.1, 0.75]), source).head(k)


class ConversationGuideTests(unittest.TestCase):
    def test_five_sections_use_correct_recorded_facts_and_conditional_next_step(self):
        guide = build_conversation_guide(profiles().iloc[0])
        self.assertEqual(list(guide), [key for key, _ in GUIDE_SECTIONS])
        text = format_conversation_guide(guide)
        for _, label in GUIDE_SECTIONS:
            self.assertIn(label, text)
        self.assertIn("12.5 km", guide["needs_questions"])
        self.assertIn("Sedan", guide["needs_questions"])
        self.assertIn("budget", guide["needs_questions"])
        self.assertIn("home charging as unavailable", guide["needs_questions"])
        self.assertIn("0 stations near home and 1 near work", guide["discussion_points"])
        self.assertIn("No subsidy is recorded", guide["discussion_points"])
        self.assertIn("some concern", guide["concerns_to_explore"])
        self.assertIn("Is now a good time", guide["opening"])
        self.assertIn("If this sounds useful", guide["next_step"])
        self.assertIn("not interested", guide["next_step"])
        self.assertTrue(1500 <= len(text) <= 2500, len(text))

    def test_charging_subsidy_and_range_conditions_change_the_relevant_sections(self):
        low = build_conversation_guide(profiles().iloc[1].to_dict())
        high = build_conversation_guide(profiles().iloc[2])
        self.assertIn("home charging as possible", low["needs_questions"])
        self.assertIn("If home charging is confirmed", low["discussion_points"])
        self.assertIn("A subsidy is recorded as available", low["discussion_points"])
        self.assertIn("do not promise a discount", low["discussion_points"])
        self.assertIn("low concern", low["concerns_to_explore"])
        self.assertIn("high concern", high["concerns_to_explore"])
        self.assertIn("Do not promise a range", high["concerns_to_explore"])

    def test_demographics_unknown_concern_code_and_score_do_not_choose_script(self):
        source = profiles().iloc[0].to_dict()
        before = build_conversation_guide(source)
        source.update(Gender="Other", Age=75, Annual_Income_USD=0,
                      Environmental_Concern_Level=5, yes_probability=0.01,
                      Will_Buy_EV="No", preferred_brand="invented")
        self.assertEqual(build_conversation_guide(source), before)
        source["id"] = "different-id"
        self.assertEqual(build_conversation_guide(source), before)
        text = format_conversation_guide(before)
        self.assertNotIn("50000", text)
        self.assertNotIn("Female", text)
        self.assertNotIn("probability", text)
        self.assertNotIn("guaranteed", text)

    def test_invalid_single_profiles_and_incomplete_section_maps_fail(self):
        for invalid in [None, [], {"id": "missing"}, {**profiles().iloc[0].to_dict(), "Range_Anxiety_Level": "unknown"}]:
            with self.subTest(profile=invalid), self.assertRaises(ProfileValidationError):
                build_conversation_guide(invalid)
        guide = build_conversation_guide(profiles().iloc[0])
        for invalid in [None, {}, {**guide, "opening": None}]:
            with self.subTest(guide=invalid), self.assertRaises(ProfileValidationError):
                format_conversation_guide(invalid)

    def test_exact_k_edits_follow_ids_and_exports_do_not_mutate_source(self):
        selected = shortlist(2)
        before = selected.copy(deep=True)
        custom = 'Xin chào, mình xác nhận lịch nhé.\n"Quoted", comma, and café.'
        exported = export_conversation_guides(selected, {"customer-3": custom, "0020": "outside current K"})
        assert_frame_equal(selected, before)
        assert_frame_equal(exported.loc[:, selected.columns], selected)
        self.assertEqual(exported.columns.tolist(), GUIDE_EXPORT_COLUMNS)
        self.assertEqual(exported["id"].tolist(), ["0001", "customer-3"])
        self.assertEqual(exported["guide_status"].tolist(), ["Template draft", "Edited by staff"])
        self.assertEqual(exported.loc[1, "conversation_guide"], custom)
        self.assertIn("12.5 km", exported.loc[0, "conversation_guide"])
        self.assertNotIn("55.2 km", exported.loc[0, "conversation_guide"])
        blank = export_conversation_guides(selected, {"0001": ""})
        self.assertEqual(blank.loc[0, "conversation_guide"], "")
        self.assertEqual(blank.loc[0, "guide_status"], "Edited by staff")
        self.assertNotIn("Will_Buy_EV", exported)

    def test_invalid_shortlists_ranks_scores_and_edits_are_rejected(self):
        selected = shortlist()
        variants = [pd.DataFrame(), selected.drop(columns="Age"), selected.drop(columns="rank")]
        for field, invalid in [("id", "0001"), ("id", " padded "), ("id", 20),
                               ("rank", 9), ("rank", True), ("yes_probability", np.nan),
                               ("yes_probability", np.inf), ("yes_probability", -0.1),
                               ("yes_probability", True), ("Gender", "unknown")]:
            changed = selected.astype({field: object}).copy()
            changed.loc[1, field] = invalid
            variants.append(changed)
        changed = selected.iloc[::-1].reset_index(drop=True)
        changed["rank"] = [1, 2, 3]
        variants.append(changed)
        variants.append(pd.concat([selected, selected[["id"]]], axis=1))
        for index, invalid in enumerate(variants):
            with self.subTest(variant=index), self.assertRaises(ProfileValidationError):
                export_conversation_guides(invalid)
        for invalid in [[], {1: "text"}, {"0001": None}, {"0001": 3}]:
            with self.subTest(edits=invalid), self.assertRaises(ProfileValidationError):
                export_conversation_guides(selected, invalid)

    def test_invalid_tie_order_is_rejected(self):
        source = profiles()
        tied = enrich_ranking(rank_scores(source["id"], [0.5, 0.5, 0.5]), source)
        self.assertEqual(len(export_conversation_guides(tied)), 3)
        reversed_tie = tied.iloc[::-1].reset_index(drop=True)
        reversed_tie["rank"] = [1, 2, 3]
        with self.assertRaises(ProfileValidationError):
            export_conversation_guides(reversed_tie)

    def test_oversized_export_fails_before_any_guide_generation(self):
        oversized = pd.concat([shortlist(1)] * (MAX_GUIDE_EXPORT + 1), ignore_index=True)
        with self.assertRaisesRegex(ProfileValidationError, "at most 1,000"):
            export_conversation_guides(oversized)

    def test_csv_preserves_multiline_edits_and_neutralizes_formula_cells_without_mutation(self):
        text = 'Xin chào.\nAre you free on Friday? "Yes", café.'
        frame = export_conversation_guides(shortlist(2), {"0001": text, "customer-3": "=SUM(1,2)"})
        frame.loc[0, "id"] = "@malicious-id"
        before = frame.copy(deep=True)
        content = conversation_guides_csv_bytes(frame)
        self.assertTrue(content.startswith(b"\xef\xbb\xbf"))
        decoded = pd.read_csv(io.BytesIO(content), dtype={"id": str}, keep_default_na=False)
        self.assertEqual(decoded.loc[0, "conversation_guide"], text)
        self.assertEqual(decoded.loc[1, "conversation_guide"], "'=SUM(1,2)")
        self.assertEqual(decoded.loc[0, "id"], "'@malicious-id")
        assert_frame_equal(frame, before)
        for risky in ["+1", "-1", "@calc", "  =1", "\tcmd", "\rline", "\nline"]:
            altered = frame.copy()
            altered.loc[0, "conversation_guide"] = risky
            decoded = pd.read_csv(io.BytesIO(conversation_guides_csv_bytes(altered)), keep_default_na=False)
            self.assertEqual(decoded.loc[0, "conversation_guide"], "'" + risky)
        for invalid in [None, frame.drop(columns="guide_status")]:
            with self.assertRaises(ProfileValidationError):
                conversation_guides_csv_bytes(invalid)


if __name__ == "__main__":
    unittest.main()
