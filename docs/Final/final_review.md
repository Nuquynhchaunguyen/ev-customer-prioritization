# Final readiness review

**Reviewed:** November 24, 2026  
**Scope:** The local project package checked against the supplied Final Project assignment, including all three optional extensions.

The local report, analysis, saved artifact, stakeholder presentation and extensions have been checked. Publication, independent classmate reproduction, the newly signed Final form and the actual presentation remain separate submission steps. This review records evidence, not a predicted grade.

## Required final work

| Requirement | Verified local evidence |
|---|---|
| Complete business report | The [17-page PDF](EV_Customer_Prioritization_Final_Report.pdf) contains all ten required sections, sources, captions, interpretations, limitations and an evidence-based pilot recommendation. |
| Complete data-to-action visual | Section 6 shows original icons and arrows from Kaggle through preparation, modeling and delivery to staff decisions and proposed outcome feedback; completed and planned work are labeled. |
| Analysis and suitable evaluation | EDA and modeling notebooks ran again without error outputs. Split, model comparison, holdout metrics, ranking and confusion tables matched the saved results exactly. Training, validation and holdout boundaries are documented. |
| Business recommendation | The report and slides connect the limited-contact decision to ranking evidence, illustrative labor cost, human review and a controlled real-data pilot. Synthetic label agreement is not presented as measured sales or ROI. |
| Source, data and readable output | Python source, supporting notebooks, standalone Final HTML and two analysis HTML reports are present. Data provenance, schema, Kaggle access and a synthetic demo are documented. A new isolated authorized download produced all three expected source CSVs with matching hashes. |
| Saved fitted artifact | The saved preprocessing/model pipeline loads in a fresh process. It ranks 500 demo profiles and selects exactly 50; first ID 668721, Yes score 0.930806, maximum reload difference 0. |
| Instructions and dependencies | The root README separates setup, commands and bold expected messages; it provides macOS and Windows setup and run instructions, including API, Docker and monitoring checks. Windows package resolution is verified; native Windows execution is still unverified. |
| Stakeholder presentation and product | The [10-slide PowerPoint](EV_Customer_Prioritization_Final_Presentation.pptx) uses plain business language, an actual product capture, interpreted results, limits and a pilot decision. Speaker notes and the [walkthrough](demo_walkthrough.md) support the product demonstration. |
| Feedback and revision log | The [revision log](revision_log.md) maps all nine P1/P2 feedback criteria to Final evidence. It preserves the positively reviewed work and distinguishes new Final improvements from instructor requests. |

## Optional extensions

All three extensions can be submitted together under the assignment's combined maximum of 15 bonus points. Each is assessed separately, up to five points.

| Extension | Actual verification |
|---|---|
| FastAPI | Live local HTTP check ranked 500 profiles and returned 50, with identical ranks and scores to the saved pipeline. Four intentionally invalid requests returned HTTP 422. All 11 API tests passed. |
| Docker | Current service image built using cached layers and ran as a non-administrator in Linux ARM64. Health, dependency check, 21 extension tests and live HTTP parity passed. Earlier Linux AMD64 emulation evidence remains identified as a November 23 run. |
| MLOps | Reference provenance verifies 468,065 training-only profiles, matching source, membership and model hashes, and no outcome labels. The included demo is stable; a deliberately shifted copy triggers investigation. Ten monitoring tests pass. The investigation policy explains why input drift does not establish prediction accuracy. |

The complete suite passed **74 tests in 3.212s**, ending with **OK**, on macOS ARM64 with the pinned Python 3.13.9 project environment. Durations vary. The analysis rerun reused that installed environment; it was not a new package installation or an independent classmate run.

Detailed measurements, tested source hashes and platform limits are in [the November 24 evidence record](evidence/final_audit_2026-11-24.json). Earlier run records remain available separately. The fitted application model and reported analysis results were not changed by the document clarifications.

## Focused corrections from this review

- Clarified that Figure 1 covers all 668,665 labeled source profiles before the 70/15/15 split.
- Renamed the appendix's separate diagnostic evidence as a fresh training trace.
- Made the slide cost expression explicit: 50 contacts × $5 each = $250.
- Replaced two technical phrases on stakeholder slides and separated adjacent chart category labels without changing chart values or its embedded workbook.

## Before submission

1. Review the final PDF, HTML, slides and README, and rehearse the product demonstration.
2. Publish the exact reviewed version when the owner authorizes publication, then verify access for the instructor and classmate.
3. Have a classmate independently clone or pull that version and follow the README without assistance. Record real outputs and any platform issues; correct failures before signing.
4. Complete and sign a new Final Assignment Validation form. Submit it separately with the repository link and required deliverables. Prior P1/P2 forms do not replace it.

Credentials, full source CSVs, local environments and private signed forms remain excluded from the public package. Follow the [submission checklist](submission_checklist.md) to record the external steps when they are actually complete.
