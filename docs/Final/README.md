# Final project package

| Material | File |
|---|---|
| Final business report | [PDF](EV_Customer_Prioritization_Final_Report.pdf) |
| Readable results | [Standalone HTML](EV_Customer_Prioritization_Final_Report.html) |
| Stakeholder presentation | [PowerPoint](EV_Customer_Prioritization_Final_Presentation.pptx) |
| Product demonstration | [Walkthrough](demo_walkthrough.md) |
| Scoring service and container | [API and Docker](api_docker.md) |
| Input monitoring | [Reference, demonstration, and investigation rules](monitoring.md) |
| Project revisions | [Revision log](revision_log.md) |
| Final owner and peer handoff | [Submission checklist](submission_checklist.md) |
| Final requirement review | [November 24 review](final_review.md) |
| Actual verification records | [Evidence folder](evidence/) |
| Editable report source | [Report content](report_content.md) |

The primary implementation is the Python source in `ev_project/` and `app.py`.
The notebooks and the EDA/modeling HTML outputs provide supporting analysis.
Follow the [root README](../../README.md) for the environment, installation,
Kaggle data access, commands, and expected output. The included saved model,
synthetic demo, and input reference allow the scoring, API, and monitoring
checks without downloading the full dataset.

Verification passed 74 automated tests on macOS ARM64. The service also built
and ran in Linux ARM64 and emulated Linux AMD64 containers, with 21 extension
tests passing in each. Windows x64 Python 3.13 prebuilt-package compatibility
was checked separately. The evidence files state exactly what was run.

Owner review, publishing the reviewed version, independent peer reproduction,
the signed validation form, and the live presentation are final handoff steps.
