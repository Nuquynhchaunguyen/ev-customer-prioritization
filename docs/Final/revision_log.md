# Final revision log

**Project owner:** Nu Quynh Chau Nguyen  
**Review date:** November 24, 2026

This log records the response to P1/P2 instructor feedback and the independently implemented improvements for Final.

## P1 and P2 instructor feedback response

The project-specific Canvas feedback posted on November 24, 2026 was reviewed from the screenshots supplied by the project owner. It identifies strengths across the reviewed criteria and requests no specific corrective revision. Final preserves those reviewed elements and documents the additional implementation and verification separately.

| Reviewed criterion and feedback summary | Response in Final | Evidence location |
|---|---|---|
| P1 topic examples and infographic connect methods to business decisions | Retain the six examples, cited EV infographic, and connection between the topic and project question | Archived P1 proposal PDF; Final Section 4 and references |
| P1 fixed-capacity question, stakeholder, and precision/lift measures are clearly defined | Retain the stakeholder question and capacity objective | Final Sections 1, 2, and 7; stakeholder slides 2, 6, and 7 |
| P1 synthetic-data limitations, target, initial class distribution, and majority baseline are documented | Retain provenance, field limits, and explicit baseline comparison | Final Sections 5–7; data dictionary and evaluation tables |
| P1 package, repository, and signed peer form provide the expected handoff evidence | Preserve the archived P1 materials; prepare a separate Final handoff | docs/P1/; docs/Final/submission_checklist.md; local Final validation form |
| P2 code and report show preparation, model selection, holdout evaluation, and ranked decision support | Retain the evaluated analysis and its split boundaries; extend delivery and monitoring | Final Sections 5–8; ev_project/modeling.py; saved model and metadata |
| P2 capacity precision/lift, completed/planned pipeline, and synthetic-data limits are interpreted | Retain the capacity results and interpretations; update the pipeline for completed extensions | Final Sections 6–7; docs/Final/figures/data_to_action.png |
| P2 handoff includes the saved model, load check, instructions, dependencies, and checked signed form | Retain artifact checks and extend reproduction instructions for API, Docker, and monitoring | Final Section 8; README steps 7 and 9–11; docs/Final/evidence/ |
| P2 paper and stakeholder PowerPoint explain the problem, evidence, and controlled-pilot recommendation | Preserve the stakeholder focus and pilot recommendation | Final Sections 1, 2, and 9; Final stakeholder PowerPoint |
| P2 documents the development from the P1 scenario to tested models | Record the actual instructor feedback and map the retained work and Final additions | This log; Final Appendix B |

The instructor's recognition of the signed P1/P2 forms applies to those checkpoints. Final still requires its own independent peer replication and signatures. The feedback does not call for replacing the model or changing the reported holdout findings.

## Independent improvements from P2 to Final

| Area | P2 state | Final revision and evidence |
|---|---|---|
| Final communication | Technical review draft and draft stakeholder slides | Ten-section final report source in `docs/Final/report_content.md`; final PDF, HTML and stakeholder presentation prepared alongside it |
| Complete workflow | Source-to-app pipeline; service packaging and monitoring planned | Final original-icon pipeline in `docs/Final/figures/data_to_action.png`, with completed software components and future outreach/outcome feedback distinguished |
| Service access | Streamlit and Python scoring | FastAPI implementation in `ev_project/api.py`; live HTTP checker in `ev_project/check_api.py`; API regression checks in `tests/test_api.py`; common saved-model scoring contract retained |
| Portable execution | Python environment setup | Built and ran Linux ARM64 and emulated Linux AMD64 service images; both healthy, all 21 extension tests passed, and live score parity verified in `docs/Final/evidence/docker_arm64.json` and `docker_amd64.json` |
| Input monitoring | Schema and observed-range warnings | `ev_project/monitoring.py`, model-linked reference at `models/monitoring_reference.json`, baseline and deliberately shifted-batch evidence under `docs/reports/monitoring/`; investigation rules distinguish drift from accuracy |
| Data provenance | Manifest and table dictionary available in reports | Consolidated access and schema notes in `data/README.md`, preserving authorized Kaggle downloads |
| Reproduction instructions | Separate macOS/Windows setup, saved-model demonstration, 53-test base suite | Expanded suite passed 74 tests in 15.281 seconds, with `pip check` clean; `docs/Final/evidence/automated_tests.json` records the run. Full Windows x64 Python 3.13 package resolution passed separately. |
| Evidence precision | Historical reload JSON used K = 25 while terminal demo used K = 50 | Final report explicitly distinguishes the historical evidence from the current default check; both concern score consistency rather than predictive accuracy |
| Command accuracy | P2 report p. 8 contains `unitteest` | Final report uses the correct `python -m unittest discover -s tests -v` |
| Scope and decision limits | Synthetic-label evaluation and proposed dealership pilot | Limits retained for unverified purchase probabilities, demographic fairness, causal effects, costs, and outcomes; no sales or ROI claim added |

The new software paths extend delivery and monitoring of the evaluated saved model. They do not establish stronger predictive performance. The model choice, split rules and reported holdout findings remain traceable to `docs/reports/modeling/` and `models/model_metadata.json`.

## Verification and external handoff

The P2 base suite was rerun locally on November 23, 2026 and all 53 existing tests passed. The expanded suite passed all 74 tests, and the local HTTP and both container checks passed. The root README and `docs/Final/evidence/` record the actual run results. Test durations vary by machine.

The final package is being prepared locally at the project owner's request. Publication, instructor submission, independent peer replication, and peer signatures must be confirmed separately. Automated checks do not stand in for an independent peer run. Native Windows execution must be reported only if actually performed; dependency resolution or Linux container execution is not that evidence.

The November 24 final audit reran EDA, modeling, artifact loading and the full suite in an isolated copy. All evaluation tables and demo scores matched exactly; **74 tests passed in 3.212 seconds**, ending with **OK**. Local HTTP, a new cached Linux ARM64 container build/run and monitoring checks also passed. The independent requirements review found no missing major local deliverable. It clarified the source-population caption and fresh diagnostic provenance, simplified stakeholder wording and cost arithmetic, and improved chart-label spacing. These were audit improvements, not additional instructor correction requests. See `docs/Final/final_review.md` and `docs/Final/evidence/final_audit_2026-11-24.json` for current scope and limitations.

## Final review actions

- Review the final paper, presentation, README and actual run evidence.
- Publish the reviewed project when the owner is ready, then confirm instructor and peer access to the repository link.
- Ask a peer to clone or pull that exact reviewed version and follow the README without assistance.
- Fix replication errors and obtain the newly completed Final Assignment Validation form and signatures.
- Submit the final deliverables, accessible link and validation form through the required course channel.

The signed validation form is submitted separately and remains excluded from version control under the existing project policy.
