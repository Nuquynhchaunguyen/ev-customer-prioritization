# Final submission review

**Owner:** Nu Quynh Chau Nguyen  
**Prepared for review:** November 23, 2026  
**Feedback review:** November 24, 2026  
**Full verification:** November 24, 2026 — [requirement review and evidence](final_review.md)

The final package is prepared locally for owner review. This checklist distinguishes the deliverables to inspect from external handoff steps that remain to be completed. It is not a certification of peer replication or grading.

## Deliverables to inspect

- [x] P1/P2 instructor feedback reviewed and mapped to retained work and Final additions in the revision log and report Appendix B.

- [x] Final business report PDF follows the ten required sections and interprets its figures and tables.
- [x] Readable final HTML and the two analysis HTML outputs open successfully.
- [x] Stakeholder PowerPoint explains the problem, key finding, recommendation, limitation and working tool in plain language.
- [x] Pipeline uses icons and arrows, names the actual tools, and separates completed prototype components from proposed outreach and outcome feedback.
- [x] Python source and supporting notebooks/modules reproduce the documented workflow.
- [x] Data source, competition access, schema, cleaning decisions and limitations are documented in `data/README.md` and the report.
- [x] Root README gives environment, installation, Kaggle data access, run order, expected output and saved-model verification.
- [x] Saved fitted preprocessing/model and metadata are included; fresh load/use check passes.
- [x] Original analysis metrics match the report's validation and holdout tables.
- [x] FastAPI accepts a valid batch, returns useful ranking, and returns clear errors for invalid inputs.
- [x] Container image builds and runs with the required artifact; the documented live check passes against it.
- [x] Monitoring reference is linked to the saved model; unchanged and deliberately changed input batches reproduce the documented behavior.
- [x] Monitoring documentation explains when to investigate and why drift does not prove lost accuracy.
- [x] Test results and live check evidence are actual run results, with platform limits stated accurately.
- [x] Credentials, private customer data, local environments and signed forms are excluded from the repository.

## Owner and peer handoff

- [ ] Owner has reviewed the final PDF, HTML, PowerPoint and README.
- [ ] Reviewed project version has been published when the owner authorizes it.
- [ ] Repository link is accessible to the professor and peer.
- [ ] Peer has independently cloned or pulled that exact version and followed the README without help.
- [ ] Peer has recorded actual output reproduction, including any platform-specific issue.
- [ ] Replication errors have been corrected and relevant checks rerun.
- [ ] New Final Assignment Validation form is completed and signed, then submitted separately.
- [ ] Final report, HTML, source/data-access package, saved artifact, PowerPoint, repository link and validation form are submitted through the course channel.
- [ ] Owner has rehearsed the presentation and product demonstration and can explain the methods and limitations.

## Suggested product demonstration

1. Start the Streamlit application using the README.
2. Select the included 500-profile demo and set contact capacity to 50.
3. Explain the resulting list as a review order based on synthetic labels.
4. Open one customer, confirm recorded commute/charging facts, and edit a guide.
5. Download the shortlist and updated guide; confirm 50 rows and correct customer ID.
6. Show a valid API result, an invalid-input response, and the monitoring shifted-batch example using the documented checks.
7. End with the proposed equal-capacity dealership pilot and the need for verified outcomes.

