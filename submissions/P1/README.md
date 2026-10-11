# P1 submission handoff

The combined PDF is the P1 content submission. The course Assignment Validation
form is kept locally and submitted separately to the professor. It is not
published in this repository. The professor's blank source form is:
https://github.com/shylasolis/machine-learning-pathway/blob/main/assignment-validation.pdf

## Files and status

- EV_Customer_Prioritization_P1.pdf: eight pages with the proposal, realistic topic examples, a single-page topic/industry infographic, initial data profile and interpreted figures, baseline evidence, and next steps. The infographic is page 3.
- Assignment Validation (local only): the prefilled official form is retained on the author's computer. Peer name, checks, date, and both signatures are pending. Submit the completed form separately to the professor; do not commit or upload it to GitHub.
- bonus_plan.json: planned evidence for all three optional extensions; these features are not implemented in P1.

## Independent peer replication

1. Clone the repository from https://github.com/Nuquynhchaunguyen/ev-customer-prioritization or pull the P1 submitted version. Record the commit being reviewed.
2. Follow the root README for Python 3.13 or 3.12 and create a new local environment. Do not copy the author's Mac venv.
3. Obtain the three competition CSVs through the documented Kaggle API workflow with your own account and authorized access.
4. Run the EDA and modeling runners from the root, then the automated checks. Confirm the expected outputs listed in README. The new modeling run recreates the splits/model/results; retained technical reports show the author's current run.
5. Start Streamlit, try the supplied demo, select K, and confirm the downloaded shortlist contains exactly K rows with the correct profile information. The app can use the supplied saved model without retraining.
6. Record replication evidence, resolve errors before submission, and complete the professor's checks and signatures only after the work is actually verified.

The README gives Mac and Windows commands. Windows execution is not claimed as
already verified; the independent reviewer records their actual platform and result.
For P1 without runnable code, the final-project instructions allow file/instruction
verification; this P1 handoff includes runnable work and should be checked accordingly.

## Before submission

- Student reviews and can explain the proposal, figures, model comparison, K, Precision@K, Lift@K, synthetic-data limits, and their own decisions.
- A peer completes independent replication and the official form truthfully.
- Submit the combined PDF and accessible GitHub link through the course assignment. Submit the newly completed Assignment Validation separately to the professor, as required; keep the form local and outside GitHub.
- Keep a record of professor feedback for P2. No P1/P2 feedback has been received or invented here.

P1 contains plans for FastAPI, Docker, and MLOps, each worth up to five optional
points in the professor rubric. Credit can be claimed only after the features
work and reproducible evidence is available in P2/final delivery.
