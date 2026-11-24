# Product demonstration

This walkthrough presents the project to a sales manager and then shows the
technical reproduction evidence. Use the included synthetic demo. No real
customer contact is part of the demonstration.

## Prepare

Open the project folder, activate the environment, and complete installation
using the root README. Test the fitted artifact first.

```bash
python -m ev_project.check_saved_model
```

Expected messages include **Ranked 500 demo profiles and shortlisted 50.** and
**Ranking matches after reload. Maximum score difference: 0.**

Open `EV_Customer_Prioritization_Final_Presentation.pptx`. Its speaker notes
provide the numerical details and source references for each slide.

## Show the staff workflow

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** and keep the terminal running.

1. Explain the illustrative decision: 1,000 profiles, 50 available contacts,
   and $250 in contact labor under the stated assumptions.
2. Choose **Try the demo**, set K to **50**, and click **Create shortlist**.
3. Confirm the list contains exactly **50** selected profiles. Explain that
   scores estimate the supplied synthetic Yes label.
4. Select the first profile. Check the ID **668721** and its estimated score
   of about **0.930806**, then review the recorded travel and charging facts.
5. Edit one line of its conversation guide. Explain that staff confirm the
   facts and approve any contact. Editing the guide leaves the ranking score
   unchanged.
6. Download the shortlist CSV and the selected guide. Confirm that the CSV has
   50 rows and that the guide refers to the selected ID. For bulk guides,
   prepare and download the guide CSV for the current shortlist.
7. Explain the finding in plain language: the highest-ranked 5% of the
   held-out synthetic sample has about 85% Yes labels, compared with about
   17% overall. This does not imply 85% of real contacts will purchase.

Press **Ctrl+C** in the server terminal when finished. Guide edits remain in
the browser session until exported.

## Show service and monitoring evidence

Follow the root README's API instructions in one terminal. In a second
activated terminal run the portable HTTP check.

```bash
python -m ev_project.check_api
```

Expect **API verified: ranked 500 profiles and shortlisted 50.** and
**Invalid inputs return HTTP 422. OK**. Open **http://127.0.0.1:8000/docs**
to inspect the request schema and response. The HTTP check compares all
scores and ranks with the saved pipeline and exercises invalid inputs.

Show the packaged service using the Docker instructions in
[API and Docker](api_docker.md). The same HTTP checker runs against port
8001. This demonstrates a built and running image with the saved artifact.

Run the monitoring example in the full Python environment.

```bash
python -m ev_project.monitoring --demo
```

Expect **Baseline: stable; shifted batch: investigate.** Open the generated
chart at `docs/reports/monitoring/drift_demo.png`. The baseline and deliberately
shifted inputs test the alert behavior. An alert calls for analyst review of
data collection, units, and population changes. It does not prove lower accuracy
or trigger automatic retraining.

## Close with the stakeholder decision

Recommend a small controlled pilot with permitted real data, staff review, and
an equal-capacity comparison against the current selection rule. Agree a
follow-up window and track contact time, appointments, verified purchases,
and outcomes across customer groups. Evaluate incremental results and all
relevant costs before expanding use.
