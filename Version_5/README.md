# Newcomer Career Navigator — Integrated Pipeline

This folder wires together every piece of Capstone Group 4 into **one Streamlit
app** that implements the full pipeline:

```
User Input
   -> ProfileEncoder            (predicted occupation, profile fit score)
   -> Employment XGBoost        (employment probability)
   -> Income XGBoost            (predicted annual income)
   -> Adzuna API                (available jobs)
   -> Recommendation Engine     (ranked jobs, match scores, explanations)
   -> Streamlit Dashboard + Folium interactive map
```

## Folder structure

```
integrated_app/
├── app.py                       # the Streamlit app — run this
├── pipeline/
│   ├── config.py                # every file path + feature list, in one place
│   ├── profile_encoder.py        # Stage 1
│   ├── xgb_models.py              # Stage 2 + 3 (employment + income)
│   ├── adzuna_client.py           # Stage 4, live API mode
│   ├── job_source.py              # Stage 4, offline dataset mode (default)
│   ├── recommendation_engine.py   # Stage 5 (from Bolaji's package, unchanged)
│   └── explanation.py             # Stage 5 helper (unchanged)
├── artifacts/
│   ├── xgboost/                   # the 4 .pkl files
│   └── profile_encoder/           # the trained ProfileEncoder network
├── data/
│   ├── housing_geocoded.csv       # for the Folium map overlay
│   └── processed_adzuna_jobs.csv  # offline job listings (Stage 4 default)
├── requirements.txt
├── runtime.txt
└── .streamlit/secrets.toml.example
```

## Setup

```bash
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then fill in your Adzuna keys
streamlit run app.py
```

## What's plugged in vs. what's still a stand-in

| Stage | Status | Notes |
|---|---|---|
| **ProfileEncoder** | ✅ wired in | Trained from `06_ProfileEncoder_revised.ipynb` against `newcomer_ontario_enriched.csv` — see metrics below. |
| **Employment XGBoost** | ✅ wired in | `artifacts/xgboost/Secondversion_XGB_classifier.pkl` |
| **Income XGBoost** | ✅ wired in | `artifacts/xgboost/Secondversion_XGB_regressor.pkl` |
| **Adzuna API** | ✅ wired in, **two modes** | Live API, or an offline dataset already collected from Adzuna — see below. Default is offline, no key needed. |
| **Recommendation Engine** | ✅ wired in | Bolaji's `recommendation_engine.py` + `explanation.py`, unchanged |
| **Folium map** | ✅ wired in | Job markers colored by match score, plus a housing-data overlay from `housing_geocoded.csv` |

### Job listings — offline dataset or live API, your choice

The sidebar has a **"Job data source"** toggle:

- **Local dataset (offline, default)** — serves jobs from
  `data/processed_adzuna_jobs.csv`, 369 postings Adzuna previously returned
  for a "data analyst" search in Toronto (collected by the original
  `adzuna_job_analysis.py`). No API key needed, no network call, fully
  reproducible for demos. `pipeline/job_source.py` filters it by keyword and
  city; if nothing matches (e.g. a very different occupation or a city
  outside the Toronto area) it falls back to the closest available matches
  rather than returning nothing.
- **Live Adzuna API** — the original `adzuna_client.py` behavior: real-time
  results for whatever city/keyword the user enters, anywhere Adzuna covers.
  Needs a valid key (see Security note below).

Both paths return the same columns, so `recommend_jobs()` and the Folium map
don't need to know which one is active. The **Pipeline Status** page shows
the offline dataset's size, company count, location count, and date range.

**Trade-off to know about:** the offline dataset skews toward IT/data/analyst
roles in the Toronto area (that's what it was originally searched for) and
it's frozen at whatever Adzuna returned when it was collected — it won't
reflect newly posted or removed jobs. It's genuinely useful for demos, offline
testing, and avoiding API rate limits, but it's not a substitute for the live
API in production once you have your own Adzuna key. If we want broader
offline coverage, run `adzuna_job_analysis.py` (or a few more searches with
different keywords/cities) and append the results to
`data/processed_adzuna_jobs.csv` — the schema just needs to match the
existing columns.

### ProfileEncoder — now trained and included

The five files `06_ProfileEncoder_revised.ipynb` produces are now in
`artifacts/profile_encoder/`:

- `profile_encoder_v1_1.keras`
- `profile_multitask_model_v1_1.keras`
- `profile_encoder_preprocessor_v1_1.joblib`
- `profile_encoder_income_scaler_v1_1.joblib`
- `profile_encoder_category_encoder_v1_1.joblib`

`pipeline/profile_encoder.py` loads these automatically -- `encode_profile()`
now returns `used_fallback=False` and a real 16-dimensional embedding.
`tensorflow-cpu` is a required dependency now (see `requirements.txt`).


## Bug found and fixed during integration

The two XGBoost models in `Austine Handoff/` (`Secondversion_XGB_classifier.pkl`
and `Secondversion_XGB_regressor.pkl`) were fit with their 10th input column
literally named **`employment_category`**, but the accompanying label encoder
dict (`Secondversion_EDA_preprocessing.pkl`) — and every other artifact,
including the ProfileEncoder — key that same column as **`occupation_category`**.
It's the same underlying feature (same 9 category values), just a naming
mismatch baked into those two pickles at training time. XGBoost's strict
feature-name validation rejects the row otherwise.

Fix applied in `pipeline/xgb_models.py` (`_XGB_COLUMN_RENAME`): the column is
renamed right before calling `.predict()` / `.predict_proba()`. If those two
models ever get retrained, we check `classifier.get_booster().feature_names`
again — if the retrained version uses `occupation_category` consistently,
delete then rename step.

## Security note — Adzuna API key

`adzuna_api.py` had a working `APP_ID`/`APP_KEY` pair hardcoded directly in
the source file, which means it's already sitting in plain text anywhere
this zip was shared. Adzuna keys are free to reissue at
<https://developer.adzuna.com/> — get a new pair and put them in
`.streamlit/secrets.toml` (see `secrets.toml.example`) rather than in source
code. `adzuna_client.py` checks Streamlit secrets first, then environment
variables, and only falls back to the old hardcoded pair as a last resort so
the app still runs before you've set anything up.


### Things worth checking on the free tier

- **Memory (1 GB on the free tier):** the actual model files are tiny
  (~1.2 MB combined), but `tensorflow-cpu` itself uses a few hundred MB of RAM
  once imported. This app comfortably fits.

- **`requirements.txt` is pinned** to the exact versions this app was built
  and tested against, and `runtime.txt` pins Python to 3.12 — this avoids
  Streamlit Cloud resolving a newer TensorFlow/Keras or scikit-learn version
  that can't load these specific `.keras`/`.pkl` files.

## Notes

- The old `Streamlit_Integration/App_Versions/app_v.1.py` is a simpler,
  standalone employment-only demo (7 features, SHAP waterfall chart, no
  ProfileEncoder/Adzuna/recommendation engine). It's not part of this
  integrated app, but its SHAP-explanation approach could be added as an
  extra expander in Stage 2's results if needed per-prediction SHAP
  charts alongside the recommendation engine's plain-language explanations.
