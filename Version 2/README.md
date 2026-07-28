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
│   ├── profile_encoder.py        # Stage 1 (+ heuristic fallback, see below)
│   ├── xgb_models.py              # Stage 2 + 3 (employment + income)
│   ├── adzuna_client.py           # Stage 4
│   ├── recommendation_engine.py   # Stage 5 (from Bolaji's package, unchanged)
│   └── explanation.py             # Stage 5 helper (unchanged)
├── artifacts/
│   ├── xgboost/                   # the 4 .pkl files (already included)
│   └── profile_encoder/           # only the config .json is included — see below
├── data/
│   └── housing_geocoded.csv       # for the Folium map overlay
├── requirements.txt
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
| **ProfileEncoder** | ⚠️ **fallback active** | See below — the trained `.keras`/`.joblib` files weren't in the handoff, only the config JSON. |
| **Employment XGBoost** | ✅ wired in | `artifacts/xgboost/Secondversion_XGB_classifier.pkl` |
| **Income XGBoost** | ✅ wired in | `artifacts/xgboost/Secondversion_XGB_regressor.pkl` |
| **Adzuna API** | ✅ wired in | Uses `adzuna_client.py`; **rotate the API key** (see Security note) |
| **Recommendation Engine** | ✅ wired in | Bolaji's `recommendation_engine.py` + `explanation.py`, unchanged |
| **Folium map** | ✅ wired in | Job markers colored by match score, plus a housing-data overlay from `housing_geocoded.csv` |

### ProfileEncoder — needs one more step from you

`06_ProfileEncoder_revised.ipynb` trains a multi-task Keras network and, at
the end, saves five files:

- `profile_encoder_v1_1.keras`
- `profile_multitask_model_v1_1.keras`
- `profile_encoder_preprocessor_v1_1.joblib`
- `profile_encoder_income_scaler_v1_1.joblib`
- `profile_encoder_category_encoder_v1_1.joblib`

Only `profile_encoder_config_v1_1.json` made it into the zip you shared —
the other five weren't there. **Re-run that notebook and drop those five
files into `artifacts/profile_encoder/`.** `pipeline/profile_encoder.py`
already knows how to load them (it rebuilds the same custom `FeatureSlice`
layer the notebook defines) — no code changes needed once the files exist.

Until then, `encode_profile()` transparently falls back to a keyword-matching
heuristic for `predicted_occupation` and a neutral `profile_fit_score`, so
the rest of the pipeline can be built, tested, and demoed today. The app
shows a warning banner whenever the fallback is active, and the **Pipeline
Status** page in the sidebar shows exactly which artifacts are (and aren't)
found.

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
models ever get retrained, check `classifier.get_booster().feature_names`
again — if the retrained version uses `occupation_category` consistently,
delete the rename step.

## Security note — Adzuna API key

`adzuna_api.py` had a working `APP_ID`/`APP_KEY` pair hardcoded directly in
the source file, which means it's already sitting in plain text anywhere
this zip was shared. Adzuna keys are free to reissue at
<https://developer.adzuna.com/> — get a new pair and put them in
`.streamlit/secrets.toml` (see `secrets.toml.example`) rather than in source
code. `adzuna_client.py` checks Streamlit secrets first, then environment
variables, and only falls back to the old hardcoded pair as a last resort so
the app still runs before you've set anything up.

## Notes

- A `LabelEncoder version mismatch` warning may print on load (the pickles
  were saved with a slightly older scikit-learn). It's a warning, not an
  error — everything still runs correctly — but re-pickling the encoders
  with your current scikit-learn version will make it go away.
- The old `Streamlit_Integration/App_Versions/app_v.1.py` is a simpler,
  standalone employment-only demo (7 features, SHAP waterfall chart, no
  ProfileEncoder/Adzuna/recommendation engine). It's not part of this
  integrated app, but its SHAP-explanation approach could be added as an
  extra expander in Stage 2's results if you want per-prediction SHAP
  charts alongside the recommendation engine's plain-language explanations.
