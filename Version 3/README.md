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
| **ProfileEncoder** | ✅ wired in | Trained from `06_ProfileEncoder_revised.ipynb` against `newcomer_ontario_enriched.csv` — see metrics below. |
| **Employment XGBoost** | ✅ wired in | `artifacts/xgboost/Secondversion_XGB_classifier.pkl` |
| **Income XGBoost** | ✅ wired in | `artifacts/xgboost/Secondversion_XGB_regressor.pkl` |
| **Adzuna API** | ✅ wired in | Uses `adzuna_client.py`; **rotate the API key** (see Security note) |
| **Recommendation Engine** | ✅ wired in | Bolaji's `recommendation_engine.py` + `explanation.py`, unchanged |
| **Folium map** | ✅ wired in | Job markers colored by match score, plus a housing-data overlay from `housing_geocoded.csv` |

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

**Test-set metrics from this training run** (also saved in
`artifacts/profile_encoder/eval_reference/profile_encoder_v1_1_test_results.csv`):

| Metric | Value |
|---|---|
| Employment accuracy | 70.9% |
| Employment macro F1 | 0.620 |
| Income MAE | $14,032 |
| Income RMSE | $18,843 |
| Income R² | 0.638 |
| Occupation-category accuracy | 60.2% |

A few things worth knowing about these numbers before treating them as final:
- **Occupation category is imbalanced and noisy at the tail.** Precision/recall
  per class range from very strong (Natural & Applied Sciences: 0.97 recall,
  Health: 0.98 recall) to weak (Manufacturing & Utilities: 0.36 recall,
  Sales & Service: 0.44 recall) -- those two categories are the ones most often
  confused with each other and with Trades. If category quality matters more
  than it currently performs, more training data or feature engineering for
  those specific categories would help most.
- **Employment recall on "Not Employed" is low (51%)** even with `class_weight="balanced"` --
  the model still leans toward predicting "Employed." The 0.40 threshold
  (chosen in the notebook via validation-set macro-F1) is already tuned for
  this; a different threshold trades precision/recall differently if the
  app's use case cares more about one class.
- These are Version 1.1 numbers on one train/val/test split with `SEED = 42`,
  exactly as the notebook defines it -- retraining will shift them slightly.
- `eval_reference/profile_encoder_recommendation_inputs_v1_1.csv` has the
  embedding + all three predictions for every one of the 6,900 profiles in
  the dataset, useful for offline evaluation, nearest-neighbour sanity checks,
  or batch recommendations without going through the live Streamlit form.

If you retrain later (more data, tuned architecture, etc.), just overwrite
these six files with the new ones -- no code changes needed.

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

## Deploying to Streamlit Community Cloud

Yes — this is a single-file Streamlit app with no external services besides
Adzuna, so it deploys cleanly to [share.streamlit.io](https://share.streamlit.io)
for free. Steps:

1. **Push this folder to a GitHub repo** (public or private — Streamlit Cloud
   can access private repos once you connect your GitHub account).
   ```bash
   cd integrated_app
   git init
   git add .
   git commit -m "Integrated newcomer career navigator pipeline"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
   `.gitignore` already excludes `.streamlit/secrets.toml` — never commit your
   real Adzuna keys.

2. **Go to share.streamlit.io → New app**, pick the repo/branch, and set the
   main file path to `app.py`.

3. **Add your Adzuna keys under the app's Settings → Secrets**, in the same
   `key = "value"` TOML format as `.streamlit/secrets.toml.example`:
   ```toml
   ADZUNA_APP_ID = "your_real_app_id"
   ADZUNA_APP_KEY = "your_real_app_key"
   ```
   Do this before sharing the deployed URL with anyone — otherwise it silently
   falls back to the old exposed demo key baked into `adzuna_client.py`.

4. **Deploy.** First build takes a few minutes (mostly installing
   `tensorflow-cpu`); redeploys after that are fast since the environment is
   cached.

### Things worth checking on the free tier

- **Memory (1 GB on the free tier):** the actual model files are tiny
  (~1.2 MB combined), but `tensorflow-cpu` itself uses a few hundred MB of RAM
  once imported. This app comfortably fits, but if you later add more heavy
  dependencies (e.g. `shap`), keep an eye on memory during testing.
- **Cold starts:** free-tier apps sleep after inactivity; the first request
  after waking will be slower while TensorFlow re-initializes. `st.cache_resource`
  is already used for all model loads so this only happens once per wake, not
  per request.
- **`requirements.txt` is pinned** to the exact versions this app was built
  and tested against, and `runtime.txt` pins Python to 3.12 — this avoids
  Streamlit Cloud resolving a newer TensorFlow/Keras or scikit-learn version
  that can't load these specific `.keras`/`.pkl` files. If you need to loosen
  a version, re-test the ProfileEncoder and XGBoost loads specifically (see
  the comment at the top of `requirements.txt`).
- **Other hosts work too** if you'd rather not use Streamlit Community Cloud —
  Hugging Face Spaces (Streamlit SDK), Render, Railway, or your own Docker
  container all run this app as-is; only the secrets-configuration step
  changes (environment variables instead of `st.secrets` on most of those,
  which `adzuna_client.py` already falls back to automatically).

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
