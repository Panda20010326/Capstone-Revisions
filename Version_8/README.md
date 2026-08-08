# Career Navigator — Version 10

Career Navigator is a Streamlit capstone application for newcomer career planning in Ontario.

## Runtime architecture

`app.py` calls only inference/recommendation modules:

1. `pipeline/profile_encoder.py`
2. `pipeline/xgb_models.py`
3. `pipeline/adzuna_client.py` / `pipeline/job_source.py`
4. `pipeline/karthika_recommendation.py`
5. dashboard + Folium map

Training notebooks remain in `notebooks/` for documentation and reproducibility. The Karthika recommendation notebook has a runtime Python conversion at `pipeline/karthika_recommendation.py`.

## Version 10 model improvement

The employment classifier now uses a tuned XGBoost model with **sigmoid probability calibration**. The supplied dataset has an employment base rate of approximately 78.1%, so calibrated probabilities are designed to reflect that base rate instead of acting like overconfident raw scores.

Validation on the held-out test split:

- ROC-AUC: ~0.710
- Brier score: ~0.155
- Mean predicted probability: ~77.8%
- Observed employment rate: ~78.1%
- Probability range: approximately 40%–91%

This is a probability calibration improvement, not an artificial rule that forces scores into a particular range.

## Custom branding

The app includes:

- `assets/career_navigator_logo.png` — browser/page icon and Streamlit sidebar logo
- `assets/career_navigator_logo.svg` — editable vector source

`st.set_page_config()` uses the custom PNG instead of the default Streamlit icon.

## Deployment

Use Python 3.12 and install the dependencies in `requirements.txt`.

For live job searches in cities that are not represented by the bundled local dataset, configure `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` in Streamlit Secrets.
