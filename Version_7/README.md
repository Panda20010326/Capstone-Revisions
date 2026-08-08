# Capstone integration — notebook → `.py` runtime

## What changed

The Streamlit app is now an inference-only application. It **does not run notebook
training code when a user clicks the prediction button**.

### Runtime modules imported by `app.py`

- `pipeline.profile_encoder` — profile/occupation stage
- `pipeline.xgb_models` — employment + income predictions
- `pipeline.adzuna_client` — optional live job retrieval
- `pipeline.job_source` — offline job retrieval
- `pipeline.recommendation_engine` — job ranking

The converted/training scripts under `training/` are **not imported by `app.py`**.

## Important prediction correction

The source project contains a mismatch between the intended classifier features and
the saved model artifact: the saved XGBoost models expect `employment_category`,
which is an outcome-derived field and is not collected by the app. The corrected
runtime models therefore use `occupation_category`, which is available from the
profile form.

This avoids feeding the model information that is only known after employment and
makes the training/inference feature contract consistent.

## How to retrain

```bash
python training/train_corrected_models.py
```

This creates:

- `artifacts/xgboost/employment_classifier_corrected.pkl`
- `artifacts/xgboost/income_regressor_corrected.pkl`
- `artifacts/xgboost/label_encoders_corrected.pkl`
- `artifacts/xgboost/rare_category_map.pkl`

The current saved corrected models were trained from the supplied
`newcomer_ontario_enriched.csv`.

## Run the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

For live Adzuna mode, set:

```bash
export ADZUNA_APP_ID="..."
export ADZUNA_APP_KEY="..."
```

The local job dataset works without API credentials.

## ProfileEncoder note

The supplied project contains the 16-dimensional ProfileEncoder and its preprocessor,
but not the full multitask model/category encoder required to decode an occupation
from the embedding. The runtime therefore does not pretend the embedding alone can
produce a category prediction. The single integration point is
`pipeline/profile_encoder.py`; if the complete multitask artifacts are added later,
only that module needs to be upgraded.

## Karthika Recommendation Engine integration

`notebooks/Recommendation_Engine_Karthika.ipynb` has been converted into the runtime module:

`pipeline/karthika_recommendation.py`

The Streamlit app calls this module after the employment/income predictions are generated. The module ranks jobs using the notebook's scoring logic and produces nearby housing recommendations. The map plots only those recommended jobs and nearby housing rather than every housing record.

The runtime module also removes invalid/offshore housing coordinates from the CMHC-derived dataset and applies a 30 km commute filter before housing is displayed on the map.
