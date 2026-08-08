# Version fixes

## 1. No more Toronto fallback for another city
The local job source previously searched by keyword across the whole file when the requested city had no local postings. Since the bundled job file is mostly Toronto, this silently returned Toronto jobs for other cities.

Version 9 returns no local jobs when the requested city is not represented. If Adzuna credentials are configured, the app automatically falls back to a live city-specific Adzuna search.

## 2. Housing is restricted to the selected city when city data exists
The Karthika runtime module now passes `preferred_city` into housing preparation and prefers housing records in that city.

## 3. Water markers removed
The Toronto housing file contains several synthetic CMHC coordinates south of the actual Toronto mainland. Version 9 adds a conservative Toronto shoreline guard and reapplies the housing land filter immediately before plotting the map.

## 4. Karthika notebook remains the source logic
`notebooks/Recommendation_Engine_Karthika.ipynb` remains as the reference notebook. `pipeline/karthika_recommendation.py` is the runtime `.py` conversion used by `app.py`.

## 5. Employment probability
- Replaced the previous class-weighted raw XGBoost probability with a tuned, unweighted XGBoost classifier.
- Added 5-fold sigmoid probability calibration (`CalibratedClassifierCV`).
- Added Brier score and observed-vs-predicted employment rate to model metadata.
- The supplied dataset has an employment rate of about 78.1%, so a calibrated model should be centered around that rate rather than returning an overconfident 90%+ score for nearly every profile.
- Runtime API is unchanged: `XGBModels.run(profile)` still returns `employment_probability`.

Validation on the supplied test split:
- ROC-AUC: ~0.710
- Brier score: ~0.155
- Mean predicted probability: ~77.8%
- Observed employment rate: ~78.1%
- Test probability range: approximately 40%–91% rather than essentially always above 90%.

## 6. Branding
- Added `assets/career_navigator_logo.png` for the Streamlit page icon and sidebar branding.
- Added `assets/career_navigator_logo.svg` as an editable/vector logo source.
- `app.py` now uses the PNG as the browser/app page icon when available.

## 7. Reliability cleanup
- Fixed the optional `filter_housing_near_jobs()` helper so it no longer references an undefined `profile` variable.
- Kept the Version 9 city-specific job fallback and housing shoreline filtering.
