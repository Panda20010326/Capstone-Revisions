# Version 10 changes

## Employment probability
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

## Branding
- Added `assets/career_navigator_logo.png` for the Streamlit page icon and sidebar branding.
- Added `assets/career_navigator_logo.svg` as an editable/vector logo source.
- `app.py` now uses the PNG as the browser/app page icon when available.

## Reliability cleanup
- Fixed the optional `filter_housing_near_jobs()` helper so it no longer references an undefined `profile` variable.
- Kept the Version 9 city-specific job fallback and housing shoreline filtering.
