# Version 9 fixes

## 1. No more Toronto fallback for another city
The local job source previously searched by keyword across the whole file when the requested city had no local postings. Since the bundled job file is mostly Toronto, this silently returned Toronto jobs for other cities.

Version 9 returns no local jobs when the requested city is not represented. If Adzuna credentials are configured, the app automatically falls back to a live city-specific Adzuna search.

## 2. Housing is restricted to the selected city when city data exists
The Karthika runtime module now passes `preferred_city` into housing preparation and prefers housing records in that city.

## 3. Water markers removed
The Toronto housing file contains several synthetic CMHC coordinates south of the actual Toronto mainland. Version 9 adds a conservative Toronto shoreline guard and reapplies the housing land filter immediately before plotting the map.

## 4. Karthika notebook remains the source logic
`notebooks/Recommendation_Engine_Karthika.ipynb` remains as the reference notebook. `pipeline/karthika_recommendation.py` is the runtime `.py` conversion used by `app.py`.
