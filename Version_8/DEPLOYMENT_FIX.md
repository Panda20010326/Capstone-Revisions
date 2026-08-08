# Deployment dependency fix

The previous requirements file installed TensorFlow even though the runtime Streamlit app does not import TensorFlow. This made dependency installation unnecessarily large and could cause Community Cloud installation failures.

The runtime app uses:
- Streamlit
- pandas / numpy
- scikit-learn
- XGBoost
- joblib
- Plotly
- Folium / streamlit-folium
- requests

TensorFlow is therefore excluded from `requirements.txt`.

For Streamlit Community Cloud, use Python 3.12 in Advanced settings. Community Cloud currently defaults to Python 3.12.

Do not add TensorFlow back unless `pipeline/profile_encoder.py` is changed to perform live TensorFlow inference.
