# Auto-converted from the original Jupyter notebook.
# Training-only script: DO NOT import this from app.py.

# # Income predictor for employed newcomers in Canada-Ontario
#
# This notebook trains an **XGBoost Regressor** to predict annual income for newcomers who are already employed.
#
# It mirrors the label-encoding approach used in `model_training.ipynb` (the employment classifier), so the two notebooks are fully consistent.

# ## Step 1 — Install & Import Libraries

# !pip install xgboost shap scikit-learn pandas matplotlib joblib

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ## Step 2 — Load the Dataset
# We load the full CSV, then immediately filter to **employed == 1** because we only predict income for people who are working.

DATA_PATH = 'newcomer_ontario_enriched.csv'

df = pd.read_csv(DATA_PATH)
print(f'Full dataset: {df.shape[0]} rows × {df.shape[1]} columns')

df.head()


# ## Step 3 — Filter to Employed Newcomers Only

# Keep only rows where the person is employed
df_employed = df[df['employed'] == 1].copy()

print(f'Rows after filtering to employed == 1: {len(df_employed)}')
print(f'Rows removed (not employed)          : {len(df) - len(df_employed)}')


df.columns


# ## Step 4 — Select Features and Target
# We use **14 features** that are known before a person starts working — no post-employment leakage.
#
# | Feature | Type |
# |---|---|
# | age | numerical |
# | sex | categorical |
# | admission_category | categorical |
# | world_region | categorical |
# | speaks_official_language | binary |
# | education_level | categorical |
# | family_size | numerical |
# | field_of_study | categorical |
# |previous_occupation | categorical |
# |employment_category | categorical |
# | years_of_experience | numerical |
# | teer_category | categorical|
# | credential_recognition_status | categorical |
# |regulated_profession | binary |

# ## Step 5 — Label Encode Categorical Columns
# XGBoost needs numbers, not text. `LabelEncoder` converts each unique string to an integer (e.g. `'Female'` → `0`, `'Male'` → `1`).
#
# > This uses the **same approach** as `Secondversion_XGB_classifier.ipynb` so the two notebooks stay consistent.

# Make a copy so we do not modify the original dataframe
df_processed = df.copy()

# Drop the province column because every row is Ontario, so it adds no useful information
df_processed.drop(columns=["province"], inplace=True)

# Scope down to the 14 features selected for the regressor.
regressor_features = [
    "age",
    "sex",
    "admission_category",
    "world_region",
    "speaks_official_language",
    "education_level",
    "family_size",
    "field_of_study",
    "previous_occupation",
    "employment_category",
    "years_of_experience",
    "teer_category",
    "credential_recognition_status",
    "regulated_profession",
]
Target = "annual_income"

df_processed = df_processed[regressor_features + [Target]]

# collapse rare categories into "Other" for the two high-cardinality features
high_cardinality_features = ["previous_occupation", "field_of_study"]
rare_threshold = 0.01  # this collapse categories below 1% frequency

rare_category_map = {}
for col in high_cardinality_features:
    freq = df_processed[col].value_counts(normalize=True)
    rare_categories = freq[freq < rare_threshold].index.tolist()
    rare_category_map[col] = rare_categories
    df_processed[col] = df_processed[col].where(~df_processed[col].isin(rare_categories), "Other")
    print(f"{col}: collapsed {len(rare_categories)} rare categories into 'Other' "
          f"-> {df_processed[col].nunique()} categories remaining")

#Label encode all 9 categorical columns
categorical_cols = [
    "sex",
    "admission_category",
    "world_region",
    "education_level",
    "field_of_study",
    "previous_occupation",
    "employment_category",
    "teer_category",
    "credential_recognition_status",
]

label_encoders = {}
for col in categorical_cols:
    encoder = LabelEncoder()
    df_processed[col] = encoder.fit_transform(df_processed[col])
    label_encoders[col] = encoder
    print(f"Encoded '{col}' column.")

print(f"Processed dataframe shape: {df_processed.shape}")

df_processed.head()


# ## Step 6 — Train / Test Split

#Define X and y
X = df_processed[regressor_features]
y = df_processed[Target]


# Split the data into 80% training and 20% testing
# stratify=y ensures the class balance is preserved in both splits
# random_state=42 makes the split reproducible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size:     {X_test.shape[0]} samples")


# Linear Regression Baseline model

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
linear_r2 = r2_score(y_test, linear_predictions)

print("Linear Regression Results")
print("MAE:", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R²:", round(linear_r2, 3))


# ## Step 7 — Train XGBoost Regressor
# `XGBRegressor` builds an ensemble of decision trees that learn to predict a continuous number (income). Using `random_state=42` keeps results reproducible.

model = XGBRegressor(random_state=42, verbosity=0)
model.fit(X_train, y_train)


# ## Step 8 — Evaluate the Model
# Three standard regression metrics:
#
# | Metric | What it tells you |
# |---|---|
# | **MAE** | Average dollar error — easy to interpret |
# | **RMSE** | Like MAE but penalises large errors more |
# | **R²** | How much variance the model explains (1.0 = perfect) |

y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print('Model Evaluation on Test Set')
print('─' * 40)
print(f'MAE  (Mean Absolute Error) : ${mae:,.0f}')
print(f'MSE (Mean Squared Error) : ${mse:,.0f}')
print(f'RMSE (Root Mean Sq. Error) : ${rmse:,.0f}')
print(f'R²   Score                 : {r2:.3f}')


#compares the predicted annual incomes with the actual values in the test dataset
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred, alpha=0.4, s=15, color="#5bc0de")

# perfect prediction reference line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", label="Perfect prediction")

ax.set_xlabel("Actual annual income")
ax.set_ylabel("Predicted annual income")
ax.set_title("Actual vs. Predicted Income")
ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted_income.png", dpi=150)
plt.show()


residuals = y_test - y_pred

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred, residuals, alpha=0.4, s=15, color="#f0ad4e")
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Predicted annual income")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot_income.png", dpi=150)
plt.show()


# Compare Linear Regression and XGBoost Regressor

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "XGBoost Regressor"],
    "MAE": [linear_mae, mae],
    "RMSE": [linear_rmse, rmse],
    "R2": [linear_r2, r2]
})

comparison


# ## Step 9 — SHAP Waterfall Chart
# SHAP (**SH**apley **A**dditive ex**P**lanations) explains *why* the model predicted a specific value for one person.
#
# The waterfall chart shows each feature's **push** (red, +) or **pull** (blue, −) on the predicted income, starting from the model's average output (`E[f(X)]`).

# Build a SHAP explainer from the trained model
explainer = shap.Explainer(model)

# Compute SHAP values for the whole test set
shap_values = explainer(X_test)

# Waterfall chart for the first test sample
first_idx = 0

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values[first_idx], show=False)

plt.title(
    f'SHAP Waterfall — Sample #1\n'
    f'Predicted: ${y_pred[first_idx]:,.0f}  |  Actual: ${y_test.iloc[first_idx]:,.0f}',
    fontsize=11, pad=12
)
plt.tight_layout()

SHAP_PATH = 'income_shap.png'
plt.savefig(SHAP_PATH, dpi=150, bbox_inches='tight')
plt.show()


# ## Step 10 — Save the Trained Model

os.makedirs('models', exist_ok=True)
MODEL_PATH = 'Secondversion_XGB_regressor.pkl'

joblib.dump(model, MODEL_PATH)

