# Auto-converted from the original Jupyter notebook.
# Training-only script: DO NOT import this from app.py.

# ## XGB classifier for updated dataset with 26 variables
# This notebook trains an XGBoost classifier to predict whether a newcomer to Ontario will be employed, using synthetic data.

# ### Step 1: Import Libraries
# removed notebook install command


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

from xgboost import XGBClassifier
import shap


# ### Step 2: Load the Dataset

# Load the newcomer dataset from the CSV file
df = pd.read_csv("newcomer_ontario_enriched.csv")

print("Dataset loaded successfully.")
print(f"Shape: {df.shape[0]} rows and {df.shape[1]} columns")
df.head()


# ### Step 3: Exploratory Data Analysis (EDA)

# Check the data types of each column
print("Column Data Types:")
print(df.dtypes)


# Summary statistics for all numeric columns
print("Summary Statistics:")
df.describe()


# Value counts for selected categorical column so we can see the spread of categories
categorical_cols = [
    "sex",
    "admission_category",
    "world_region",
    "education_level",
    "field_of_study",
    "previous_occupation",
    "occupation_category",
    "teer_category",
    "credential_recognition_status",
]

for col in categorical_cols:
    print(f"\nValue counts for '{col}':")
    print(df[col].value_counts())


# Check the class balance for the target column 'employed'
# This tells us if the data is imbalanced (more employed than not, or vice versa)
print("Class balance for 'employed' column:")
print(df["employed"].value_counts())
print(f"\nPercentage breakdown:")
print(df["employed"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")


# Plot the distribution of age and annual_income side by side
# and save the figure to a file for reference

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Age distribution
axes[0].hist(df["age"], bins=20, color="steelblue", edgecolor="white")
axes[0].set_title("Distribution of Age", fontsize=14)
axes[0].set_xlabel("Age")
axes[0].set_ylabel("Count")

# Annual income distribution
axes[1].hist(df["annual_income"], bins=20, color="coral", edgecolor="white")
axes[1].set_title("Distribution of Annual Income", fontsize=14)
axes[1].set_xlabel("Annual Income (CAD)")
axes[1].set_ylabel("Count")

plt.suptitle("EDA: Key Feature Distributions", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()


# ### Step 4: Preprocessing

# Make a copy so we do not modify the original dataframe
df_processed = df.copy()

# Drop the province column because every row is Ontario, so it adds no useful information
df_processed.drop(columns=["province"], inplace=True)

# Scope down to the 13 features selected for the classifier
classifier_features = [
    "age",
    "sex",
    "admission_category",
    "world_region",
    "speaks_official_language",
    "education_level",
    "family_size",
    "field_of_study",
    "previous_occupation",
    "years_of_experience",
    "teer_category",
    "credential_recognition_status",
    "regulated_profession",
]
Target = "employed"

df_processed = df_processed[classifier_features + [Target]]

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


#Define X and y
X = df_processed[classifier_features]
y = df_processed[Target]

print("Feature columns used for training:")
print(X.columns.tolist())
print(f"\nTotal features: {X.shape[1]}")
print(f"Total samples: {X.shape[0]}")


# For X features we droped the target and all columns that are outcomes of being employed,not predictors of it. Keeping them would cause data leakage and give unrealistically perfect scores.
# Columns dropped and why:
#  annual_income: you only earn income after getting a job,
#  affordability_score: derived from income, so also post-employment,
#  months_to_employment: measures time after getting hired, not before,
#  rent_to_income_ratio: every unemployed person has exactly 1.75 (no income), making it a perfect but meaningless signal, housing_affordable: always 0 for unemployed, directly leaks the label

# Split the data into 80% training and 20% testing
# stratify=y ensures the class balance is preserved in both splits
# random_state=42 makes the split reproducible

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size:     {X_test.shape[0]} samples")


# Calculate scale_pos_weight to handle class imbalance
# This tells XGBoost to pay more attention to the minority class
# Formula: number of negative cases / number of positive cases

neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

print(f"Negative (not employed) count: {neg_count}")
print(f"Positive (employed) count:     {pos_count}")
print(f"scale_pos_weight set to:       {scale_pos_weight:.4f}")

# Train the XGBoost classifier
model = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss",
)

model.fit(X_train, y_train)


# In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# ### Step 7: Evaluate the Model

# Make predictions on the test set
from sklearn.metrics import precision_score, recall_score


y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probability scores for the positive class

# Print core evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:  {recall:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Employed", "Employed"]))


# Print the confusion matrix
# Rows are actual labels, columns are predicted labels
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"                Predicted Not Employed   Predicted Employed")
print(f"Actual Not Employed        {cm[0][0]:<22} {cm[0][1]}")
print(f"Actual Employed            {cm[1][0]:<22} {cm[1][1]}")


# Plot and save the ROC curve
# The ROC curve shows the tradeoff between true positive rate and false positive rate
# A higher AUC (area under the curve) means a better model

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"XGBoost (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Baseline")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve - Newcomer Employment Prediction", fontsize=14)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
plt.show()


# SHAP (SHapley Additive exPlanations) waterfall chart for the first test sample
# SHAP explains WHY the model made a specific prediction by showing the contribution
# of each feature to the final output for that one sample

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Waterfall plot for the very first test sample
plt.figure()
shap.plots.waterfall(shap_values[0], show=False)
plt.title("SHAP Waterfall Chart - First Test Sample", fontsize=13)
plt.tight_layout()
plt.savefig("shap_plot.png", dpi=150, bbox_inches="tight")
plt.show()


# ### Step 8: Save the Trained Model

# Create the models folder if it does not already exist
os.makedirs("models", exist_ok=True)

# Save the trained model to disk using joblib
# This lets us reload the model later without retraining it from scratch
model_path = "Secondversion_XGB_classifier.pkl"
joblib.dump(model, model_path)

