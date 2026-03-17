"""
Customer Churn Prediction - Training Script
Dataset: Telco Customer Churn (Kaggle)
Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
Save as 'WA_Fn-UseC_-Telco-Customer-Churn.csv' in the same folder as this script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
import xgboost as xgb
import shap
import pickle
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading data")
print("=" * 50)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"Shape: {df.shape}")
print(df.head(3))


# ─────────────────────────────────────────
# 2. EDA (Exploratory Data Analysis)
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 2: EDA")
print("=" * 50)

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nChurn distribution:")
print(df["Churn"].value_counts())
print(f"Churn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")

# Plot 1: Churn distribution
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df["Churn"].value_counts().plot(kind="bar", ax=axes[0], color=["#1D9E75", "#D85A30"])
axes[0].set_title("Churn Distribution")
axes[0].set_xlabel("Churn")
axes[0].set_ylabel("Count")
axes[0].tick_params(rotation=0)

# Plot 2: Churn by contract type
churn_contract = df.groupby(["Contract", "Churn"]).size().unstack()
churn_contract.plot(kind="bar", ax=axes[1], color=["#1D9E75", "#D85A30"])
axes[1].set_title("Churn by Contract Type")
axes[1].tick_params(rotation=15)

# Plot 3: Tenure distribution by churn
df[df["Churn"] == "Yes"]["tenure"].hist(ax=axes[2], alpha=0.7, label="Churned", color="#D85A30", bins=20)
df[df["Churn"] == "No"]["tenure"].hist(ax=axes[2], alpha=0.7, label="Stayed", color="#1D9E75", bins=20)
axes[2].set_title("Tenure by Churn")
axes[2].set_xlabel("Months")
axes[2].legend()

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=120)
print("Saved: eda_plots.png")
plt.close()


# ─────────────────────────────────────────
# 3. DATA CLEANING & FEATURE ENGINEERING
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 3: Cleaning + Feature Engineering")
print("=" * 50)

# Fix TotalCharges (has spaces, should be numeric)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Drop customerID (not useful)
df.drop("customerID", axis=1, inplace=True)

# Encode target
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# Feature engineering
df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
df["HasMultipleServices"] = (
    (df["PhoneService"] == "Yes").astype(int) +
    (df["InternetService"] != "No").astype(int) +
    (df["OnlineSecurity"] == "Yes").astype(int) +
    (df["StreamingTV"] == "Yes").astype(int)
)

# Encode all categorical columns
le = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

print(f"Final feature count: {df.shape[1] - 1}")
print("Sample features:", df.columns.tolist()[:8])


# ─────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")


# ─────────────────────────────────────────
# 5. MODEL COMPARISON
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 4: Comparing 3 Models")
print("=" * 50)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=5, use_label_encoder=False,
        eval_metric="logloss", random_state=42
    )
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {"AUC": auc, "F1": report["1"]["f1-score"]}
    print(f"\n{name}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  F1 (churn class): {report['1']['f1-score']:.4f}")

# XGBoost wins — use it as final model
best_model = models["XGBoost"]


# ─────────────────────────────────────────
# 6. CONFUSION MATRIX
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, best_model.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stayed", "Churned"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
print("\nSaved: confusion_matrix.png")
plt.close()


# ─────────────────────────────────────────
# 7. SHAP EXPLAINABILITY
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 5: SHAP Feature Importance")
print("=" * 50)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Top Features Driving Churn (SHAP)")
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=120)
print("Saved: shap_importance.png")
plt.close()


# ─────────────────────────────────────────
# 8. SAVE MODEL + SCALER + FEATURE NAMES
# ─────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6: Saving model artifacts")
print("=" * 50)

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Saved: model.pkl, scaler.pkl, feature_names.pkl")
print("\nAll done! Now run: streamlit run app.py")