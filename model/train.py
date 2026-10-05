"""
DengueGuard BD - Model training script
Trains two sets of classifiers on the Bangladesh Dengue dataset:
  1. Screening models  - symptoms/demographics only (no lab test), realistic early-warning use case
  2. Confirmatory models - includes NS1/IgG/IgM lab test results (near-deterministic, used for validation)
The Screening model is the one shipped in the Streamlit app, since lab results
usually aren't available at the point a person wants a risk estimate.
"""
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_XGB = False

RANDOM_STATE = 42
sns.set_style("whitegrid")

df = pd.read_csv("data/dataset.csv")
df = df.drop(columns=["District"])  # single value across dataset, no signal
df["Joint_Pain"] = df["Joint_Pain"].fillna("None")

TARGET = "Outcome"
NUMERIC_FEATURES = ["Age", "Fever_Duration", "Body_Temperature", "Platelet_Count", "WBC_Count"]
SYMPTOM_FEATURES = ["Headache", "Retro_Orbital_Pain", "Myalgia", "Rash"]
CATEGORICAL_FEATURES = ["Gender", "Area", "AreaType", "HouseType", "Joint_Pain"]
LAB_FEATURES = ["NS1", "IgG", "IgM"]

SCREENING_FEATURES = NUMERIC_FEATURES + SYMPTOM_FEATURES + CATEGORICAL_FEATURES
FULL_FEATURES = SCREENING_FEATURES + LAB_FEATURES


def build_models():
    m = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        m["XGBoost"] = XGBClassifier(n_estimators=300, eval_metric="logloss", random_state=RANDOM_STATE)
    else:
        m["XGBoost (HistGB fallback)"] = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    return m


def run_experiment(features, tag):
    X = df[features]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        remainder="passthrough"
    )

    results = {}
    fitted = {}
    roc_data = {}

    for name, clf in build_models().items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall": round(recall_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(auc(*roc_curve(y_test, y_proba)[:2]), 4),
        }
        fitted[name] = pipe
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, results[name]["roc_auc"])

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Dengue", "Dengue"], yticklabels=["No Dengue", "Dengue"])
        plt.title(f"Confusion Matrix - {name} ({tag})")
        plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.tight_layout()
        plt.savefig(f"assets/cm_{tag}_{name.split()[0].lower()}.png", dpi=120)
        plt.close()

    best_name = max(results, key=lambda k: results[k]["f1"])

    plt.figure(figsize=(5, 4))
    for name, (fpr, tpr, roc_auc_v) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_v:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {tag.capitalize()} Models")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"assets/roc_curve_{tag}.png", dpi=120)
    plt.close()

    try:
        clf = fitted[best_name].named_steps["clf"]
        prep = fitted[best_name].named_steps["prep"]
        feat_names = prep.get_feature_names_out()
        importances = clf.feature_importances_
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
            "importance", ascending=False).head(12)
        plt.figure(figsize=(6, 5))
        sns.barplot(data=imp_df, x="importance", y="feature", color="#d9534f")
        plt.title(f"Top Feature Importances - {best_name} ({tag})")
        plt.tight_layout()
        plt.savefig(f"assets/feature_importance_{tag}.png", dpi=120)
        plt.close()
    except Exception as e:
        print(f"[{tag}] feature importance skipped:", e)

    return results, fitted, best_name


screening_results, screening_fitted, screening_best = run_experiment(SCREENING_FEATURES, "screening")
full_results, full_fitted, full_best = run_experiment(FULL_FEATURES, "full")

print("=== SCREENING (no lab test) ===")
print(json.dumps(screening_results, indent=2))
print("Best:", screening_best)
print("=== FULL (with lab test) ===")
print(json.dumps(full_results, indent=2))
print("Best:", full_best)

with open("model/results.json", "w") as f:
    json.dump({
        "screening": {"results": screening_results, "best_model": screening_best, "features": SCREENING_FEATURES},
        "full": {"results": full_results, "best_model": full_best, "features": FULL_FEATURES},
    }, f, indent=2)

# Ship the screening model in the app (realistic use case: no lab test yet available)
joblib.dump(screening_fitted[screening_best], "model/dengueguard_model.joblib")
joblib.dump(SCREENING_FEATURES, "model/feature_list.joblib")
joblib.dump(screening_best, "model/best_model_name.joblib")

# ---------- EDA plots ----------
plt.figure(figsize=(4, 3.5))
sns.countplot(data=df, x="Outcome", hue="Outcome", palette=["#5cb85c", "#d9534f"], legend=False)
plt.xticks([0, 1], ["No Dengue", "Dengue"]); plt.title("Outcome Distribution")
plt.tight_layout(); plt.savefig("assets/eda_outcome.png", dpi=120); plt.close()

plt.figure(figsize=(5, 3.5))
sns.histplot(data=df, x="Age", hue="Outcome", multiple="stack", bins=20, palette=["#5cb85c", "#d9534f"])
plt.title("Age Distribution by Outcome"); plt.tight_layout()
plt.savefig("assets/eda_age.png", dpi=120); plt.close()

plt.figure(figsize=(6, 4))
area_outcome = df.groupby("Area")["Outcome"].mean().sort_values(ascending=False)
sns.barplot(x=area_outcome.values, y=area_outcome.index, color="#d9534f")
plt.title("Dengue Positive Rate by Area (Dhaka)"); plt.xlabel("Positive rate")
plt.tight_layout(); plt.savefig("assets/eda_area.png", dpi=120); plt.close()

plt.figure(figsize=(4, 3.5))
sns.countplot(data=df, x="AreaType", hue="Outcome", palette=["#5cb85c", "#d9534f"])
plt.title("Outcome by Area Type"); plt.tight_layout()
plt.savefig("assets/eda_areatype.png", dpi=120); plt.close()

print("HAS_XGB:", HAS_XGB)
print("Done.")
