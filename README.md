# 🦟 DengueGuard BD
### ML-based Dengue Risk Prediction & Analysis for Bangladesh

DengueGuard BD is a machine learning project that predicts a person's dengue risk from
demographics, location and symptoms, using a real dengue dataset collected in Dhaka,
Bangladesh. It includes a full ML workflow (EDA → preprocessing → model comparison →
explainability) and an interactive Streamlit web app.

> **Disclaimer:** This is an educational / portfolio project, not a certified medical
> device. It must not be used as a substitute for professional medical diagnosis. If you
> suspect dengue, consult a doctor and get an NS1/IgG/IgM test.


## 📌 Overview

**Problem:** Dengue is a recurring public health crisis in Bangladesh. Getting an early
  risk estimate — before a lab test — can help people seek care sooner.
**Approach:** Two modeling scenarios were built and compared:
**Screening Model** — uses demographics, location, vitals and symptoms only
    (no lab test). This is the model shipped in the web app, since lab results usually
    aren't available at the point someone wants a risk estimate.
**Confirmatory Model** — adds NS1 / IgG / IgM lab test results, closer to an
    actual clinical diagnosis.
**Models compared:** Logistic Regression, Random Forest, XGBoost
**Explainability:** Feature importance + SHAP summary plots

## 🗂️ Dataset

**A Comprehensive Dengue Dataset of Bangladesh**, Mendeley Data (2025), Md Kawsar Ahmad
🔗 https://data.mendeley.com/datasets/zdtc3n6xv2

1,000 records collected from patients and community members in Dhaka, including
demographics, NS1/IgG/IgM test results, area, house type, vitals (body temperature,
platelet count, WBC count) and symptoms (headache, joint pain, myalgia, rash, etc.).

## 📁 Project Structure

```
dengueguard-bd/
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md
├── data/
│   └── dataset.csv            # Dataset used for training
├── model/
│   ├── train.py                    # Training script (regenerates everything below)
│   ├── dengueguard_model.joblib    # Saved best model (screening)
│   ├── feature_list.joblib         # Feature list used by the model
│   ├── best_model_name.joblib      # Name of the best model
│   └── results.json                # All models' metrics
├── notebook/
│   └── DengueGuard_BD.ipynb   # Full analysis notebook (Colab-ready)
└── assets/                    # Saved plots (EDA, confusion matrices, ROC, etc.)
```






## 🧠 Model Performance

Screening models (no lab test), on a held-out 20% test set:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| Random Forest | 1.00 | 1.00 | 1.00 | 1.00 |
| XGBoost | 1.00 | 1.00 | 1.00 | 1.00 |

*(Generated with `python model/train.py`. Re-run the notebook in Colab, where XGBoost
is preinstalled, to confirm — the numbers above were produced with scikit-learn's
HistGradientBoosting as a local XGBoost stand-in.)*

**Note on accuracy:** Vitals such as body temperature and platelet count separate the
two classes almost perfectly in this public dataset, so scores are very high for both
the screening and confirmatory models. This reflects the dataset's clean, synthetic-style
construction rather than a claim that dengue is trivially predictable from symptoms in
the real world — a limitation worth mentioning when discussing this project.

## 🖥️ App Features

- **Home** — project overview and dataset summary
- **Risk Prediction** — form-based dengue risk estimate with a probability score
- **Data Insights** — EDA charts (outcome distribution, age, area-wise positive rate)
- **Model Performance** — metrics table, ROC curve, feature importance for all models
- **About** — project, dataset and author info

## 🛠️ Tech Stack

Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Streamlit · Matplotlib · Seaborn

## 👤 Author

Krisanu Das
www.linkedin.com/in/krisanu-das

## 📄 License

This project is released under the MIT License. The dataset is credited to its original
authors (see Dataset section above); check the dataset's own license terms on Mendeley
before any commercial use.
