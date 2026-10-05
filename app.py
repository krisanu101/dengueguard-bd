"""
DengueGuard BD
ML-based Dengue Risk Prediction & Analysis for Bangladesh
Streamlit web application
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

st.set_page_config(page_title="DengueGuard BD", page_icon="🦟", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- Load model & data (cached) ----------------
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "model", "dengueguard_model.joblib"))
    features = joblib.load(os.path.join(BASE_DIR, "model", "feature_list.joblib"))
    best_name = joblib.load(os.path.join(BASE_DIR, "model", "best_model_name.joblib"))
    return model, features, best_name

@st.cache_data
def load_results():
    with open(os.path.join(BASE_DIR, "model", "results.json")) as f:
        return json.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "dataset.csv"))
    df["Joint_Pain"] = df["Joint_Pain"].fillna("None")
    return df

model, FEATURES, BEST_MODEL_NAME = load_model()
results = load_results()
df = load_data()

AREAS = sorted(df["Area"].unique().tolist())

# ---------------- Sidebar navigation ----------------
st.sidebar.title("🦟 DengueGuard BD")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Risk Prediction", "Data Insights", "Model Performance", "About"]
)
st.sidebar.markdown("---")
st.sidebar.caption("ML-based Dengue Risk Prediction & Analysis for Bangladesh")

# ==================================================================
# HOME
# ==================================================================
if page == "Home":
    st.title("🦟 DengueGuard BD")
    st.subheader("ML-based Dengue Risk Prediction & Analysis for Bangladesh")

    st.markdown(
        """
        Dengue is a recurring public health challenge in Bangladesh, especially in Dhaka.
        **DengueGuard BD** uses machine learning trained on a real Bangladeshi dengue dataset
        to estimate a person's dengue risk from demographics, location and symptoms —
        **before** a lab test is even done.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", len(df))
    col2.metric("Dengue Positive", int(df["Outcome"].sum()))
    col3.metric("Areas Covered", df["Area"].nunique())
    col4.metric("Best Model", BEST_MODEL_NAME)

    st.markdown("### How to use this app")
    st.markdown(
        """
        1. Go to **Risk Prediction** and fill in basic information and symptoms.
        2. Get an instant dengue risk estimate with a probability score.
        3. Explore **Data Insights** to see dengue patterns across Dhaka.
        4. Check **Model Performance** to see how the underlying models were evaluated.
        """
    )

    st.warning(
        "⚠️ This tool is for educational / portfolio purposes only. "
        "It is **not** a medical diagnostic device. If you suspect dengue, "
        "consult a doctor and get an NS1/IgG/IgM test immediately."
    )

# ==================================================================
# RISK PREDICTION
# ==================================================================
elif page == "Risk Prediction":
    st.title("🩺 Dengue Risk Prediction")
    st.caption("Screening model — uses demographics & symptoms only (no lab test required)")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age", 1, 90, 30)
            gender = st.selectbox("Gender", sorted(df["Gender"].unique()))
            area = st.selectbox("Area (Dhaka)", AREAS)
            area_type = st.selectbox("Area Type", sorted(df["AreaType"].unique()))
            house_type = st.selectbox("House Type", sorted(df["HouseType"].unique()))
        with c2:
            fever_duration = st.slider("Fever Duration (days)", 0, 14, 3)
            body_temp = st.slider("Body Temperature (°C)", 35.0, 41.0, 37.0, 0.1)
            platelet = st.number_input("Platelet Count", min_value=10000, max_value=500000, value=200000, step=1000)
            wbc = st.number_input("WBC Count", min_value=1000, max_value=15000, value=6000, step=100)
            joint_pain = st.selectbox("Joint Pain", ["None", "Moderate", "Severe"])

        st.markdown("**Other symptoms**")
        s1, s2, s3, s4 = st.columns(4)
        headache = s1.checkbox("Headache")
        retro_orbital = s2.checkbox("Retro-orbital Pain")
        myalgia = s3.checkbox("Myalgia (muscle pain)")
        rash = s4.checkbox("Rash")

        submitted = st.form_submit_button("Predict Risk", use_container_width=True)

    if submitted:
        input_dict = {
            "Age": age,
            "Fever_Duration": fever_duration,
            "Body_Temperature": body_temp,
            "Platelet_Count": platelet,
            "WBC_Count": wbc,
            "Headache": int(headache),
            "Retro_Orbital_Pain": int(retro_orbital),
            "Myalgia": int(myalgia),
            "Rash": int(rash),
            "Gender": gender,
            "Area": area,
            "AreaType": area_type,
            "HouseType": house_type,
            "Joint_Pain": joint_pain,
        }
        input_df = pd.DataFrame([input_dict])[FEATURES]
        proba = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.markdown("### Result")
        if proba >= 0.66:
            st.error(f"🔴 High Risk — {proba*100:.1f}% estimated probability of dengue")
        elif proba >= 0.33:
            st.warning(f"🟠 Medium Risk — {proba*100:.1f}% estimated probability of dengue")
        else:
            st.success(f"🟢 Low Risk — {proba*100:.1f}% estimated probability of dengue")

        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(
            "This is a statistical estimate based on patterns in historical data, "
            "not a medical diagnosis. Please consult a doctor for a confirmatory NS1/IgG/IgM test."
        )

# ==================================================================
# DATA INSIGHTS
# ==================================================================
elif page == "Data Insights":
    st.title("📊 Data Insights")
    st.caption("Exploratory analysis of the Bangladesh dengue dataset")

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(BASE_DIR, "assets", "eda_outcome.png"), caption="Outcome Distribution", use_container_width=True)
        st.image(os.path.join(BASE_DIR, "assets", "eda_areatype.png"), caption="Outcome by Area Type", use_container_width=True)
    with col2:
        st.image(os.path.join(BASE_DIR, "assets", "eda_age.png"), caption="Age Distribution by Outcome", use_container_width=True)

    st.image(os.path.join(BASE_DIR, "assets", "eda_area.png"), caption="Dengue Positive Rate by Area", use_container_width=True)

    st.markdown("### Filter the raw data")
    sel_area = st.multiselect("Filter by Area", AREAS, default=[])
    filtered = df if not sel_area else df[df["Area"].isin(sel_area)]
    st.dataframe(filtered, use_container_width=True, height=300)

# ==================================================================
# MODEL PERFORMANCE
# ==================================================================
elif page == "Model Performance":
    st.title("📈 Model Performance")

    st.markdown("### Screening models (demographics + symptoms, no lab test)")
    st.dataframe(pd.DataFrame(results["screening"]["results"]).T, use_container_width=True)
    st.caption(f"Best model: **{results['screening']['best_model']}** — shipped in this app.")

    c1, c2 = st.columns(2)
    with c1:
        st.image(os.path.join(BASE_DIR, "assets", "roc_curve_screening.png"), use_container_width=True)
    with c2:
        fi_path = os.path.join(BASE_DIR, "assets", "feature_importance_screening.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)

    st.markdown("### Confirmatory models (with NS1 / IgG / IgM lab test)")
    st.dataframe(pd.DataFrame(results["full"]["results"]).T, use_container_width=True)
    st.info(
        "Note: Lab-test and clinical-vitals features (Body Temperature, Platelet Count) separate "
        "the two classes almost perfectly in this public dataset, so both the screening and "
        "confirmatory models reach very high scores. This reflects the dataset's clean, synthetic "
        "construction rather than a claim that dengue is trivially predictable in the real world."
    )

# ==================================================================
# ABOUT
# ==================================================================
elif page == "About":
    st.title("ℹ️ About DengueGuard BD")
    st.markdown(
        """
        **DengueGuard BD** is a machine learning portfolio project that predicts dengue risk
        using a real Bangladeshi dengue dataset (demographics, location, symptoms, and clinical
        test results collected from Dhaka).

        **Tech stack:** Python, Pandas, Scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn

        **Dataset:** *A Comprehensive Dengue Dataset of Bangladesh*, Mendeley Data (2025),
        Md Kawsar Ahmad. [data.mendeley.com/datasets/zdtc3n6xv2](https://data.mendeley.com/datasets/zdtc3n6xv2)

        ---
        **Author:** Your Name Here
        **GitHub:** github.com/your-username/dengueguard-bd
        **LinkedIn:** linkedin.com/in/your-profile

        ---
        ⚠️ *Disclaimer: This project is for educational and portfolio purposes only.
        It is not a certified medical device and must not be used as a substitute
        for professional medical diagnosis.*
        """
    )
