"""Streamlit demo for the medical cost prediction model."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_candidates,
    build_preprocessor,
    load_data,
)

MODEL_PATH = Path("models/model.joblib")
DATA_PATH = Path("data/insurance.csv")


@st.cache_resource
def get_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    # First run on a fresh Space: train inline so the demo works without
    # committing a binary model file to git.
    df = load_data(DATA_PATH)
    x = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["charges"]
    model = build_candidates(build_preprocessor())["random_forest"]
    model.fit(x, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


st.set_page_config(page_title="Medical Cost Prediction", page_icon="🏥")
st.title("🏥 Medical Cost Prediction")
st.caption(
    "Predicts individual medical insurance charges from demographic and "
    "health attributes. Random Forest regressor, test R² = 0.868."
)

model = get_model()

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=29)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.3, step=0.1)
with col2:
    children = st.number_input("Number of children", min_value=0, max_value=10, value=1)
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

if st.button("Predict charges", type="primary"):
    row = pd.DataFrame(
        [{"age": age, "bmi": bmi, "children": children, "sex": sex, "smoker": smoker, "region": region}]
    )
    prediction = model.predict(row)[0]
    st.metric("Predicted medical charges", f"${prediction:,.2f}")

st.divider()
st.caption("Source: [GitHub repo](https://github.com/mukulkatewa/medical-cost-prediction)")
