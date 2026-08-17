import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Insurance Charge Predictor", page_icon="💰", layout="centered")

# --- Load model artifacts ---
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

st.title("💰 Insurance Charge Predictor")
st.write(
    "Estimate a person's annual medical insurance charges based on their "
    "profile, using a linear regression model trained on the classic "
    "insurance cost dataset."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)

with col2:
    children = st.number_input("Number of children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

if st.button("Predict Charges", type="primary", use_container_width=True):
    # --- Replicate the notebook's preprocessing ---
    is_female = 1 if sex == "Female" else 0
    is_smoker = 1 if smoker == "Yes" else 0
    region_southeast = 1 if region == "Southeast" else 0

    if bmi <= 18.5:
        bmi_category_obese = 0
    elif bmi <= 24.9:
        bmi_category_obese = 0
    elif bmi <= 29.9:
        bmi_category_obese = 0
    else:
        bmi_category_obese = 1

    row = pd.DataFrame([{
        "age": age,
        "is_female": is_female,
        "bmi": bmi,
        "children": children,
        "is_smoker": is_smoker,
        "region_southeast": region_southeast,
        "bmi_category_Obese": bmi_category_obese,
    }])[feature_cols]

    scale_cols = ["age", "bmi", "children"]
    row[scale_cols] = scaler.transform(row[scale_cols])

    prediction = model.predict(row)[0]
    prediction = max(prediction, 0)

    st.success(f"### Estimated Annual Charges: **${prediction:,.2f}**")

    with st.expander("See how this was calculated"):
        st.write("Features fed into the model (after scaling age/bmi/children):")
        st.dataframe(row)

st.divider()
st.caption(
    "Model: Linear Regression · Trained on the Kaggle Medical Cost Personal "
    "Dataset · For educational purposes only, not actual insurance advice."
)
