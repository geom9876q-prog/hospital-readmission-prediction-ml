import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Hospital Readmission Predictor")

# ---------------- LOAD MODEL ----------------
model = CatBoostClassifier()
model.load_model("readmission_catboost.cbm")

# ---------------- UI ----------------
st.title("🏥 Hospital Readmission Risk Predictor")
st.write("Predicts 30-day hospital readmission risk for diabetic patients")
st.caption("⚠️ Educational demo only — not for clinical use")

# ---------------- INPUTS ----------------
age = st.selectbox(
    "Age group",
    ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
     "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

time_in_hospital = st.slider("Time in hospital (days)", 1, 14, 3)
num_lab_procedures = st.slider("Number of lab procedures", 1, 120, 40)
num_procedures = st.slider("Number of procedures", 0, 10, 1)
num_medications = st.slider("Number of medications", 1, 60, 10)
number_diagnoses = st.slider("Number of diagnoses", 1, 10, 3)

medical_specialty = st.selectbox(
    "Medical specialty",
    [
        "InternalMedicine",
        "Family/GeneralPractice",
        "Cardiology",
        "Emergency/Trauma",
        "Orthopedics",
        "Other",
        "Unknown"
    ]
)

diag_1 = st.text_input("Primary diagnosis (diag_1)", "250")
diag_2 = st.text_input("Secondary diagnosis (diag_2)", "401")
diag_3 = st.text_input("Tertiary diagnosis (diag_3)", "428")

insulin = st.selectbox("Insulin usage", ["No", "Steady", "Up", "Down"])
change = st.selectbox("Medication change during stay", ["No", "Ch"])
diabetesMed = st.selectbox("On diabetes medication", ["Yes", "No"])

# ---------------- MODEL COLUMN ORDER (CRITICAL) ----------------
MODEL_COLUMNS = [
    "age",
    "gender",
    "time_in_hospital",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "diag_1",
    "diag_2",
    "diag_3",
    "insulin",
    "change",
    "diabetesMed"
]

# ---------------- PREDICTION ----------------
if st.button("Predict Readmission Risk"):

    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "time_in_hospital": time_in_hospital,
        "medical_specialty": medical_specialty,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_diagnoses": number_diagnoses,
        "diag_1": str(diag_1),
        "diag_2": str(diag_2),
        "diag_3": str(diag_3),
        "insulin": insulin,
        "change": change,
        "diabetesMed": diabetesMed
    }])

    # enforce exact column order
    input_df = input_df[MODEL_COLUMNS]

    # safety: no NaNs in categorical columns
    input_df = input_df.fillna("Unknown")

    # prediction (NO cat_features here)
    prob = model.predict_proba(input_df)[0][1]

    st.subheader("Result")
    st.write(f"📊 **Readmission Probability:** {prob:.2%}")

    if prob >= 0.35:
        st.error("🔴 High risk of 30-day readmission")
    elif prob >= 0.2:
        st.warning("🟠 Moderate risk of 30-day readmission")
    else:
        st.success("🟢 Low risk of 30-day readmission")

