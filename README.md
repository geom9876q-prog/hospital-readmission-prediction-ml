# 🏥 Hospital Readmission Prediction (ML)

Predicting **30-day hospital readmission risk** for diabetic patients using machine learning models, with a **live Streamlit web application**.

This project focuses on handling **real-world healthcare data challenges** such as class imbalance, categorical features, and model evaluation beyond accuracy.

---

## 📌 Problem Statement
Hospital readmissions within 30 days are costly and often preventable.  
The goal of this project is to **predict whether a patient will be readmitted within 30 days** after discharge, using clinical and hospitalization data.

---

## 📊 Dataset
- **Name:** Diabetes 130-US Hospitals (1999–2008)
- **Source:** UCI Machine Learning Repository  
  https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008
- **Size:** ~100,000 patient records, 50+ features

---

## 🧹 Data Preprocessing
- Removed identifiers and leakage-prone columns
- Dropped sparse and noisy medication features
- Handled missing categorical values using `"Unknown"`
- Converted target variable:
  - `<30` → `1` (readmitted)
  - `NO`, `>30` → `0` (not readmitted)
- Addressed **class imbalance** during modeling

---

## 🧠 Features Used
- Demographics: `age`, `gender`
- Hospital stay: `time_in_hospital`
- Utilization: `num_lab_procedures`, `num_procedures`, `num_medications`
- Diagnoses: `diag_1`, `diag_2`, `diag_3`
- Treatment indicators: `insulin`, `change`, `diabetesMed`
- Severity proxy: `number_diagnoses`

---

## 🤖 Models Trained

### 1️⃣ Logistic Regression (Baseline)
- One-hot encoding for categorical features
- `class_weight="balanced"`
- Strengths:
  - Interpretable
  - Higher recall for readmissions
- ROC-AUC ≈ **0.64**

---

### 2️⃣ CatBoost Classifier (Final Model)
- Native handling of categorical features
- No one-hot encoding required
- Better ranking performance
- ROC-AUC ≈ **0.66**

---

## 📐 Evaluation Metrics
Due to heavy class imbalance, accuracy was **not** the primary metric.

Metrics used:
- Recall (for readmitted patients)
- Precision
- F1-score
- **Macro Average**
- **ROC-AUC**

---

## 🌍 Live Web Application
A Streamlit-based web app allows users to input patient details and receive:
- Readmission probability
- Risk level (Low / Moderate / High)

> ⚠️ This application is for **educational purposes only** and not for clinical use.

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- CatBoost
- Streamlit

---

## 📁 Repository Structure
hospital-readmission-prediction-ml/
├── app.py
├── readmission_catboost.cbm
├── requirements.txt
├── README.md
├── notebooks/
│ ├── 01_data_cleaning.ipynb
│ ├── 02_logistic_regression.ipynb
│ └── 03_catboost_model.ipynb
└── data/
└── README.md



## 🚀 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
