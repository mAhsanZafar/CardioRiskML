# CardioRiskML

**Machine Learning–Based Prediction of 30-Day Heart Failure Readmission**

---

## 🎯 Project Overview

**CardioRiskML** is an end-to-end Python project that uses machine learning to predict **30-day hospital readmission risk** for patients with **heart failure (HF)** using structured EHR-like data.

Hospital readmissions within 30 days are a key quality and cost metric in healthcare. This project demonstrates how predictive models can help identify high-risk patients at discharge, enabling clinicians to take proactive measures and improve patient outcomes.

This repository includes:
- Data preprocessing and feature engineering
- Multiple ML models with evaluation
- A reproducible Jupyter notebook
- Model serialization for reuse

---

## 🚀 Repository Structure

CardioRiskML/ ├── data/ │   └── CardioPulse.csv                # Synthetic EHR dataset │ ├── notebooks/ │   └── CardioRiskML.ipynb             # Full ML pipeline notebook │ ├── models/ │   └── cardioguard_model.pkl          # Serialized trained model │ ├── scripts/ │   └── generate_cardiopulse.py        # Synthetic dataset generator │ ├── README.md ├── requirements.txt └── .gitignore

---

## 📦 Features

**Data Processing**
- Imputes missing data and encodes categorical variables
- Normalizes numeric features
- Engineered temporal features:
  - Length of hospital stay
  - Days since last admission
  - Prior admission patterns

**Machine Learning Models**
- **Logistic Regression** — interpretable baseline classifier
- **Gradient Boosting (XGBoost)** — higher performance model

**Evaluation Metrics**
- AUC-ROC (primary metric)
- Precision, Recall, F1-score

**Explainability**
- Feature importance analysis to identify the strongest predictors

---

## 📈 Sample Results (Example)

These metrics help assess clinical usefulness:

| Model | AUC-ROC | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| Logistic Regression | ~0.75 | Balanced | High | Receivable |
| XGBoost | ~0.82 | Higher | High | Strong |

> Models prioritize **recall** to minimize missed high-risk patients. 0

---

## 🧠 Key Predictors Identified

- Elevated **BNP** levels at discharge
- Recent prior hospital admission
- Longer length of stay
- Hyponatremia (low sodium)
- Renal dysfunction (creatinine)

---

## 📊 Clinical Use Cases

This model supports:
- Early outpatient cardiology follow-ups
- Case-management enrollment
- Remote monitoring/telehealth interventions

---

## 🛠️ Getting Started

### 🔹 1. Clone the Repository

```bash
git clone https://github.com/mAhsanZafar/CardioRiskML.git
cd CardioRiskML
```
### 🔹 2. Install Dependencies

Use the provided requirements.txt:

pip install -r requirements.txt


---

### 🔹 3. Generate Synthetic Dataset (Optional)

If CardioPulse.csv is not present:

python scripts/generate_cardiopulse.py


---

### 🔹 4. Run the Notebook

Open the main notebook:

jupyter notebook notebooks/CardioRiskML.ipynb


---

### 🔹 5. Use the Trained Model

import joblib

model = joblib.load("models/cardioguard_model.pkl")
# Example: model.predict_proba(new_data)


---

### 📦 Model Serialization

The trained model is saved under:

models/cardioguard_model.pkl

Load it with joblib to make predictions on new EHR data.


---

### ⚠️ Disclaimer

This project uses synthetic data and is intended for research and learning purposes only.
It should not be used for clinical decision-making without rigorous validation, clinical oversight, and compliance with healthcare regulations.


---

### 💻 Technologies Used

The stack includes:

Python 3.10+

Pandas, NumPy

Scikit-Learn

XGBoost

Matplotlib & Seaborn

Joblib for model saving



---

### ✍️ Author

Muhammad Ahsan Zafar
AI/ML Developer & Healthcare AI Enthusiast

---

