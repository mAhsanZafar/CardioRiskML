import numpy as np
import pandas as pd
from datetime import timedelta

np.random.seed(42)

N = 5000  # number of patients

# -----------------------------
# Demographics
# -----------------------------
age = np.random.normal(68, 12, N).clip(30, 95).astype(int)
sex = np.random.choice(["Male", "Female"], N, p=[0.55, 0.45])
race = np.random.choice(
    ["White", "Black", "Hispanic", "Asian", "Other"],
    N,
    p=[0.55, 0.25, 0.12, 0.05, 0.03]
)

insurance = np.random.choice(
    ["Medicare", "Medicaid", "Private", "Uninsured"],
    N,
    p=[0.6, 0.15, 0.2, 0.05]
)

# -----------------------------
# Comorbidities
# -----------------------------
diabetes = np.random.binomial(1, 0.4, N)
hypertension = np.random.binomial(1, 0.7, N)
ckd = np.random.binomial(1, 0.3, N)
copd = np.random.binomial(1, 0.25, N)

# -----------------------------
# Labs (clinically realistic)
# -----------------------------
bnp = np.random.lognormal(mean=6.5, sigma=0.7, size=N)  # HF severity
creatinine = np.random.normal(1.6, 0.6, N).clip(0.6, 5.0)
sodium = np.random.normal(137, 4, N).clip(120, 150)
hemoglobin = np.random.normal(12.5, 1.8, N).clip(7, 18)

# -----------------------------
# Vitals
# -----------------------------
sbp = np.random.normal(130, 20, N).clip(80, 200)
dbp = np.random.normal(75, 12, N).clip(40, 120)
heart_rate = np.random.normal(82, 15, N).clip(40, 140)

# -----------------------------
# Medications at discharge
# -----------------------------
ace_inhibitor = np.random.binomial(1, 0.65, N)
beta_blocker = np.random.binomial(1, 0.75, N)
diuretic = np.random.binomial(1, 0.8, N)

# -----------------------------
# Admission history
# -----------------------------
admissions_last_6m = np.random.poisson(1.2, N).clip(0, 6)

length_of_stay = (
    np.random.gamma(shape=2.2, scale=3.0, size=N)
    .clip(1, 30)
    .astype(int)
)

days_since_last_admission = np.where(
    admissions_last_6m == 0,
    np.random.randint(90, 365, N),
    np.random.randint(5, 90, N)
)

# -----------------------------
# Dates
# -----------------------------
admission_date = pd.to_datetime("2023-01-01") + pd.to_timedelta(
    np.random.randint(0, 365, N), unit="D"
)

discharge_date = admission_date + pd.to_timedelta(length_of_stay, unit="D")
prev_discharge_date = admission_date - pd.to_timedelta(
    days_since_last_admission, unit="D"
)

# -----------------------------
# Readmission Risk Model (Ground Truth)
# -----------------------------
risk_score = (
    0.004 * bnp +
    0.8 * (sodium < 135) +
    1.2 * (admissions_last_6m >= 2) +
    0.04 * length_of_stay +
    0.6 * ckd +
    0.4 * copd -
    0.5 * beta_blocker -
    0.4 * ace_inhibitor
)

prob_readmit = 1 / (1 + np.exp(-risk_score))
readmitted_30d = np.random.binomial(1, prob_readmit)

# -----------------------------
# Build DataFrame
# -----------------------------
df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "race": race,
    "insurance": insurance,
    "diabetes": diabetes,
    "hypertension": hypertension,
    "ckd": ckd,
    "copd": copd,
    "bnp": bnp.round(1),
    "creatinine": creatinine.round(2),
    "sodium": sodium.round(1),
    "hemoglobin": hemoglobin.round(1),
    "sbp": sbp.round(0),
    "dbp": dbp.round(0),
    "heart_rate": heart_rate.round(0),
    "ace_inhibitor": ace_inhibitor,
    "beta_blocker": beta_blocker,
    "diuretic": diuretic,
    "admissions_last_6_months": admissions_last_6m,
    "length_of_stay": length_of_stay,
    "days_since_last_admission": days_since_last_admission,
    "admission_date": admission_date,
    "discharge_date": discharge_date,
    "prev_discharge_date": prev_discharge_date,
    "readmitted_30d": readmitted_30d
})

# -----------------------------
# Save
# -----------------------------
df.to_csv("CardioPulse.csv", index=False)
print("Synthetic CardioPulse dataset generated: CardioPulse.csv")
print("Readmission rate:", df["readmitted_30d"].mean().round(3))
