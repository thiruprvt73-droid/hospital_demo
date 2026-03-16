# 🏥 ReadmitAI — Hospital Readmission Predictor

A Streamlit demo app that predicts 30-day hospital readmission risk using a
Random Forest ML model. Built for hackathon demos.

---

## 📁 Project Structure

```
hospital_demo/
├── app.py                  ← Streamlit web app (run this)
├── train_model.py          ← Train & save the ML model
├── requirements.txt        ← Python dependencies
├── data/
│   └── hospital_patients.csv
└── model/                  ← Auto-created when you train
    ├── readmission_model.pkl
    ├── le_gender.pkl
    ├── le_diagnosis.pkl
    └── le_treatment.pkl

**High-risk patient (will show ~80%+ risk)**
- Age: 78, Female, Heart Disease, Surgery, 9 days, 5 prior admissions

**Low-risk patient (will show ~15% risk)**
- Age: 29, Male, Pneumonia, Antibiotics, 2 days, 0 prior admissions

---

## 📊 Model Details

| Property       | Value                        |
|----------------|------------------------------|
| Algorithm      | Random Forest (100 trees)    |
| Test Accuracy  | 83%                          |
| CV Accuracy    | 80% ± 7%                     |
| Features       | 6 (age, gender, diagnosis, treatment, days, prior admissions) |
| Dataset        | 30 patients (prototype)      |

---
