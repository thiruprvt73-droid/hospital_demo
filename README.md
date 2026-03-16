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
```

---

## 🚀 Step-by-Step Setup (VS Code)

### Step 1 — Open the project in VS Code
```
File → Open Folder → select hospital_demo/
```

### Step 2 — Open the terminal in VS Code
```
Terminal → New Terminal  (or Ctrl + `)
```

### Step 3 — Create a virtual environment
```bash
python -m venv venv
```

### Step 4 — Activate the virtual environment
```bash
# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```
You should see (venv) in your terminal prompt.

### Step 5 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 6 — Train the model (run once)
```bash
python train_model.py
```
This creates the `model/` folder with saved model files.

### Step 7 — Launch the app
```bash
streamlit run app.py
```
Your browser will open automatically at http://localhost:8501

---

## 🎯 How to Demo at Hackathon

1. Use the **left sidebar** to enter a patient's details
2. The **risk gauge** updates instantly with readmission probability
3. Show **Feature Importance** tab → explains what drives risk
4. Show **Patient Data** tab → all patients with color-coded risk tiers
5. Show **Analytics** tab → charts by diagnosis, age, prior admissions

### Two good demo patients to show judges:

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

## ⚠️ Limitations to Mention

- Dataset is small (30 patients) — prototype only
- Production would use real EHR data (e.g. MIMIC-III)
- Model not validated for clinical use
- Missing features: lab values, medications, comorbidities
