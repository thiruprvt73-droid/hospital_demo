import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/hospital_patients.csv")

# ── Clean data ─────────────────────────────────────────────────────────────────
df['age'] = df['age'].replace('unknown', np.nan)
df['age'] = pd.to_numeric(df['age'])
df['age'] = df['age'].fillna(round(df['age'].mean()))
df['diagnosis'] = df['diagnosis'].fillna(df['diagnosis'].mode()[0])

# ── Encode categorical columns ─────────────────────────────────────────────────
le_gender    = LabelEncoder().fit(df['gender'])
le_diagnosis = LabelEncoder().fit(df['diagnosis'])
le_treatment = LabelEncoder().fit(df['treatment'])

df['gender_enc']    = le_gender.transform(df['gender'])
df['diagnosis_enc'] = le_diagnosis.transform(df['diagnosis'])
df['treatment_enc'] = le_treatment.transform(df['treatment'])
df['readmitted_enc'] = (df['readmitted'] == 'Yes').astype(int)

FEATURES = ['age', 'gender_enc', 'diagnosis_enc',
            'treatment_enc', 'days_in_hospital', 'num_previous_admissions']

X = df[FEATURES]
y = df['readmitted_enc']

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ── Train model ────────────────────────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────────
preds = model.predict(X_test)
acc   = accuracy_score(y_test, preds)
cv    = cross_val_score(model, X, y, cv=5)

print(f"Test Accuracy  : {acc*100:.0f}%")
print(f"CV Accuracy    : {cv.mean()*100:.0f}% ± {cv.std()*100:.0f}%")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=['Not Readmitted', 'Readmitted']))

# ── Save everything ────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model,       "model/readmission_model.pkl")
joblib.dump(le_gender,   "model/le_gender.pkl")
joblib.dump(le_diagnosis,"model/le_diagnosis.pkl")
joblib.dump(le_treatment,"model/le_treatment.pkl")

print("\n✅  Model saved to model/readmission_model.pkl")
