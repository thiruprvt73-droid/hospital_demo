import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ReadmitAI — Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size:13px !important; }

/* Main bg */
.main { background: #f8fafc; }

/* Metric cards */
.metric-card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Risk badge */
.risk-high   { background:#fff1f2; border:2px solid #fda4af; color:#be123c; border-radius:12px; padding:16px 24px; text-align:center; }
.risk-medium { background:#fffbeb; border:2px solid #fcd34d; color:#92400e; border-radius:12px; padding:16px 24px; text-align:center; }
.risk-low    { background:#f0fdf4; border:2px solid #86efac; color:#166534; border-radius:12px; padding:16px 24px; text-align:center; }
.risk-label  { font-size:13px; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:4px; }
.risk-pct    { font-size:42px; font-weight:600; font-family:'DM Mono', monospace; line-height:1; }
.risk-desc   { font-size:12px; margin-top:6px; opacity:0.8; }

/* Section header */
.section-header {
    font-size:11px; font-weight:600; letter-spacing:0.12em;
    text-transform:uppercase; color:#64748b; margin-bottom:12px;
}

/* Patient row */
.patient-row {
    display:flex; align-items:center; gap:12px;
    background:white; border-radius:10px; padding:12px 16px;
    border:1px solid #e2e8f0; margin-bottom:8px;
}

/* Disclaimer */
.disclaimer {
    background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px;
    padding:12px 16px; font-size:12px; color:#1e40af; margin-top:16px;
}
</style>
""", unsafe_allow_html=True)

# ── Load model + encoders ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model    = joblib.load("model/readmission_model.pkl")
    le_g     = joblib.load("model/le_gender.pkl")
    le_d     = joblib.load("model/le_diagnosis.pkl")
    le_t     = joblib.load("model/le_treatment.pkl")
    return model, le_g, le_d, le_t

@st.cache_data
def load_data():
    df = pd.read_csv("data/hospital_patients.csv")
    df['age'] = df['age'].replace('unknown', np.nan)
    df['age'] = pd.to_numeric(df['age']).fillna(round(pd.to_numeric(df['age'], errors='coerce').mean()))
    df['diagnosis'] = df['diagnosis'].fillna(df['diagnosis'].mode()[0])
    return df

model, le_gender, le_diagnosis, le_treatment = load_model()
df = load_data()

DIAGNOSES  = sorted(le_diagnosis.classes_.tolist())
TREATMENTS = sorted(le_treatment.classes_.tolist())
GENDERS    = sorted(le_gender.classes_.tolist())

FEATURE_NAMES = [
    'Age', 'Gender', 'Diagnosis',
    'Treatment', 'Days in Hospital', 'Previous Admissions'
]

def predict_patient(age, gender, diagnosis, treatment, days, prev_admissions):
    g = le_gender.transform([gender])[0]
    d = le_diagnosis.transform([diagnosis])[0]
    t = le_treatment.transform([treatment])[0]
    X = np.array([[age, g, d, t, days, prev_admissions]])
    prob = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]
    return pred, prob

def risk_tier(prob):
    if prob >= 0.70: return "HIGH",   "risk-high",   "Immediate follow-up recommended"
    if prob >= 0.40: return "MEDIUM", "risk-medium",  "Monitor closely after discharge"
    return "LOW", "risk-low", "Standard discharge protocol"

# ── Sidebar — Patient Input ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 ReadmitAI")
    st.markdown("<div style='color:#475569;font-size:13px;margin-bottom:24px'>Hospital Readmission Risk Predictor</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### Patient Details")

    age   = st.slider("Age", 18, 100, 55)
    gender = st.selectbox("Gender", GENDERS)
    diagnosis  = st.selectbox("Diagnosis", DIAGNOSES)
    treatment  = st.selectbox("Treatment Type", TREATMENTS)
    days       = st.slider("Days in Hospital", 1, 30, 5)
    prev_adm   = st.slider("Previous Admissions", 0, 10, 1)

    st.markdown("---")
    predict_btn = st.button("🔍  Predict Risk", use_container_width=True, type="primary")

    st.markdown("""
    <div class='disclaimer'>
    ⚠️ For demo purposes only.<br>Not a substitute for clinical judgment.
    </div>
    """, unsafe_allow_html=True)

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown("## Hospital Readmission Risk Dashboard")
st.markdown("<div style='color:#64748b;margin-bottom:24px'>Predict 30-day readmission risk using patient history & clinical features</div>", unsafe_allow_html=True)

# Top KPI strip ────────────────────────────────────────────────────────────────
total     = len(df)
readmit_n = (df['readmitted'] == 'Yes').sum()
avg_days  = df['days_in_hospital'].mean()
avg_prev  = df['num_previous_admissions'].mean()

k1, k2, k3, k4 = st.columns(4)
for col, label, val, sub in [
    (k1, "Total Patients",       total,              "in dataset"),
    (k2, "Readmitted",           f"{readmit_n} ({readmit_n/total*100:.0f}%)", "within 30 days"),
    (k3, "Avg. Stay",            f"{avg_days:.1f} days",   "per admission"),
    (k4, "Avg. Prior Admissions",f"{avg_prev:.1f}",        "per patient"),
]:
    col.markdown(f"""
    <div class='metric-card'>
        <div class='section-header'>{label}</div>
        <div style='font-size:28px;font-weight:600;color:#0f172a'>{val}</div>
        <div style='font-size:13px;color:#94a3b8;margin-top:2px'>{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Prediction result ────────────────────────────────────────────────────────────
pred_col, chart_col = st.columns([1, 2], gap="large")

with pred_col:
    st.markdown("<div class='section-header'>Prediction Result</div>", unsafe_allow_html=True)

    if predict_btn or True:   # show result on load with defaults too
        pred, prob = predict_patient(age, gender, diagnosis, treatment, days, prev_adm)
        tier, css_class, rec = risk_tier(prob)

        st.markdown(f"""
        <div class='{css_class}'>
            <div class='risk-label'>{tier} RISK</div>
            <div class='risk-pct'>{prob*100:.0f}%</div>
            <div class='risk-desc'>{rec}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Patient Summary</div>", unsafe_allow_html=True)
        summary_data = {
            "Field": ["Age", "Gender", "Diagnosis", "Treatment", "Days Admitted", "Prior Admissions"],
            "Value": [age, gender, diagnosis, treatment, days, prev_adm]
        }
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100),
            number={'suffix': '%', 'font': {'size': 28}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#be123c" if prob >= 0.7 else "#d97706" if prob >= 0.4 else "#16a34a"},
                'steps': [
                    {'range': [0, 40],  'color': '#f0fdf4'},
                    {'range': [40, 70], 'color': '#fffbeb'},
                    {'range': [70, 100],'color': '#fff1f2'},
                ],
                'threshold': {'line': {'color': "black", 'width': 2}, 'thickness': 0.8, 'value': prob*100}
            },
            title={'text': "Readmission Probability", 'font': {'size': 13}}
        ))
        fig_gauge.update_layout(height=220, margin=dict(t=40, b=0, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)

# Charts ───────────────────────────────────────────────────────────────────────
with chart_col:
    tab1, tab2, tab3 = st.tabs(["📊  Feature Importance", "📋  Patient Data", "📈  Analytics"])

    with tab1:
        st.markdown("<div class='section-header'>What drives readmission risk?</div>", unsafe_allow_html=True)
        importances = model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi_df['Importance'],
            y=fi_df['Feature'],
            orientation='h',
            marker=dict(
                color=fi_df['Importance'],
                colorscale=[[0,'#bfdbfe'],[0.5,'#3b82f6'],[1,'#1e3a8a']],
                showscale=False
            ),
            text=[f"{v:.1%}" for v in fi_df['Importance']],
            textposition='outside',
        ))
        fig_fi.update_layout(
            height=300,
            margin=dict(l=0, r=60, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(tickfont=dict(size=13)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        top_feat = fi_df.iloc[-1]['Feature']
        st.info(f"💡 **{top_feat}** is the strongest predictor of readmission in this dataset.")

    with tab2:
        st.markdown("<div class='section-header'>All Patients</div>", unsafe_allow_html=True)

        # Compute risk for all patients
        rows = []
        for _, r in df.iterrows():
            try:
                _, p = predict_patient(
                    r['age'], r['gender'], r['diagnosis'],
                    r['treatment'], r['days_in_hospital'], r['num_previous_admissions']
                )
                t, _, _ = risk_tier(p)
                rows.append({**r.to_dict(), 'Risk %': f"{p*100:.0f}%", 'Risk Tier': t})
            except Exception:
                pass

        display_df = pd.DataFrame(rows)[['age','gender','diagnosis','treatment',
                                          'days_in_hospital','num_previous_admissions',
                                          'readmitted','Risk %','Risk Tier']]
        display_df.columns = ['Age','Gender','Diagnosis','Treatment',
                               'Days','Prev Admissions','Actual','Risk %','Tier']

        def color_tier(val):
            return {
                'HIGH':   'background-color:#fff1f2;color:#be123c;font-weight:600',
                'MEDIUM': 'background-color:#fffbeb;color:#92400e;font-weight:600',
                'LOW':    'background-color:#f0fdf4;color:#166534;font-weight:600',
            }.get(val, '')

        styled = display_df.style.applymap(color_tier, subset=['Tier'])
        st.dataframe(styled, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("<div class='section-header'>Dataset Analytics</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            # Readmission by diagnosis
            diag_df = df.groupby('diagnosis')['readmitted'].apply(
                lambda x: (x == 'Yes').mean() * 100).reset_index()
            diag_df.columns = ['Diagnosis', 'Readmission Rate (%)']
            fig_d = px.bar(diag_df, x='Diagnosis', y='Readmission Rate (%)',
                           color='Readmission Rate (%)',
                           color_continuous_scale='Blues',
                           title='Readmission Rate by Diagnosis')
            fig_d.update_layout(height=250, margin=dict(t=40,b=20,l=0,r=0),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_d, use_container_width=True)

        with c2:
            # Age distribution
            fig_a = px.histogram(df, x='age', color='readmitted',
                                 barmode='overlay', nbins=10,
                                 title='Age Distribution by Readmission',
                                 color_discrete_map={'Yes':'#3b82f6','No':'#e2e8f0'})
            fig_a.update_layout(height=250, margin=dict(t=40,b=20,l=0,r=0),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                legend_title='Readmitted')
            st.plotly_chart(fig_a, use_container_width=True)

        # Prev admissions vs readmission
        fig_b = px.box(df, x='readmitted', y='num_previous_admissions',
                       color='readmitted',
                       color_discrete_map={'Yes':'#3b82f6','No':'#94a3b8'},
                       title='Previous Admissions vs Readmission Outcome',
                       labels={'readmitted':'Readmitted','num_previous_admissions':'Previous Admissions'})
        fig_b.update_layout(height=250, margin=dict(t=40,b=20,l=0,r=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

# Footer ───────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:12px'>"
    "ReadmitAI · Built with Streamlit · Random Forest · 83% Accuracy · "
    f"Model trained on {total} patients · {datetime.now().strftime('%Y')}"
    "</div>",
    unsafe_allow_html=True
)
