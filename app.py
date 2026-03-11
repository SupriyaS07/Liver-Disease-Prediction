import streamlit as st
import numpy as np
import joblib
from PIL import Image

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="LiverGuard AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #F0FDF4; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #166534 0%, #14532D 100%);
        border-right: none;
    }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] .stRadio label {
        color: #FFFFFF !important;
        font-weight: 500;
        font-size: 15px;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.2) !important;
    }

    /* Hide default header */
    #MainMenu, footer, header { visibility: hidden; }

    /* Global text */
    html, body, [class*="css"] { color: #14532D; }
    h1, h2, h3, h4 { color: #166534 !important; }
    p, li, span { color: #166534; }

    /* ── METRIC CARDS ── */
    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 24px 16px;
        text-align: center;
        border: 1.5px solid #BBF7D0;
        box-shadow: 0 4px 16px rgba(34,197,94,0.10);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value {
        font-size: 36px;
        font-weight: 900;
        color: #16A34A;
        margin: 0;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 13px;
        color: #4ADE80;
        margin-top: 6px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── INFO CARDS ── */
    .info-card {
        background: white;
        border-radius: 14px;
        padding: 22px;
        border: 1.5px solid #BBF7D0;
        box-shadow: 0 2px 10px rgba(34,197,94,0.08);
        margin-bottom: 16px;
    }
    .info-card h4 {
        color: #166534 !important;
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .info-card p, .info-card li {
        color: #166534;
        font-size: 14px;
        line-height: 1.7;
    }

    /* ── SECTION HEADER ── */
    .section-header {
        background: linear-gradient(135deg, #16A34A 0%, #15803D 100%);
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 24px;
        color: white !important;
    }
    .section-header h2 {
        color: white !important;
        margin: 0;
        font-size: 22px;
    }
    .section-header p {
        color: rgba(255,255,255,0.8) !important;
        margin: 4px 0 0 0;
        font-size: 13px;
    }

    /* ── PREDICT RESULT ── */
    .result-disease {
        background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
        border: 2px solid #FECACA;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(239,68,68,0.12);
    }
    .result-healthy {
        background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
        border: 2px solid #86EFAC;
        border-radius: 20px;
        padding: 32px 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(34,197,94,0.12);
    }
    .result-icon { font-size: 52px; margin-bottom: 12px; }
    .result-title {
        font-size: 24px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .result-sub {
        font-size: 14px;
        color: #4B5563;
        line-height: 1.6;
    }

    /* ── INPUT FORM ── */
    .form-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1.5px solid #BBF7D0;
        box-shadow: 0 2px 12px rgba(34,197,94,0.08);
    }

    /* ── BUTTONS ── */
    .stButton > button {
        background: linear-gradient(135deg, #16A34A, #15803D) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 14px 28px !important;
        width: 100%;
        transition: all 0.3s !important;
        box-shadow: 0 4px 12px rgba(22,163,74,0.3) !important;
        letter-spacing: 0.3px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #15803D, #166534) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(22,163,74,0.4) !important;
    }

    /* ── INPUTS ── */
    .stNumberInput input {
        border: 1.5px solid #BBF7D0 !important;
        border-radius: 10px !important;
        background: #F8FFFA !important;
        color: #14532D !important;
        font-weight: 500;
    }
    .stNumberInput input:focus {
        border-color: #16A34A !important;
        box-shadow: 0 0 0 2px rgba(22,163,74,0.15) !important;
    }
    .stSelectbox > div > div {
        border: 1.5px solid #BBF7D0 !important;
        border-radius: 10px !important;
        background: #F8FFFA !important;
        color: #14532D !important;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: white;
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #BBF7D0;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 14px;
        color: #4ADE80 !important;
        border-radius: 8px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: #16A34A !important;
        color: white !important;
    }

    /* ── METRIC WIDGET ── */
    [data-testid="metric-container"] {
        background: white;
        border: 1.5px solid #BBF7D0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 10px rgba(34,197,94,0.08);
    }
    [data-testid="metric-container"] label { color: #4ADE80 !important; }
    [data-testid="stMetricValue"] { color: #16A34A !important; font-weight: 800 !important; }
    [data-testid="stMetricDelta"] { color: #4ADE80 !important; }

    /* ── TABLES ── */
    table { border-collapse: collapse; width: 100%; border-radius: 12px; overflow: hidden; }
    th {
        background: #16A34A !important;
        color: white !important;
        font-weight: 700;
        padding: 12px 16px;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    td { padding: 10px 16px; border-bottom: 1px solid #DCFCE7; color: #166534; font-size: 14px; }
    tr:nth-child(even) td { background: #F0FDF4; }
    tr:hover td { background: #DCFCE7; }

    /* ── DIVIDER ── */
    hr { border: none; border-top: 1.5px solid #BBF7D0; margin: 20px 0; }

    /* ── CAPTION ── */
    .stCaption { color: #4ADE80 !important; font-size: 12px; }

    /* ── ALERTS ── */
    .stInfo { background: #DCFCE7 !important; border-left: 4px solid #16A34A !important; border-radius: 10px; }
    .stSuccess { background: #F0FDF4 !important; border-left: 4px solid #4ADE80 !important; border-radius: 10px; }
    .stWarning { background: #FFFBEB !important; border-left: 4px solid #F59E0B !important; border-radius: 10px; }

    /* ── SCROLLBAR ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #F0FDF4; }
    ::-webkit-scrollbar-thumb { background: #86EFAC; border-radius: 10px; }

    /* ── IMAGE CARD ── */
    .img-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        border: 1.5px solid #BBF7D0;
        box-shadow: 0 2px 12px rgba(34,197,94,0.08);
        margin-bottom: 20px;
    }
    .img-title {
        font-size: 16px;
        font-weight: 700;
        color: #166534;
        margin-bottom: 12px;
    }
    .img-caption {
        font-size: 12px;
        color: #4ADE80;
        margin-top: 10px;
        font-style: italic;
    }

    /* ── BADGE ── */
    .badge {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #BBF7D0;
        margin: 3px;
    }

    /* ── ABOUT CARD ── */
    .about-card {
        background: linear-gradient(135deg, #166534, #14532D);
        border-radius: 20px;
        padding: 32px;
        color: white;
        margin-bottom: 20px;
    }
    .about-card h3, .about-card p, .about-card li { color: white !important; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("lr_tuned.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── LOAD IMAGES ────────────────────────────────────────────────
def load_img(path):
    try:
        return Image.open(path)
    except:
        return None

hero     = load_img("app_images/hero_banner.png")
logo     = load_img("app_images/logo.png")
cls_dist = load_img("app_images/class_distribution.png")
mdl_cmp  = load_img("app_images/model_comparison.png")
feat_imp = load_img("app_images/feature_importance.png")
roc      = load_img("app_images/roc_curve.png")
metrics  = load_img("app_images/best_model_metrics.png")

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    if logo:
        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.image(logo, width=65)
        with col_r:
            st.markdown("### LiverGuard")
            st.markdown("*AI Prediction*")
    else:
        st.markdown("## 🩺 LiverGuard AI")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔬 Predict", "📊 EDA Dashboard", "📈 Model Results", "ℹ️ About"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**📋 Dataset**")
    st.markdown("583 Patient Records")
    st.markdown("10 Lab Features")
    st.markdown("Binary Classification")
    st.markdown("---")
    st.markdown("**🏆 Best Model**")
    st.markdown("Logistic Regression")
    st.markdown("Recall → **1.000** ✅")
    st.markdown("F1-Score → **0.834**")
    st.markdown("Grade → **A+ 96/100**")
    st.markdown("---")
    st.markdown("**🧪 Tech Stack**")
    st.markdown("Python • Scikit-learn")
    st.markdown("Streamlit • Joblib")


# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home":

    if hero:
        st.image(hero, use_container_width=True)
    else:
        st.markdown("""
        <div class="section-header">
            <h2>🩺 LiverGuard AI — Liver Disease Prediction</h2>
            <p>Machine Learning powered healthcare prediction system</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric Cards ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">583</div>
            <div class="metric-label">Patient Records</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">8</div>
            <div class="metric-label">Models Trained</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">1.0</div>
            <div class="metric-label">Perfect Recall</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">96/100</div>
            <div class="metric-label">Project Grade</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── About + Highlights ──
    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🌿 About LiverGuard AI</h4>
            <p>
            <b>LiverGuard AI</b> is a machine learning application that predicts
            liver disease using biochemical lab test values from the
            <b>Indian Liver Patient Dataset (ILPD)</b>.<br><br>
            The app helps in early detection of liver disease by analyzing
            key markers such as Bilirubin levels, liver enzymes and protein values.
            Early detection can save lives by enabling faster treatment.
            </p>
            <br>
            <p><b>What this app offers:</b></p>
            <ul>
                <li>🔬 <b>Predict</b> liver disease for any new patient instantly</li>
                <li>📊 <b>Explore</b> the dataset through EDA charts</li>
                <li>📈 <b>Compare</b> performance of 8 machine learning models</li>
                <li>🏆 <b>Best Model:</b> Logistic Regression — Perfect Recall 1.0</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>📌 Key Highlights</h4>
            <p>🏥 <b>Domain:</b> Healthcare</p>
            <p>📋 <b>Dataset:</b> ILPD — 583 records</p>
            <p>🎯 <b>Problem:</b> Binary Classification</p>
            <p>⚖️ <b>Challenge:</b> 71.4% vs 28.6% class imbalance</p>
            <p>🔧 <b>Tuning:</b> GridSearchCV 5-fold CV</p>
            <p>✅ <b>Recall:</b> 1.000 — no patient missed</p>
            <p>🏆 <b>Grade:</b> A+ — 96/100</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h4>🧬 Top Features</h4>
            <span class="badge">Alkaline Phosphotase</span>
            <span class="badge">Direct Bilirubin</span>
            <span class="badge">ALT</span>
            <span class="badge">Total Bilirubin</span>
            <span class="badge">AST</span>
            <span class="badge">Age</span>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "🔬 Predict":

    st.markdown("""
    <div class="section-header">
        <h2>🔬 Patient Liver Disease Prediction</h2>
        <p>Enter the patient's biochemical lab values and get instant prediction</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1], gap="large")

    with col1:
        st.markdown("""<div class="form-card">""", unsafe_allow_html=True)
        st.markdown("#### 🧪 Patient Lab Values")
        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            age  = st.number_input("🧑 Age (years)", 1, 100, 45)
            tb   = st.number_input("🩸 Total Bilirubin", 0.0, 80.0, 1.0, 0.1)
            db   = st.number_input("💉 Direct Bilirubin", 0.0, 20.0, 0.3, 0.1)
            alkp = st.number_input("🔬 Alkaline Phosphotase", 50, 3000, 200)
            alt  = st.number_input("🧬 Alamine Aminotransferase", 5, 3000, 35)
        with c2:
            gender = st.selectbox("⚧ Gender", ["Male", "Female"])
            ast    = st.number_input("🔴 Aspartate Aminotransferase", 5, 5000, 40)
            tp     = st.number_input("🥩 Total Protiens", 1.0, 10.0, 6.5, 0.1)
            alb    = st.number_input("🟡 Albumin", 0.5, 6.0, 3.2, 0.1)
            agr    = st.number_input("📊 Albumin Globulin Ratio", 0.1, 3.0, 1.0, 0.1)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict Now", type="primary", use_container_width=True)
        st.markdown("""</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📋 Prediction Result")

        if predict_btn:
            gender_enc   = 1 if gender == "Male" else 0
            input_data   = np.array([[age, gender_enc, tb, db, alkp, alt, ast, tp, alb, agr]])
            input_scaled = scaler.transform(input_data)
            prediction   = model.predict(input_scaled)[0]
            probability  = model.predict_proba(input_scaled)[0]

            if prediction == 1:
                st.markdown(f"""
                <div class="result-disease">
                    <div class="result-icon">⚠️</div>
                    <div class="result-title" style="color:#DC2626;">
                        Liver Disease Detected
                    </div>
                    <div class="result-sub">
                        The model predicts this patient likely has liver disease.<br>
                        Please consult a liver specialist immediately.
                    </div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Disease Probability", f"{probability[0]*100:.1f}%", "High Risk")
            else:
                st.markdown(f"""
                <div class="result-healthy">
                    <div class="result-icon">✅</div>
                    <div class="result-title" style="color:#16A34A;">
                        No Liver Disease
                    </div>
                    <div class="result-sub">
                        The model predicts this patient is healthy.<br>
                        Regular checkups are still recommended.
                    </div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Healthy Probability", f"{probability[1]*100:.1f}%", "Low Risk")

            st.markdown("---")
            st.markdown("**📝 Input Summary**")
            st.markdown(f"""
            | Feature | Value |
            |---------|-------|
            | Age | {age} yrs |
            | Gender | {gender} |
            | Total Bilirubin | {tb} |
            | Direct Bilirubin | {db} |
            | Alkaline Phosphotase | {alkp} |
            | ALT | {alt} |
            | AST | {ast} |
            | Total Protiens | {tp} |
            | Albumin | {alb} |
            | AG Ratio | {agr} |
            """)

        else:
            st.markdown("""
            <div style='background:white; border:1.5px solid #BBF7D0;
                        border-radius:16px; padding:52px 24px; text-align:center;'>
                <div style='font-size:56px; margin-bottom:16px;'>🔬</div>
                <div style='font-size:16px; font-weight:600; color:#166534;'>
                    Fill in the patient details
                </div>
                <div style='font-size:13px; color:#4ADE80; margin-top:8px;'>
                    and click <b>Predict Now</b> to get the result
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#DCFCE7; border-radius:10px; padding:12px 16px;
                    border-left:4px solid #16A34A;'>
            <span style='font-size:12px; color:#166534;'>
            ⚠️ <b>Disclaimer:</b> This tool is for educational purposes only.
            Always consult a qualified medical professional for diagnosis.
            </span>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — EDA DASHBOARD
# ══════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":

    st.markdown("""
    <div class="section-header">
        <h2>📊 Exploratory Data Analysis</h2>
        <p>Visual insights from the Indian Liver Patient Dataset — 583 records, 10 features</p>
    </div>""", unsafe_allow_html=True)

    # Class Distribution
    st.markdown("""<div class="img-card">
        <div class="img-title">📊 Class Distribution</div>""",
        unsafe_allow_html=True)
    if cls_dist:
        st.image(cls_dist, use_container_width=True)
    st.markdown("""
        <div class="img-caption">
        71.4% of patients have liver disease (416) vs 28.6% no disease (167).
        This class imbalance was the biggest challenge — F1-Score and Recall
        were used as primary metrics instead of accuracy.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""<div class="img-card">
            <div class="img-title">🌿 Feature Importance</div>""",
            unsafe_allow_html=True)
        if feat_imp:
            st.image(feat_imp, use_container_width=True)
        st.markdown("""
            <div class="img-caption">
            Alkaline Phosphotase (0.23) is the most important feature followed by
            Direct Bilirubin (0.17) and ALT (0.14). Gender has almost zero importance.
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="img-card">
            <div class="img-title">📈 ROC Curve — All Models</div>""",
            unsafe_allow_html=True)
        if roc:
            st.image(roc, use_container_width=True)
        st.markdown("""
            <div class="img-caption">
            Logistic Regression achieves the highest AUC of 0.719.
            All models perform above the random baseline (0.5 dashed line).
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Key EDA findings
    st.markdown("""
    <div class="section-header">
        <h2>🔍 Key EDA Findings</h2>
        <p>Important insights discovered during data exploration</p>
    </div>""", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3, gap="large")
    with f1:
        st.markdown("""
        <div class="info-card">
            <h4>⚖️ Class Imbalance</h4>
            <p>71.4% disease vs 28.6% no disease. Stratified split and F1-Score used to handle this.</p>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="info-card">
            <h4>🔗 High Correlation</h4>
            <p>Total & Direct Bilirubin (r=0.87). ALT & AST (r=0.79). Tree models handle this naturally.</p>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="info-card">
            <h4>📊 Outliers Found</h4>
            <p>Bilirubin up to 75, Alkaline Phosphotase up to 2000+. Clinically real — retained in data.</p>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════
elif page == "📈 Model Results":

    st.markdown("""
    <div class="section-header">
        <h2>📈 Model Performance Results</h2>
        <p>Comparing all 8 models — 4 default and 4 tuned with GridSearchCV 5-fold cross validation</p>
    </div>""", unsafe_allow_html=True)

    # Model comparison image
    st.markdown("""<div class="img-card">
        <div class="img-title">🏆 All 8 Models — F1-Score Comparison</div>""",
        unsafe_allow_html=True)
    if mdl_cmp:
        st.image(mdl_cmp, use_container_width=True)
    st.markdown("""
        <div class="img-caption">
        Dark blue = Tuned models. Light blue = Default models.
        All tuned models outperform their default versions.
        Decision Tree showed highest improvement of +0.102 after tuning.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Best model metrics image
    st.markdown("""<div class="img-card">
        <div class="img-title">⭐ Best Model — Logistic Regression vs Random Forest</div>""",
        unsafe_allow_html=True)
    if metrics:
        st.image(metrics, use_container_width=True)
    st.markdown("""
        <div class="img-caption">
        Logistic Regression achieves perfect Recall of 1.0 —
        meaning no liver disease patient is ever missed.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metric cards
    st.markdown("""
    <div class="section-header">
        <h2>⭐ Best Model Scorecard — Logistic Regression Tuned</h2>
        <p>Performance on 117 test records after hyperparameter tuning</p>
    </div>""", unsafe_allow_html=True)

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Accuracy",  "71.8%",  "+2.6%")
    with m2:
        st.metric("Precision", "0.716",  "+0.012")
    with m3:
        st.metric("Recall",    "1.000",  "Perfect ✅")
    with m4:
        st.metric("F1-Score",  "0.834",  "+0.028")
    with m5:
        st.metric("AUC",       "0.719",  "Highest")

    st.markdown("<br>", unsafe_allow_html=True)

    # Two tables side by side
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📊 All 8 Models Comparison")
        st.markdown("""
        | Model | F1-Score | Recall | AUC |
        |-------|----------|--------|-----|
        | ⭐ LR Tuned | **0.834** | **1.000** | **0.719** |
        | DT Tuned | 0.818 | 0.976 | 0.644 |
        | LR Default | 0.806 | 0.904 | — |
        | RF Tuned | 0.794 | 0.904 | 0.694 |
        | KNN Tuned | 0.784 | 0.831 | 0.672 |
        | KNN Default | 0.747 | 0.747 | — |
        | RF Default | 0.727 | 0.771 | — |
        | DT Default | 0.716 | 0.699 | — |
        """)

    with col2:
        st.markdown("#### 🔄 Cross Validation Results (5-Fold)")
        st.markdown("""
        | Model | Mean F1 | Std Dev | Stability |
        |-------|---------|---------|-----------|
        | ⭐ LR Tuned | **0.834** | **0.003** | Most Stable |
        | RF Tuned | 0.831 | 0.017 | Good |
        | KNN Tuned | 0.820 | 0.011 | Good |
        | DT Tuned | 0.807 | 0.019 | Moderate |
        """)


# ══════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":

    st.markdown("""
    <div class="section-header">
        <h2>ℹ️ About This Project</h2>
        <p>Liver Disease Prediction — Machine Learning Capstone Project</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
        <div class="about-card">
            <h3>🌿 Project Details</h3>
            <p>📌 <b>Project:</b> Liver Disease Prediction</p>
            <p>🏥 <b>Domain:</b> Healthcare</p>
            <p>👩‍💻 <b>Student:</b> Supriya</p>
            <p>📋 <b>Dataset:</b> Indian Liver Patient Dataset</p>
            <p>🔢 <b>Records:</b> 583 patients</p>
            <p>🧪 <b>Features:</b> 10 biochemical lab features</p>
            <p>🤖 <b>Models:</b> 8 (4 default + 4 tuned)</p>
            <p>🏆 <b>Grade:</b> A+ — 96/100</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🧰 Tech Stack</h4>
            <span class="badge">Python</span>
            <span class="badge">Scikit-learn</span>
            <span class="badge">Pandas</span>
            <span class="badge">NumPy</span>
            <span class="badge">Matplotlib</span>
            <span class="badge">Seaborn</span>
            <span class="badge">Streamlit</span>
            <span class="badge">Joblib</span>
            <span class="badge">GridSearchCV</span>
            <span class="badge">PIL</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h4>📦 Models Saved</h4>
            <span class="badge">lr_tuned.pkl ⭐</span>
            <span class="badge">rf_tuned.pkl</span>
            <span class="badge">knn_tuned.pkl</span>
            <span class="badge">dt_tuned.pkl</span>
            <span class="badge">lr_default.pkl</span>
            <span class="badge">rf_default.pkl</span>
            <span class="badge">knn_default.pkl</span>
            <span class="badge">dt_default.pkl</span>
            <span class="badge">scaler.pkl</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # STAR method
    st.markdown("""
    <div class="section-header">
        <h2>⭐ STAR Method Summary</h2>
        <p>Project explained in interview format</p>
    </div>""", unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4, gap="large")
    with s1:
        st.markdown("""
        <div class="info-card">
            <h4>S — Situation</h4>
            <p>583 patient records with class imbalance (71.4% vs 28.6%),
            missing values, extreme outliers in lab features.</p>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="info-card">
            <h4>T — Task</h4>
            <p>Build complete ML pipeline — EDA, train and compare 8 models,
            document all design decisions and challenges.</p>
        </div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="info-card">
            <h4>A — Action</h4>
            <p>Cleaned data, performed EDA, trained 4 classifiers default +
            GridSearchCV tuned, evaluated with F1, Recall, AUC, Cross Validation.</p>
        </div>""", unsafe_allow_html=True)
    with s4:
        st.markdown("""
        <div class="info-card">
            <h4>R — Result</h4>
            <p>Logistic Regression — Perfect Recall 1.0, F1-Score 0.834,
            AUC 0.719. Grade A+ 96/100. Production ready pkl files.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#DCFCE7; border-radius:12px; padding:16px 20px;
                border-left:4px solid #16A34A; text-align:center;'>
        <span style='font-size:13px; color:#166534;'>
        ⚠️ <b>Disclaimer:</b> This application is for educational purposes only.
        It is not a substitute for professional medical advice, diagnosis or treatment.
        Always consult a qualified healthcare professional.
        </span>
    </div>
    """, unsafe_allow_html=True)