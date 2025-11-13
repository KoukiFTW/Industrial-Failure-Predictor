from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt

# =====================================================
# PATH SETUP
# =====================================================
APP_DIR = Path(__file__).resolve().parent
PROJ_DIR = APP_DIR.parent
SRC_DIR = PROJ_DIR / "src"
MODELS_DIR = PROJ_DIR / "models"
DATA_DIR = PROJ_DIR / "data" / "raw"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from features import build_features

# =====================================================
# PAGE LAYOUT CONFIG
# =====================================================
st.set_page_config(
    page_title="Industrial Machine Health Dashboard",
    page_icon="🛠️",
    layout="wide"
)

# CUSTOM HEADER
st.markdown(
    """
    <style>
        .big-title {
            font-size: 40px !important;
            font-weight: 700 !important;
            margin-bottom: -10px !important;
        }
        .sub-title {
            font-size: 18px !important;
            color: #666;
            margin-top: -10px !important;
        }
    </style>
    <p class="big-title">🛠 Industrial Machine Health Dashboard</p>
    <p class="sub-title">AI-powered Remaining Useful Life (RUL) prediction for turbofan engines</p>
    """,
    unsafe_allow_html=True,
)

st.write("")

# =====================================================
# LOAD MODEL ARTIFACTS
# =====================================================
preproc_path = MODELS_DIR / "preproc.joblib"
model_path = MODELS_DIR / "xgb_rul_fd001.json"

if not preproc_path.exists() or not model_path.exists():
    st.error(
        "Model files not found.\n"
        "Train your model first:\n\n"
        "`cd src`\n`python train_fe.py`"
    )
    st.stop()

meta = joblib.load(preproc_path)
scaler = meta["scaler"]
feature_cols = meta["features"]

model = xgb.Booster()
model.load_model(str(model_path))

# =====================================================
# SECTION 1 — DATA INPUT PANEL
# =====================================================
st.markdown("## 1. Data Input")

st.info(
    "Upload engine sensor data to analyse machine health. "
    "If no file is uploaded, the dashboard uses a built-in example dataset."
)

uploaded = st.file_uploader("Upload engine data file", type=["csv", "txt"])

if uploaded:
    df = pd.read_csv(uploaded, sep=r"\s+|,", engine="python", header=None)
else:
    sample_path = DATA_DIR / "test_FD001.txt"
    df = pd.read_csv(sample_path, sep=r"\s+", header=None)
    st.warning("Using sample dataset (test_FD001.txt)")

cols = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
df.columns = cols[:df.shape[1]]

with st.expander("📄 View Raw Data"):
    st.dataframe(df.head(50), height=300)

# =====================================================
# SECTION 2 — FEATURE GENERATION & PREDICTION
# =====================================================
st.markdown("## 2. Machine Health Prediction")

with st.spinner("Processing sensor data..."):
    df_feat = build_features(df)

    valid_cols = [c for c in feature_cols if c in df_feat.columns]
    X = scaler.transform(df_feat[valid_cols])
    df_feat["RUL_pred"] = model.predict(xgb.DMatrix(X))

# Latest cycle per engine
latest = df_feat.groupby("unit")["cycle"].idxmax()
summary = df_feat.loc[latest, ["unit", "cycle", "RUL_pred"]].sort_values("RUL_pred")

summary["risk"] = pd.cut(
    summary["RUL_pred"],
    bins=[-1, 30, 75, 1e9],
    labels=["🔴 CRITICAL", "🟠 WARNING", "🟢 HEALTHY"]
)

# =====================================================
# SECTION 3 — KPI CARDS (INDUSTRY LOOK)
# =====================================================
st.markdown("## 3. Fleet-Level Machine Health Overview")

critical = (summary["risk"] == "🔴 CRITICAL").sum()
warning = (summary["risk"] == "🟠 WARNING").sum()
healthy = (summary["risk"] == "🟢 HEALTHY").sum()

col1, col2, col3 = st.columns(3)

col1.metric("🔴 Critical Machines", critical)
col2.metric("🟠 Machines to Monitor", warning)
col3.metric("🟢 Healthy Machines", healthy)

st.write("")
st.dataframe(
    summary.rename(
        columns={
            "unit": "Engine ID",
            "cycle": "Last Observed Cycle",
            "RUL_pred": "Estimated Remaining Life (cycles)",
            "risk": "Risk State"
        }
    ),
    use_container_width=True,
    height=350
)

# =====================================================
# SECTION 4 — ENGINE-LEVEL ANALYSIS
# =====================================================
st.markdown("## 4. Engine-Level Analysis")

engine_ids = summary["unit"].tolist()
selected_engine = st.selectbox("Select an engine to inspect:", engine_ids)

engine_df = df_feat[df_feat["unit"] == selected_engine].sort_values("cycle")

st.markdown("### Remaining Life Prediction Over Time")
st.line_chart(
    engine_df.set_index("cycle")[["RUL_pred"]].rename(
        columns={"RUL_pred": "Estimated Remaining Life"}
    )
)

# sensor selection
all_sensors = [c for c in df.columns if c.startswith("s")]
default_sensors = ["s2", "s3", "s4"]

picked_sensors = st.multiselect(
    "Select sensors to visualize",
    options=all_sensors,
    default=[s for s in default_sensors if s in all_sensors]
)

if picked_sensors:
    st.markdown("### Sensor Behaviour Over Time")
    st.line_chart(engine_df.set_index("cycle")[picked_sensors])

# =====================================================
# SECTION 5 — DOWNLOAD REPORT
# =====================================================
st.markdown("## 5. Export Report")
st.download_button(
    label="📥 Download Engine Summary (CSV)",
    data=summary.to_csv(index=False),
    file_name="engine_health_report.csv",
    mime="text/csv"
)

# =====================================================
# SECTION 6 — TECHNICAL (OPTIONAL) MODEL ACCURACY
# =====================================================
st.markdown("## 6. Model Accuracy (Technical Section)")

results_path = MODELS_DIR / "test_results_fd001.csv"

if not results_path.exists():
    st.info(
        "`test_results_fd001.csv` not found.\n"
        "Generate it first:\n\n`cd src`\n`python test_evaluate.py`"
    )
else:
    test_res = pd.read_csv(results_path)

    y_true = test_res["RUL_true"]
    y_pred = test_res["RUL_pred_adjusted"]

    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    mae = np.mean(np.abs(y_true - y_pred))

    st.metric("Model RMSE", f"{rmse:.2f} cycles")
    st.metric("Model MAE", f"{mae:.2f} cycles")

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.7)
    ax.plot([0, max(y_true)], [0, max(y_true)], "r--")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title("Predicted vs True RUL (Test Set)")
    ax.grid(True)
    st.pyplot(fig)

st.caption("AI-powered predictive maintenance demo — built with Python, Streamlit & XGBoost.")