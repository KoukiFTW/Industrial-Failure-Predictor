from pathlib import Path
import sys

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------------------------
# Paths and imports
# -------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJ_DIR = APP_DIR.parent
SRC_DIR = PROJ_DIR / "src"
MODELS_DIR = PROJ_DIR / "models"
DATA_DIR = PROJ_DIR / "data" / "raw"

# Make sure we can import from src/
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from dataload import load_fd_file
from features import build_features

# -------------------------------------------------
# Streamlit page setup
# -------------------------------------------------
st.set_page_config(
    page_title="Machine Health Monitor",
    layout="wide"
)

st.title("🛠️ Machine Health Monitor")
st.markdown(
    """
This tool estimates **how much life is left** in each engine based on its sensor readings.

- Each row represents **one engine**.
- The app estimates **remaining life in cycles**.
- Engines are grouped into **Green / Amber / Red** risk levels.
"""
)

st.markdown("---")

# -------------------------------------------------
# Load model artifacts
# -------------------------------------------------
preproc_path = MODELS_DIR / "preproc.joblib"
model_path = MODELS_DIR / "xgb_rul_fd001.json"

@st.cache_data(show_spinner=False)
def _artifacts_exist():
    return preproc_path.exists() and model_path.exists()

if not _artifacts_exist():
    st.error(
        "The trained model files were not found.\n\n"
        "Please train the model first by running in a terminal:\n\n"
        "```bash\n"
        "cd src\n"
        "python train_fe.py\n"
        "```\n"
        "This will create `models/preproc.joblib` and `models/xgb_rul_fd001.json`."
    )
    st.stop()

meta = joblib.load(preproc_path)
scaler = meta["scaler"]
feature_cols = meta["features"]

model = xgb.Booster()
model.load_model(str(model_path))

# -------------------------------------------------
# 1) Choose data
# -------------------------------------------------
st.header("1. Choose data to analyse")

st.markdown(
    """
You can either:

- **Upload a file** with engine sensor readings, or  
- Use the built-in **example dataset** from the NASA engine simulation.

The file should have columns like:  
`unit, cycle, op1, op2, op3, s1, s2, ..., s21`
"""
)

uploaded = st.file_uploader(
    "Upload engine data (CSV or space-separated text)",
    type=["csv", "txt"]
)

if uploaded is not None:
    df = pd.read_csv(uploaded, sep=r"\s+|,", engine="python", header=None)
    st.success("✅ File uploaded and loaded.")
else:
    fallback = DATA_DIR / "train_FD001.txt"
    if not fallback.exists():
        st.warning(
            "No file uploaded and no example dataset found at `data/raw/train_FD001.txt`.\n"
            "Please upload a CMAPSS-style file with engine data."
        )
        st.stop()
    df = pd.read_csv(fallback, sep=r"\s+", header=None)
    st.info("ℹ️ Using built-in example file: `data/raw/train_FD001.txt`")

# Assign CMAPSS FD001 column names
cols = ["unit", "cycle"] + [f"op{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
df.columns = cols[: df.shape[1]]

with st.expander("Show raw data (for reference)", expanded=False):
    st.dataframe(df.head(50), width="stretch")

# -------------------------------------------------
# 2) Predict remaining life for each engine
# -------------------------------------------------
st.header("2. Estimated remaining life for each engine")

with st.spinner("Analysing sensor trends and estimating remaining life..."):
    df_feat = build_features(df).copy()

    # Make sure we only use features that the model knows
    use_cols = [c for c in feature_cols if c in df_feat.columns]
    if not use_cols:
        st.error(
            "The model could not find any matching features in the data.\n"
            "Please double-check that your file has columns like: unit, cycle, op1..op3, s1..s21."
        )
        st.stop()

    X = scaler.transform(df_feat[use_cols].values)
    d = xgb.DMatrix(X)
    df_feat["RUL_pred"] = model.predict(d)

# Get the most recent reading for each engine
latest_idx = df_feat.groupby("unit")["cycle"].idxmax()
summary = df_feat.loc[latest_idx, ["unit", "cycle", "RUL_pred"]].sort_values("RUL_pred").reset_index(drop=True)

# Assign risk bands based on remaining life
summary["risk"] = pd.cut(
    summary["RUL_pred"],
    bins=[-1, 30, 75, 1e12],
    labels=["RED (Critical)", "AMBER (Monitor)", "GREEN (Healthy)"]
)

# Create a more human-friendly view
display = summary.rename(
    columns={
        "unit": "Engine ID",
        "cycle": "Current cycle (time)",
        "RUL_pred": "Estimated remaining life (cycles)"
    }
)

st.markdown(
    """
Each row below is **one engine**:

- **Estimated remaining life (cycles)**: how many cycles it can still run, based on sensor history.  
- **Risk level**:
  - 🟥 **RED (Critical)** – consider immediate inspection or maintenance  
  - 🟧 **AMBER (Monitor)** – keep an eye on it  
  - 🟩 **GREEN (Healthy)** – no urgent action needed  
"""
)

st.dataframe(display, width="stretch")

# -------------------------------------------------
# 3) Drill down into a single engine
# -------------------------------------------------
st.header("3. Look at one engine in detail")

if len(summary) == 0:
    st.warning("No engines found in the processed data.")
    st.stop()

engine_ids = display["Engine ID"].tolist()
selected_engine = st.selectbox("Choose an engine to inspect:", engine_ids)

u = df_feat[df_feat["unit"] == selected_engine].sort_values("cycle")

st.markdown(
    f"""
You are looking at **Engine {selected_engine}**.

The chart below shows how the **estimated remaining life** changes over time
as the engine accumulates more operating cycles.
"""
)

life_chart_data = u.set_index("cycle")[["RUL_pred"]].rename(
    columns={"RUL_pred": "Estimated remaining life (cycles)"}
)
st.line_chart(life_chart_data)

sensor_choices = [c for c in df.columns if c.startswith("s")]
default_sensors = [s for s in ["s2", "s3", "s4"] if s in sensor_choices][:3]

st.markdown(
    """
You can also see how specific sensor readings change over time.
This helps connect the **sensor behaviour** with the **remaining life estimate**.
"""
)

picked_sensors = st.multiselect(
    "Choose sensors to display (optional):",
    options=sensor_choices,
    default=default_sensors
)

if picked_sensors:
    sensor_chart_data = u.set_index("cycle")[picked_sensors]
    st.line_chart(sensor_chart_data)

# -------------------------------------------------
# 4) Download report
# -------------------------------------------------
st.header("4. Download summary for reporting")

st.markdown(
    """
You can download the current engine health summary as a CSV file
and share it with your team or attach it to reports.
"""
)

st.download_button(
    "⬇️ Download engine health summary (CSV)",
    data=display.to_csv(index=False),
    file_name="engine_health_summary.csv",
    mime="text/csv"
)

# -------------------------------------------------
# 5) Model accuracy on test data (for curious users)
# -------------------------------------------------
st.header("5. How accurate is this model? (optional technical section)")

st.markdown(
    """
This section is optional and intended for users who want to understand how well
the model performs on a separate test dataset.

We use a dedicated test set (not shown above) with known remaining life values.
We then compare the model's predictions to those true values.
"""
)

test_results_path = MODELS_DIR / "test_results_fd001.csv"

if not test_results_path.exists():
    st.info(
        "Test evaluation file not found.\n\n"
        "To generate it, run this once in a terminal:\n\n"
        "```bash\n"
        "cd src\n"
        "python test_evaluate.py\n"
        "```\n"
        "After that, refresh this page."
    )
else:
    test_res = pd.read_csv(test_results_path)

    if {"RUL_true", "RUL_pred_adjusted"}.issubset(test_res.columns):
        y_true = test_res["RUL_true"].values
        y_pred = test_res["RUL_pred_adjusted"].values

        rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
        mae = np.abs(y_true - y_pred).mean()

        st.markdown(
            f"""
**Test accuracy (on unseen data):**

- Average error (RMSE): **{rmse:.2f} cycles**  
- Average absolute error (MAE): **{mae:.2f} cycles**  

This means that, on average, the model's estimate of remaining life
is within roughly **{mae:.0f}–{rmse:.0f} cycles** of the true value.
"""
        )

        st.markdown("**True vs predicted remaining life for test engines:**")
        st.dataframe(
            test_res.head(10).rename(
                columns={
                    "unit": "Engine ID",
                    "RUL_true": "True remaining life (cycles)",
                    "RUL_pred_adjusted": "Predicted remaining life (cycles)"
                }
            ),
            width="stretch"
        )

        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.7)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
        ax.set_xlabel("True remaining life (cycles)")
        ax.set_ylabel("Predicted remaining life (cycles)")
        ax.set_title("Model accuracy on test engines")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)
    else:
        st.warning(
            "The file `test_results_fd001.csv` does not have the expected columns.\n"
            "Please re-run `python test_evaluate.py` from the `src` folder."
        )

st.markdown("---")
st.caption("Built as a learning and portfolio project – combining machine learning with a practical maintenance use case.")