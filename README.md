# Industrial Failure Predictor (IFP)

Predict Remaining Useful Life (RUL) from multivariate sensor time‑series (e.g., turbofan engine CMAPSS FD001–FD004) with classic baselines and a feature‑engineered XGBoost model. Includes a Streamlit app for interactive exploration and inference.

---

## Introduction

Industrial Failure Predictor is a machine learning–based system designed to forecast when industrial machines, such as jet engines, are likely to fail. Using real sensor data collected over time — including readings like temperature, pressure, and vibration — the system analyzes performance patterns to estimate the Remaining Useful Life (RUL) of each component.

Built around NASA’s C-MAPSS turbofan engine dataset, the project implements a complete predictive maintenance pipeline: it automatically cleans and processes the data, engineers meaningful features, trains advanced models such as XGBoost, and delivers real-time insights through an interactive Streamlit web application.

By visualizing how an engine’s condition changes over time and predicting how long it can safely operate, the Industrial Failure Predictor enables smarter maintenance decisions. In short, it’s a predictive maintenance tool that helps organizations detect early signs of failure, minimize unplanned downtime, and optimize operational costs.

---

## 🚀 Project Overview

This repository trains and serves machine‑learning models that estimate the **Remaining Useful Life (RUL)** for industrial equipment from time‑stamped sensor readings. It provides:

* Clean data loaders for the FD00x splits
* Labeling utilities to compute RUL
* Reproducible feature pipelines (lags, rolling stats, slopes)
* Baseline model (Ridge/XGBoost on simple features)
* Feature‑engineered model (XGBoost on rich features)
* A Streamlit app to visualize predictions per unit over time

---

## ✨ Features

* **End‑to‑end flow**: load ➜ label ➜ feature‑engineer ➜ train ➜ evaluate ➜ infer.
* **Two modeling tracks**:

  * **Baseline**: lightweight features (`cycle_norm`, ops & raw sensors) + Ridge / XGBoost.
  * **Feature‑engineered (FE)**: windowed lags/rolling stats & recent slope features + XGBoost.
* **Group‑aware CV** by unit for fair evaluation.
* **Saved artifacts** (model + scaler + feature list) for consistent inference.
* **Interactive UI** via Streamlit to browse units, plot RUL predictions, and overlay raw sensors.

---

## 🧩 Project Structure

```
Industrial-Failure-Predictor-main/
├─ app/
│  └─ streamlit_app.py          # Streamlit UI for inference/visualization
├─ data/
│  └─ raw/                      # FD00x train/test/RUL files
│     ├─ train_FD001.txt        # Example split (others included)
│     ├─ test_FD001.txt
│     ├─ RUL_FD001.txt
│     └─ ...                    # FD002–FD004 equivalents
├─ models/                      # Saved models & preprocessing artifacts
│  └─ .gitkeep
├─ notebooks/
│  └─ 01_explore.ipynb          # Exploratory analysis
├─ src/
│  ├─ dataload.py               # Parsers for FD00x text files
│  ├─ features.py               # Feature builders (lags, rolling, slopes)
│  ├─ label.py                  # RUL labeling
│  ├─ train_baseline.py         # Baseline training (Ridge/XGB)
│  ├─ train_fe.py               # Feature‑engineered XGB training
│  ├─ infer_baseline.py         # Inference for baseline model
│  └─ infer_fe.py               # Inference for FE model
├─ requirements.txt             # Python dependencies
└─ .gitignore
```

---

## 🔧 Installation

> Requires Python 3.9+ (recommended) and a virtual environment.

```bash
# 1) Clone the repo
git clone https://github.com/IsthisAlif/Industrial-Failure-Predictor

# 2) Create & activate a virtual environment (example: venv)
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (Powershell)
.venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt
```

> **Note**: `xgboost` may require build tools on some platforms.

---

## 🗂️ Data

The models use the **C-MAPSS turbofan engine dataset** (FD001–FD004) originally released by **NASA**.

### Where to Download the Data

You can obtain the CMAPSS dataset from publicly available mirrors:

* **Kaggle** (most convenient): [https://www.kaggle.com/datasets/behrad3d/nasa-cmaps](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)
* **NASA Open Data Portal** (original source, may require login or request): [https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

  * Search for: **"CMAPSS Jet Engine Simulated Data"**

### After Downloading

1. Extract the files.
2. Locate the split you want (e.g., `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`).
3. Place them into:

```
data/raw/
```

Example structure:

```
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

Each `train/test` row has: `unit`, `cycle`, `op1..op3`, and `s1..s21` sensor columns.

---

## 🧠 What Each Code File Does

**`src/dataload.py`**

* `load_fd_file(path)` — loads one split (space‑separated text) with proper column names.
* `load_fd001_split(root, split)` — convenience loader for FD001 (`train|test|rul`).

**`src/label.py`**

* `add_rul_labels(df)` — computes per‑row RUL as `max_cycle(unit) - cycle`.

**`src/features.py`**

* `add_cycle_norm(df)` — normalizes cycles within each unit.
* `add_lags(df, lags=(1,3,5))` — lagged sensor features per unit.
* `add_rolling_stats(df)` — rolling mean/std features.
* `add_slope_last10(df)` — linear slope over last 10 cycles.
* `build_features(df)` — full FE pipeline; drops early NaNs after lagging.
* `build_baseline_features(df)` — smaller feature set for baseline track.

**`src/train_baseline.py`**

* Trains **Ridge** (default) or **XGBoost** on baseline features using group k‑fold by `unit`.
* Persists: `models/baseline_ridge.joblib` (or `baseline_xgb.json`) and `models/baseline_preproc.joblib` containing scaler & feature list.

**`src/train_fe.py`**

* Trains **XGBoost** on rich FE features; saves best fold model to `models/xgb_rul_fd001.json` and preprocessing to `models/preproc.joblib`.

**`src/infer_baseline.py`**

* Loads baseline artifacts and predicts RUL for a file; returns a table of **latest cycle per unit** sorted by predicted RUL.

**`src/infer_fe.py`**

* Loads FE artifacts and outputs latest‑cycle predictions per unit using the FE model.

**`app/streamlit_app.py`**

* Simple UI: pick model (Baseline/FE), pick split/file, view predicted **RUL over time** and optionally overlay raw sensors.

---

## 🏗️ Training — Step by Step

> Examples below use FD001; you can extend loaders for FD002–FD004 similarly.

### 1) Baseline model (Ridge by default)

```bash
# from project root
python -m src.train_baseline  # saves model + preprocessing under models/
```

Optional XGBoost baseline:

```bash
python - <<'PY'
from src.train_baseline import train_and_eval
train_and_eval(model_type="xgb")
PY
```

### 2) Feature‑engineered XGBoost

```bash
python -m src.train_fe  # saves xgb_rul_fd001.json + preproc.joblib
```

> Artifacts are written to `models/`. Keep these files to run inference or the Streamlit app.

---

## 🔮 Inference — CLI

### Baseline artifacts

```bash
python -m src.infer_baseline --  # or run as a script
```

Inside Python:

```python
from src.infer_baseline import predict_file
print(predict_file("data/raw/test_FD001.txt").head())
```

### Feature‑engineered artifacts

```python
from src.infer_fe import predict_latest_per_unit
print(predict_latest_per_unit("data/raw/test_FD001.txt").head())
```

Returns a dataframe with columns like: `unit, cycle, RUL_pred` (latest cycle per unit).

---

## 📊 Streamlit App

After training (so models exist under `models/`), launch:

```bash
streamlit run app/streamlit_app.py
```

Then:

1. Choose **Model** (Baseline or Feature‑Engineered).
2. Select a **Data file** (e.g., `train_FD001.txt` or `test_FD001.txt`).
3. Pick a **Unit** to visualize its predicted RUL trajectory.
4. (Optional) Overlay raw sensors for context.

---

## 📐 Evaluation

Both training scripts perform **GroupKFold** by `unit` and log **RMSE/MAE** per fold and average. Use these to compare baseline vs FE.

---

## 🛠️ Configuration Tips

* If you add or remove sensors, ensure feature builders and saved `feature_cols` remain aligned.
* When scoring new data, the **same scaler & features** from training (in `models/*preproc*.joblib`) must be applied.
* You can adapt `dataload.py` to other equipment formats by adjusting `COLS`.

---

## 🧪 Reproducibility

* Fix random seeds where applicable (NumPy/XGBoost) if you need exact repeatability.
* Record your environment using `pip freeze > requirements.lock.txt`.

---

## 🐛 Troubleshooting

* **`xgboost`**** install issues**: ensure compiler/build tools are available or use prebuilt wheels.
* **`ValueError: feature mismatch`**: you likely changed features; retrain or realign to the saved `feature_cols`.
* **Predictions are ****************`NaN`****************/empty**: early rows are dropped after lagging; ensure each unit has enough cycles for the selected lags/windows.

---

## 🗺️ Roadmap (ideas)

* Hyperparameter search (Optuna)
* Additional models (LightGBM, CatBoost, GRU/Transformer baselines)
* Cross‑dataset training for FD002–FD004 with unified loaders
* Model explainability (SHAP) and drift monitoring

---

## 🤝 Contributing

PRs welcome! Please open an issue describing the change first and keep contributions small & reviewable.

---

## 🙌 Acknowledgements

The FD00x data layout resembles the widely used turbofan CMAPSS setup. Credit to original dataset providers and the open‑source Python community.
