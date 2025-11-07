# 🏭 Industrial Failure Predictor (Remaining Useful Life Prediction)
### *Predictive Maintenance with CMAPSS, Feature Engineering, XGBoost, and a Streamlit Dashboard*

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)]()
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![Model](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)]()
[![Author](https://img.shields.io/badge/Author-Abdul-black)](https://github.com/KoukiFTW)

---

## 🧩 Overview

**Industrial Failure Predictor** forecasts machine failures **before** they happen using **NASA’s CMAPSS FD001 turbofan dataset**.  
It estimates each engine’s **Remaining Useful Life (RUL)** through feature engineering and machine learning, then visualizes results in a sleek **Streamlit dashboard**.

### 🧠 Problem Statement
Unexpected machine breakdowns cause downtime, costs, and safety risks.  
This project applies **predictive maintenance** to anticipate failures early and schedule maintenance proactively.

### 🚀 What This Project Does
- Loads **multivariate time-series sensor data**
- Engineers temporal features (lags, rolling stats, slopes)
- Trains an **XGBoost regression model** for RUL prediction
- Provides an **interactive Streamlit dashboard** for analysis, ranking, and CSV exports

---

## 🎯 Objectives
- Load and explore NASA CMAPSS FD001 dataset  
- Engineer predictive time-series features  
- Train and validate an RUL regressor with GroupKFold (no leakage)  
- Save model and preprocessing artifacts  
- Serve an interactive Streamlit dashboard for insights  

---

## 📦 Dataset: NASA CMAPSS (FD001)

| File | Description |
|------|--------------|
| `train_FD001.txt` | Training data; engines run to failure |
| `test_FD001.txt`  | Test data; truncated before failure |
| `RUL_FD001.txt`   | True RUL for each test engine |

**Schema**

| Column | Description |
|---------|-------------|
| unit | Engine ID |
| cycle | Time step (operating cycle) |
| op1..op3 | Operating conditions |
| s1..s21 | Sensor measurements |

🧾 **Label Definition:**  
> RUL = max(cycle per unit) - current_cycle  

📎 Files are **space-delimited** without headers.

---

## 🗂️ Project Structure

industrial-failure-predictor/
├── data/
│ └── raw/ <- CMAPSS dataset files
├── models/ <- Trained model & preprocessing artifacts
├── src/
│ ├── dataload.py <- File loading & column naming
│ ├── label.py <- RUL labeling
│ ├── features.py <- Feature engineering
│ ├── train_baseline.py <- Baseline Ridge/XGBoost
│ ├── train_fe.py <- Full feature + XGBoost training
│ ├── infer_fe.py <- RUL inference
│ └── utils.py <- Helpers
├── app/
│ └── streamlit_app.py <- Streamlit dashboard
├── notebooks/ <- EDA / experiments
├── tests/
├── requirements.txt
└── README.md

yaml
Copy code

---

## ⚙️ Setup & Installation

### 🧰 Prerequisites
- **OS:** Windows (tested)
- **Python:** 3.11+
- **Tools:** Git, optional FFmpeg

### 🔧 Create Environment
```bash
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
📦 Install Requirements
bash
Copy code
pip install -r requirements.txt
Example requirements.txt:

nginx
Copy code
pandas
numpy
scikit-learn
xgboost
joblib
matplotlib
plotly
streamlit
pyarrow
📂 Dataset Placement
bash
Copy code
data/raw/
  ├── train_FD001.txt
  ├── test_FD001.txt
  └── RUL_FD001.txt
🧠 How to Use
🏋️ Train Model
bash
Copy code
cd src
python train_fe.py
Artifacts saved to:

pgsql
Copy code
models/
  ├── xgb_rul_fd001.json
  └── preproc.joblib
💻 Run Dashboard
bash
Copy code
cd ..
streamlit run app/streamlit_app.py
Open in browser: http://localhost:8501

🔄 Typical Workflow
Confirm dataset files in data/raw/

Train model → saves artifacts

Launch app → explore fleet status

Export results as CSV

📊 Dashboard Features
✅ Upload CMAPSS-like file or use sample
✅ View per-unit predicted RUL with risk bands
✅ Drilldown plots: RUL over cycles per unit
✅ Overlay raw sensor signals
✅ CSV download of predictions

🚨 Risk Bands
Band	Threshold	Color
Critical	RUL ≤ 30	🔴 Red
Warning	30 < RUL ≤ 75	🟠 Amber
Healthy	RUL > 75	🟢 Green

Adjust thresholds per use case or industry standard.

🧬 Code Explanations
File	Purpose
dataload.py	Loads CMAPSS files, assigns column names, handles train/test/RUL splits.
label.py	Computes Remaining Useful Life = max(cycle) - cycle.
features.py	Adds normalized cycle, lag (t-1,t-3,t-5), rolling (mean/std/min/max), slope features.
train_baseline.py	Baseline Ridge/XGBoost model with minimal features.
train_fe.py	Full feature engineering + XGBoost training with GroupKFold.
infer_fe.py	Loads artifacts, rebuilds features, predicts latest RUL per unit.
streamlit_app.py	Dashboard to visualize predictions, risk bands, and trends.

🧮 Model Details
Setting	Description
Algorithm	XGBoost Regressor (reg:squarederror)
Validation	GroupKFold (5 splits by engine unit)
Metrics	RMSE, MAE
Features	Lag (t-1,t-3,t-5), Rolling (mean/std/min/max), Slopes (OLS over 10 cycles), op1–op3, cycle_norm
Artifacts	models/xgb_rul_fd001.json, models/preproc.joblib

📈 Example Performance
Fold	RMSE	MAE
1	18.2	13.9
2	17.8	14.2
3	18.0	13.7
4	17.5	13.5
5	18.1	14.0
Avg	17.9 ± 0.3	13.9 ± 0.3

📋 Example Output
unit	cycle	RUL_pred	risk
3	115	22.1	🔴 RED
5	87	61.3	🟠 AMBER
7	140	124.9	🟢 GREEN

🔍 Interpretation:
Lower predicted RUL → higher maintenance priority.
Use charts to confirm degradation patterns.

🧰 Troubleshooting
Issue	Fix
FileNotFoundError	Ensure files are in data/raw/.
mean_squared_error got unexpected keyword 'squared'	Use latest scikit-learn or compute RMSE manually.
Streamlit warning: ScriptRunContext missing	Launch with streamlit run app/streamlit_app.py.
“No feature overlap” error	Ensure dataset columns match CMAPSS schema (unit, cycle, op1..op3, s1..s21).

🧱 Technologies
Language: Python 3.x
Libraries: pandas, numpy, scikit-learn, xgboost, joblib, matplotlib, plotly, streamlit
Environment: Windows
Version Control: Git + GitHub

🧩 ML Concepts
Time-series feature engineering (lags, rolling stats)

Grouped cross-validation (avoid leakage)

Gradient boosting regression

Feature scaling & standardization

Maintenance risk mapping via RUL thresholds

🌐 Deployment (Optional via Hugging Face Spaces)
Steps

Create a new Hugging Face Space

Select SDK: Streamlit

Connect your GitHub repo

Default command:

bash
Copy code
streamlit run app/streamlit_app.py
Benefit: Shareable live demo for recruiters or portfolio display ✨

🔮 Future Improvements
Support CMAPSS FD002–FD004 (multi-condition)

Add SHAP feature explainability

Conformal prediction for uncertainty bounds

Asymmetric loss (penalize under-prediction)

Alerting system (Slack/email) for red units

Compare with sequence models (LSTM, Transformer)

🤝 Contributing
This is a solo portfolio project, but contributions are welcome!
Feel free to fork, improve, and submit PRs.
Please avoid committing large raw datasets.

👤 Author
Abdul
🎓 Computer Science Graduate
💼 GitHub Profile
🖥️ Project Type: Machine Learning Portfolio Project (Windows)

⚖️ License
MIT License © 2025 Abdul
sql
Copy code
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction...
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND.
🙏 Acknowledgments
NASA Prognostics Data Repository (CMAPSS)

scikit-learn, XGBoost, pandas, Streamlit open-source teams

Data science community for shared knowledge & tutorials

💡 “Predict failures before they happen — save time, money, and engines.”
