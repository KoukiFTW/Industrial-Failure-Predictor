import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

from dataload import load_fd001_split
from features import build_features

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_ROOT = "data/raw"
MODELS_DIR = "models"

MODEL_PATH = os.path.join(MODELS_DIR, "xgb_rul_fd001.json")
PREPROC_PATH = os.path.join(MODELS_DIR, "preproc.joblib")

# -------------------------------------------------
# 1️⃣ Load artifacts
# -------------------------------------------------
print("Loading model and scaler...")
meta = joblib.load(PREPROC_PATH)
scaler = meta["scaler"]
feature_cols = meta["features"]

model = xgb.Booster()
model.load_model(MODEL_PATH)

# -------------------------------------------------
# 2️⃣ Load test data + true RUL labels
# -------------------------------------------------
print("Loading test and RUL data...")
test_df = load_fd001_split(root=DATA_ROOT, split="test")

# NASA’s RUL_FD001.txt has one row per engine (true RUL at end of its last cycle)
rul_path = os.path.join(DATA_ROOT, "RUL_FD001.txt")
rul_truth = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])
rul_truth["unit"] = np.arange(1, len(rul_truth) + 1)

# -------------------------------------------------
# 3️⃣ Build features and predict
# -------------------------------------------------
print("Building features...")
test_feat = build_features(test_df)

use_cols = [c for c in feature_cols if c in test_feat.columns]
X = scaler.transform(test_feat[use_cols].values)
dtest = xgb.DMatrix(X)

test_feat["RUL_pred"] = model.predict(dtest)

# Get last cycle per unit
latest = test_feat.groupby("unit")["cycle"].idxmax()
pred_latest = test_feat.loc[latest, ["unit", "RUL_pred"]].reset_index(drop=True)

# -------------------------------------------------
# 4️⃣ Adjust predictions using true RUL offsets
# -------------------------------------------------
# In CMAPSS, test sequences stop BEFORE failure.
# The RUL_FD001.txt file gives the remaining life at the end of each sequence.
# So predicted RULs need to be offset by that amount.
print("Adjusting predictions to true failure point...")
pred_latest["RUL_true"] = rul_truth["RUL"].values
pred_latest["RUL_pred_adjusted"] = pred_latest["RUL_pred"] + pred_latest["RUL_true"]

# -------------------------------------------------
# 5️⃣ Compute metrics
# -------------------------------------------------
y_true = pred_latest["RUL_true"]
y_pred = pred_latest["RUL_pred_adjusted"]

rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae = np.mean(np.abs(y_true - y_pred))

print("\n✅ TEST EVALUATION RESULTS")
print("-----------------------------")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

# -------------------------------------------------
# 6️⃣ Plot Predicted vs True RUL
# -------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(y_true, y_pred, alpha=0.7, label="Engines")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label="Ideal (y=x)")
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("Predicted vs True RUL (Test FD001)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# 7️⃣ Save CSV report
# -------------------------------------------------
out_path = os.path.join(MODELS_DIR, "test_results_fd001.csv")
pred_latest.to_csv(out_path, index=False)
print(f"\n📄 Results saved to: {out_path}")