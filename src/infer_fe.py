# src/infer_fe.py
import os, pandas as pd, joblib, xgboost as xgb
from dataload import load_fd_file
from features import build_features

def predict_latest_per_unit(path="data/raw/train_FD001.txt"):
    # load artifacts
    meta = joblib.load("models/preproc.joblib")
    scaler = meta["scaler"]; feature_cols = meta["features"]

    model = xgb.Booster()
    model.load_model("models/xgb_rul_fd001.json")

    # load data
    df = load_fd_file(path)
    df_feat = build_features(df).copy()

    # align columns
    use_cols = [c for c in feature_cols if c in df_feat.columns]
    X = scaler.transform(df_feat[use_cols].values)

    d = xgb.DMatrix(X)
    df_feat["RUL_pred"] = model.predict(d)

    # last cycle per unit
    idx = df_feat.groupby("unit")["cycle"].idxmax()
    out = df_feat.loc[idx, ["unit","cycle","RUL_pred"]].sort_values("RUL_pred").reset_index(drop=True)
    return out

if __name__ == "__main__":
    print(predict_latest_per_unit().head())
