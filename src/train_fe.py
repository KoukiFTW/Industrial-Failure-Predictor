# src/train_fe.py
import os, numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from dataload import load_fd001_split
from label import add_rul_labels
from features import build_features

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train_xgb_with_fe(train_path="data/raw/train_FD001.txt"):
    # 1) Load and label
    df = load_fd001_split(root=os.path.dirname(train_path), split="train")
    df = add_rul_labels(df)

    # 2) Build engineered features
    df_feat = build_features(df)

    # 3) Matrix
    feature_cols = [c for c in df_feat.columns if c not in ["unit","cycle","RUL"]]
    X = df_feat[feature_cols].values
    y = df_feat["RUL"].values
    groups = df_feat["unit"].values

    # 4) Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 5) Group K-Fold
    gkf = GroupKFold(n_splits=5)
    rmses, maes, models = [], [], []

    params = dict(
        objective="reg:squarederror",
        max_depth=6,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        nthread=0,
        min_child_weight=1,
        reg_lambda=1.0
    )

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        dtr = xgb.DMatrix(X[tr], label=y[tr])
        dva = xgb.DMatrix(X[va], label=y[va])

        model = xgb.train(
            params, dtr,
            num_boost_round=1200,
            evals=[(dva, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        pred = model.predict(dva)
        mse = mean_squared_error(y[va], pred)
        rmse = mse ** 0.5
        mae  = mean_absolute_error(y[va], pred)

        print(f"Fold {fold}: RMSE={rmse:.2f}  MAE={mae:.2f}")
        rmses.append(rmse); maes.append(mae)
        models.append(model)

    print(f"\nCV RMSE: {np.mean(rmses):.2f} ± {np.std(rmses):.2f}")
    print(f"CV  MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}")

    # 6) Save best fold model + preprocessing
    best = int(np.argmin(rmses))
    best_model = models[best]
    best_model.save_model(os.path.join(MODELS_DIR, "xgb_rul_fd001.json"))
    joblib.dump(
        {"scaler": scaler, "features": feature_cols, "model_type": "xgb_fe"},
        os.path.join(MODELS_DIR, "preproc.joblib")
    )
    print("Saved: models/xgb_rul_fd001.json")
    print("Saved: models/preproc.joblib")

if __name__ == "__main__":
    train_xgb_with_fe()
