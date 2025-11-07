# src/features.py
import pandas as pd
import numpy as np

SENSORS = [f"s{i}" for i in range(1,22)]

def add_cycle_norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cycle_norm"] = out["cycle"] / out.groupby("unit")["cycle"].transform("max")
    return out

def add_lags(df: pd.DataFrame, sensors=SENSORS, lags=(1,3,5)) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("unit")
    for s in sensors:
        for L in lags:
            out[f"{s}_lag{L}"] = g[s].shift(L)
    return out

def add_rolling_stats(df: pd.DataFrame, sensors=SENSORS) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("unit")
    for s in sensors:
        out[f"{s}_mean5"] = g[s].rolling(5).mean().reset_index(level=0, drop=True)
        out[f"{s}_std5"]  = g[s].rolling(5).std().reset_index(level=0, drop=True)
        out[f"{s}_min10"] = g[s].rolling(10).min().reset_index(level=0, drop=True)
        out[f"{s}_max10"] = g[s].rolling(10).max().reset_index(level=0, drop=True)
    return out

def _rolling_slope(values: pd.Series) -> float:
    # OLS slope on last window
    v = values.to_numpy()
    if np.isnan(v).any() or len(v) < 2:
        return np.nan
    x = np.arange(len(v))
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, v, rcond=None)[0]
    return m

def add_slope_last10(df: pd.DataFrame, sensors=SENSORS) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("unit")
    for s in sensors:
        out[f"{s}_slope10"] = (g[s].rolling(10).apply(_rolling_slope, raw=False)
                                .reset_index(level=0, drop=True))
    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_cycle_norm(df)
    df = add_lags(df)
    df = add_rolling_stats(df)
    df = add_slope_last10(df)
    # After adding lags/rolling, early rows per unit will be NaN; drop them
    return df.dropna().reset_index(drop=True)

def build_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    # kept for compatibility with Step 2
    df = add_cycle_norm(df)
    keep = ["op1","op2","op3","cycle_norm"] + [c for c in df.columns if c in SENSORS]
    return df[["unit","cycle"] + keep + (["RUL"] if "RUL" in df.columns else [])]