"""
03_per_crop_models.py — Train one Random Forest per crop type.

Rationale
---------
Crop identity accounts for ~61 % of yield variance (see feature importances
in script 02). Training dedicated per-crop models removes this dominant
categorical signal and lets each model focus purely on environmental drivers
(temperature, rainfall, pesticides, location, year). This is agronomically
correct: wheat and cassava respond to weather in fundamentally different ways.

Outputs
-------
- saved_models/<crop_name>.pkl   : one pickled RF per crop
- saved_models/encoders.pkl      : LabelEncoders (needed at inference)
- results/per_crop_metrics.csv
- results/fig_per_crop_r2.png
"""

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")

from config import (
    ENV_FEATURES, MODELS_DIR, RANDOM_SEED,
    RESULTS_DIR, RF_PARAMS, TARGET, TEST_SIZE, CV_FOLDS,
)
from preprocessing import run as load_data


# ── Per-crop training ─────────────────────────────────────────────────────────

def train_per_crop(df: pd.DataFrame, encoders: dict) -> tuple[dict, pd.DataFrame]:
    """
    Train and evaluate one RF per crop.

    Returns
    -------
    crop_models : {crop_name: fitted RandomForestRegressor}
    metrics_df  : DataFrame with per-crop R², RMSE, MAE, CV-R², n
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    crops = df["Item"].unique()
    rows  = []
    crop_models = {}

    print(f"\n  Training per-crop models ({len(crops)} crops) …\n")

    for crop in sorted(crops):
        sub = df[df["Item"] == crop].copy()
        X   = sub[ENV_FEATURES]
        y   = sub[TARGET]
        n   = len(sub)

        # Train/test split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )

        # Cross-validated R²
        rf_cv = RandomForestRegressor(**RF_PARAMS)
        cv_r2 = cross_val_score(
            rf_cv, X_tr, y_tr,
            cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
            scoring="r2", n_jobs=-1,
        ).mean()

        # Final fit on full train split
        rf = RandomForestRegressor(**RF_PARAMS)
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_te)

        r2   = r2_score(y_te, preds)
        rmse = mean_squared_error(y_te, preds) ** 0.5
        mae  = mean_absolute_error(y_te, preds)

        crop_models[crop] = rf

        # Feature importances for this crop
        fi = pd.Series(rf.feature_importances_, index=ENV_FEATURES)
        top_feat = fi.idxmax()

        rows.append({
            "Crop":           crop,
            "n":              n,
            "R²":             round(r2, 4),
            "CV R² (5-fold)": round(cv_r2, 4),
            "RMSE":           round(rmse, 1),
            "MAE":            round(mae, 1),
            "Top feature":    top_feat,
        })

        print(f"  {crop:<28} n={n:>4}  R²={r2:.4f}  CV-R²={cv_r2:.4f}"
              f"  RMSE={rmse:>10,.1f}  top={top_feat}")

    metrics_df = (
        pd.DataFrame(rows)
        .set_index("Crop")
        .sort_values("R²", ascending=False)
    )
    return crop_models, metrics_df


# ── Persistence ───────────────────────────────────────────────────────────────

def save_models(crop_models: dict, encoders: dict) -> None:
    """Pickle each crop model and the encoders for use at inference time."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    for crop, model in crop_models.items():
        safe_name = crop.replace(" ", "_").replace(",", "").lower()
        path = os.path.join(MODELS_DIR, f"{safe_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)

    enc_path = os.path.join(MODELS_DIR, "encoders.pkl")
    with open(enc_path, "wb") as f:
        pickle.dump(encoders, f)

    print(f"\n  {len(crop_models)} models saved to '{MODELS_DIR}/'")
    print(f"  Encoders saved → {enc_path}")


def load_models() -> tuple[dict, dict]:
    """
    Load all per-crop models and encoders from disk.
    Call this in the DT integration script or MQTT handler.
    """
    encoders_path = os.path.join(MODELS_DIR, "encoders.pkl")
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)

    crop_models = {}
    for fname in os.listdir(MODELS_DIR):
        if not fname.endswith(".pkl") or fname == "encoders.pkl":
            continue
        with open(os.path.join(MODELS_DIR, fname), "rb") as f:
            model = pickle.load(f)
        # Reconstruct crop name from filename
        crop_key = fname.replace(".pkl", "").replace("_", " ")
        crop_models[crop_key] = model

    return crop_models, encoders


# ── Inference helper (reusable in DT pipeline) ─────────────────────────────

def predict_yield(
    crop_name: str,
    area: str,
    year: int,
    avg_temp: float,
    rainfall_mm: float,
    pesticides_tonnes: float,
    crop_models: dict,
    encoders: dict,
) -> float:
    """
    Predict yield for a single observation using the per-crop model.

    Parameters
    ----------
    crop_name          : e.g. "Wheat"
    area               : country name as in training data, e.g. "Italy"
    year               : integer year
    avg_temp           : annual mean temperature (°C)
    rainfall_mm        : annual rainfall (mm)
    pesticides_tonnes  : annual pesticide use (tonnes)
    crop_models        : dict of {crop_name: fitted RF}
    encoders           : {'Area': LabelEncoder, 'Item': LabelEncoder}

    Returns
    -------
    float : predicted yield in hg/ha
    """
    if crop_name not in crop_models:
        raise ValueError(f"No model for crop '{crop_name}'. "
                         f"Available: {list(crop_models.keys())}")

    le_area = encoders["Area"]
    try:
        area_enc = le_area.transform([area])[0]
    except ValueError:
        # Country not seen during training → use mean encoding
        area_enc = int(le_area.classes_.shape[0] / 2)

    X = pd.DataFrame([{
        "average_rain_fall_mm_per_year": rainfall_mm,
        "pesticides_tonnes":             pesticides_tonnes,
        "avg_temp":                      avg_temp,
        "Area_enc":                      area_enc,
        "Year":                          year,
    }])[ENV_FEATURES]  # enforce column order

    return float(crop_models[crop_name].predict(X)[0])


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_per_crop_r2(metrics_df: pd.DataFrame) -> None:
    """Horizontal bar chart — per-crop R² with CV R² error indication."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    crops = metrics_df.index.tolist()
    r2    = metrics_df["R²"].values
    cv_r2 = metrics_df["CV R² (5-fold)"].values

    y_pos = np.arange(len(crops))
    bars = ax.barh(y_pos, r2, color="#534AB7", alpha=0.85, height=0.5, label="Test R²")
    ax.barh(y_pos, cv_r2, color="#A9A6E0", alpha=0.6, height=0.5, label="CV R² (5-fold)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(crops, fontsize=10)
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel("R² score")
    ax.set_title("Per-Crop Random Forest Performance", fontsize=12)
    ax.axvline(0.95, color="red", linewidth=0.8, linestyle="--", label="R²=0.95 threshold")
    ax.legend(fontsize=9)

    for bar, v in zip(bars, r2):
        ax.text(v + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_per_crop_r2.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-crop R² figure saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> tuple[dict, dict]:
    """
    Full per-crop training pipeline.

    Returns
    -------
    crop_models : {crop_name: fitted RF}
    encoders    : {'Area': LabelEncoder, 'Item': LabelEncoder}
    """
    print("\n[03] Training per-crop Random Forest models …")

    df, encoders = load_data(verbose=False)
    crop_models, metrics_df = train_per_crop(df, encoders)

    print("\n── Per-crop metrics ─────────────────────────────────────")
    print(metrics_df.to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, "per_crop_metrics.csv"))
    print(f"\n  Metrics saved → {RESULTS_DIR}/per_crop_metrics.csv")

    save_models(crop_models, encoders)
    plot_per_crop_r2(metrics_df)

    # Quick inference demo
    print("\n── Inference demo ───────────────────────────────────────")
    demo_cases = [
        ("Wheat",     "Italy",  2010, 14.5, 800,  12000),
        ("Maize",     "Italy",  2010, 18.0, 850,  55000),
        ("Potatoes",  "France", 2008, 12.0, 700,  90000),
    ]
    for crop, area, yr, temp, rain, pest in demo_cases:
        pred = predict_yield(crop, area, yr, temp, rain, pest, crop_models, encoders)
        print(f"  {crop:<14} ({area}, {yr}) → {pred:,.0f} hg/ha")

    print("\n[03] Per-crop training complete.\n")
    return crop_models, encoders


if __name__ == "__main__":
    run()
