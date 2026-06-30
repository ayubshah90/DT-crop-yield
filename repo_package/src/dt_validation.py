"""
04_dt_validation.py — Validate DT-generated sensor data against historical FAO data.

Two experiments are run:

  Exp A — KS-test
      Compare the distribution of yield predictions produced from:
        (i)  real historical test-set features
        (ii) DT-simulated sensor readings (Gaussian noise added to the same
             base values to mimic real sensor variability)
      A small KS statistic (p > 0.05) means DT-generated data produces
      statistically indistinguishable predictions — validating the DT as
      a proxy for the physical system.

  Exp B — Seasonal aggregation pipeline
      The paper's core temporal-mismatch problem: the FAO model is annual
      but the DT streams sensor readings every few seconds.
      This script shows how to accumulate daily sensor readings into a
      seasonal window and invoke the model once per season — the correct
      scientific approach.

Outputs
-------
- Console: KS statistics, p-values, aggregation demo
- results/ks_test_results.csv
- results/fig_prediction_distributions.png
- results/fig_seasonal_aggregation.png
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

from config import (
    ENV_FEATURES, GLOBAL_FEATURES, RANDOM_SEED, RESULTS_DIR,
    RF_PARAMS, SEASON_WINDOW_DAYS, SENSOR_NOISE, TARGET, TEST_SIZE,
)
from preprocessing import run as load_data
from per_crop_models import predict_yield, run as train_per_crop_models


# ── Experiment A: KS-test ─────────────────────────────────────────────────────

def simulate_dt_sensor_data(
    X_reference: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Add realistic sensor noise to reference feature values to produce
    a dataset that mimics what an Eclipse Ditto DT would stream via MQTT.

    Noise levels are defined in config.SENSOR_NOISE.
    """
    X_sim = X_reference.copy()
    X_sim["avg_temp"] += rng.normal(
        0, SENSOR_NOISE["avg_temp"], len(X_sim)
    )
    X_sim["average_rain_fall_mm_per_year"] += rng.normal(
        0, SENSOR_NOISE["average_rain_fall_mm_per_year"], len(X_sim)
    )
    X_sim["pesticides_tonnes"] *= (
        1 + rng.normal(0, SENSOR_NOISE["pesticides_tonnes"], len(X_sim))
    )
    # Clip to physically plausible ranges
    X_sim["avg_temp"]  = X_sim["avg_temp"].clip(lower=0)
    X_sim["average_rain_fall_mm_per_year"] = X_sim[
        "average_rain_fall_mm_per_year"
    ].clip(lower=0)
    X_sim["pesticides_tonnes"] = X_sim["pesticides_tonnes"].clip(lower=0)
    return X_sim


def run_ks_experiment(
    rf_global: RandomForestRegressor,
    X_te: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    For each crop in the test set:
      1. Get predictions from real features  → preds_hist
      2. Add sensor noise                   → preds_dt
      3. Run KS test between distributions
    Also run the test on the full test set (all crops combined).
    """
    X_dt  = simulate_dt_sensor_data(X_te, rng)
    preds_hist = rf_global.predict(X_te)
    preds_dt   = rf_global.predict(X_dt)

    rows = []

    # Full dataset
    ks, p = ks_2samp(preds_hist, preds_dt)
    rows.append({
        "Subset":      "ALL CROPS",
        "n":           len(preds_hist),
        "KS statistic": round(ks, 4),
        "p-value":      round(p, 4),
        "Match (p>0.05)": p > 0.05,
    })

    # Per crop (using Item_enc to identify crops in test set)
    # We rely on the Item_enc column that was preserved in X_te
    if "Item_enc" in X_te.columns:
        for enc_val in X_te["Item_enc"].unique():
            mask = X_te["Item_enc"] == enc_val
            ks_c, p_c = ks_2samp(preds_hist[mask], preds_dt[mask.values])
            rows.append({
                "Subset":      f"Crop enc={enc_val}",
                "n":           int(mask.sum()),
                "KS statistic": round(ks_c, 4),
                "p-value":      round(p_c, 4),
                "Match (p>0.05)": p_c > 0.05,
            })

    return pd.DataFrame(rows).set_index("Subset"), preds_hist, preds_dt


def plot_prediction_distributions(preds_hist, preds_dt) -> None:
    """Overlay histogram: historical predictions vs DT-simulated predictions."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(0, max(preds_hist.max(), preds_dt.max()), 80)
    ax.hist(preds_hist, bins=bins, alpha=0.6, color="#534AB7",
            label="Historical (FAO test set)", density=True)
    ax.hist(preds_dt,   bins=bins, alpha=0.6, color="#1D9E75",
            label="DT-simulated sensor data", density=True)
    ax.set_xlabel("Predicted yield (hg/ha)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Prediction Distribution: Historical vs DT-Simulated\n"
                 "(KS-test p > 0.05 → distributions are statistically indistinguishable)",
                 fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_prediction_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Distribution figure saved → {out}")


# ── Experiment B: Seasonal aggregation pipeline ───────────────────────────────

class SeasonalAggregator:
    """
    Accumulates daily sensor readings streamed from the DT over a growing
    season and triggers a yield prediction once the window is full.

    This solves the temporal-scale mismatch: the RF model was trained on
    annual (season-level) aggregates, so it should be called once per season,
    not on every 10-second sensor reading.

    Usage
    -----
    agg = SeasonalAggregator(window_days=180)
    for reading in mqtt_stream:
        result = agg.ingest(reading)
        if result is not None:
            print("Season prediction:", result)
            agg.reset()
    """

    def __init__(self, window_days: int = SEASON_WINDOW_DAYS):
        self.window_days = window_days
        self.buffer: list[dict] = []

    def ingest(self, reading: dict) -> dict | None:
        """
        Add one daily reading to the buffer.
        Returns a seasonal aggregate dict when the window is full, else None.

        Parameters
        ----------
        reading : dict with keys:
            avg_temp, average_rain_fall_mm_per_year, pesticides_tonnes,
            day_index (int, 1-based)
        """
        self.buffer.append(reading)
        if len(self.buffer) >= self.window_days:
            return self._aggregate()
        return None

    def _aggregate(self) -> dict:
        buf = pd.DataFrame(self.buffer)
        return {
            "avg_temp":                      buf["avg_temp"].mean(),
            "average_rain_fall_mm_per_year": buf["average_rain_fall_mm_per_year"].mean(),
            "pesticides_tonnes":             buf["pesticides_tonnes"].mean(),
            "n_days":                        len(buf),
        }

    def reset(self):
        self.buffer = []

    @property
    def days_accumulated(self) -> int:
        return len(self.buffer)


def demo_seasonal_aggregation(
    crop_models: dict,
    encoders: dict,
    rng: np.random.Generator,
) -> None:
    """
    Simulate 365 days of sensor data for two crops and predict yield once
    per season (180-day window) using the per-crop models.
    """
    print("\n── Seasonal aggregation demo ────────────────────────────")

    scenarios = [
        {
            "crop": "Wheat", "area": "Italy", "year": 2023,
            "base_temp": 14.5, "base_rain": 780, "base_pest": 12000,
        },
        {
            "crop": "Maize", "area": "Italy", "year": 2023,
            "base_temp": 18.2, "base_rain": 820, "base_pest": 55000,
        },
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, sc in zip(axes, scenarios):
        agg = SeasonalAggregator(window_days=SEASON_WINDOW_DAYS)
        daily_temps = []
        predictions = []

        for day in range(1, 366):
            reading = {
                "avg_temp": sc["base_temp"] + rng.normal(0, 2),
                "average_rain_fall_mm_per_year": sc["base_rain"] + rng.normal(0, 15),
                "pesticides_tonnes": sc["base_pest"] * (1 + rng.normal(0, 0.01)),
                "day_index": day,
            }
            daily_temps.append(reading["avg_temp"])
            result = agg.ingest(reading)

            if result is not None:
                # Season window full → predict
                pred = predict_yield(
                    crop_name=sc["crop"],
                    area=sc["area"],
                    year=sc["year"],
                    avg_temp=result["avg_temp"],
                    rainfall_mm=result["average_rain_fall_mm_per_year"],
                    pesticides_tonnes=result["pesticides_tonnes"],
                    crop_models=crop_models,
                    encoders=encoders,
                )
                predictions.append({"day": day, "yield_hg_ha": pred})
                print(f"  {sc['crop']:<14} Day {day:>3}: "
                      f"seasonal mean temp={result['avg_temp']:.1f}°C  "
                      f"→ predicted yield = {pred:,.0f} hg/ha")
                agg.reset()

        # Plot daily temperature stream
        ax.plot(range(1, 366), daily_temps, color="#888", linewidth=0.6, alpha=0.7)
        ax.set_xlabel("Day of year")
        ax.set_ylabel("Daily temp (°C)")
        ax.set_title(f"{sc['crop']} — daily sensor stream\n"
                     f"(model invoked every {SEASON_WINDOW_DAYS} days)")

        # Mark prediction days
        for p in predictions:
            ax.axvline(p["day"], color="#534AB7", linewidth=1.2, linestyle="--")
            ax.text(p["day"] - 5, ax.get_ylim()[1] * 0.95,
                    f"{p['yield_hg_ha']:,.0f}", fontsize=8,
                    color="#534AB7", ha="right")

    plt.suptitle("Seasonal Aggregation: sensor stream → seasonal mean → ML prediction",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_seasonal_aggregation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Seasonal aggregation figure saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("\n[04] Running DT validation experiments …")
    rng = np.random.default_rng(RANDOM_SEED)

    # Load data
    df, encoders = load_data(verbose=False)
    X = df[GLOBAL_FEATURES]
    y = df[TARGET]
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Train a global RF for the KS experiment
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    rf_global = RandomForestRegressor(**RF_PARAMS)
    rf_global.fit(X_tr, y_tr)

    # Exp A — KS test
    print("\n── Experiment A: KS-test ────────────────────────────────")
    ks_df, preds_hist, preds_dt = run_ks_experiment(rf_global, X_te, rng)
    print(ks_df.to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ks_df.to_csv(os.path.join(RESULTS_DIR, "ks_test_results.csv"))
    print(f"\n  KS results saved → {RESULTS_DIR}/ks_test_results.csv")
    plot_prediction_distributions(preds_hist, preds_dt)

    # Exp B — Seasonal aggregation
    crop_models, enc = train_per_crop_models()
    demo_seasonal_aggregation(crop_models, enc, rng)

    print("\n[04] DT validation complete.\n")


if __name__ == "__main__":
    run()
