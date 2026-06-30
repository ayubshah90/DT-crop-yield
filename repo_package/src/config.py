"""
config.py — shared constants for the DT precision-agriculture pipeline.
All scripts import from here so changes propagate everywhere.
"""

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH    = "yield_df.csv"          # raw FAO dataset
MODELS_DIR   = "saved_models"          # pickled per-crop RF models
RESULTS_DIR  = "results"               # CSV metrics, figures

# ── Feature / target names ───────────────────────────────────────────────────
# Features used when training the GLOBAL model (crop identity included)
GLOBAL_FEATURES = [
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "Area_enc",
    "Item_enc",
    "Year",
]

# Features used in PER-CROP models (crop identity removed — redundant)
ENV_FEATURES = [
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "Area_enc",
    "Year",
]

TARGET = "hg/ha_yield"

# ── Modelling ────────────────────────────────────────────────────────────────
TEST_SIZE   = 0.20
RANDOM_SEED = 42
CV_FOLDS    = 5

# Random Forest hyper-parameters (tuned via 5-fold CV in 03_per_crop_models)
RF_PARAMS = dict(
    n_estimators = 200,
    max_features = "sqrt",
    min_samples_leaf = 2,
    random_state = RANDOM_SEED,
    n_jobs = -1,
)

# XGBoost hyper-parameters
XGB_PARAMS = dict(
    n_estimators  = 200,
    learning_rate = 0.05,
    max_depth     = 6,
    subsample     = 0.8,
    random_state  = RANDOM_SEED,
    n_jobs        = -1,
    verbosity     = 0,
)

# ── DT / MQTT simulation ─────────────────────────────────────────────────────
# Sensor noise levels for simulated DT readings
SENSOR_NOISE = {
    "avg_temp":                      0.5,   # ± °C
    "average_rain_fall_mm_per_year": 10.0,  # ± mm
    "pesticides_tonnes":             0.02,  # ± 2 % of value
}

# Number of daily readings the DT aggregates before invoking the ML model
SEASON_WINDOW_DAYS = 180

# MQTT topic template (Eclipse Ditto routing)
MQTT_TOPIC_TEMPLATE = "sensors/{sensor_id}/telemetry"
