"""
01_preprocessing.py — Load, explore, and preprocess the FAO crop-yield dataset.

Outputs
-------
- Console: dataset overview, class balance, descriptive statistics
- results/eda_summary.csv : per-crop summary statistics
- Encoders (Area_enc, Item_enc) returned as a dict for downstream scripts
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

from config import DATA_PATH, TARGET, RESULTS_DIR


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, drop index column if present, remove nulls."""
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} rows with missing values.")
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode Area and Item columns.
    Returns the modified DataFrame and a dict of fitted encoders,
    so they can be reused at inference time.
    """
    encoders = {}
    for col in ["Area", "Item"]:
        le = LabelEncoder()
        df[f"{col}_enc"] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def print_eda(df: pd.DataFrame) -> None:
    """Print a compact exploratory summary to the console."""
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"  Rows          : {len(df):,}")
    print(f"  Crops (unique): {df['Item'].nunique()}  → {sorted(df['Item'].unique())}")
    print(f"  Countries     : {df['Area'].nunique()}")
    print(f"  Year range    : {df['Year'].min()} – {df['Year'].max()}")
    print(f"  Missing values: {df.isnull().sum().sum()}")

    print("\n── Target distribution ──────────────────────────────────")
    print(df[TARGET].describe().round(1).to_string())
    skew = df[TARGET].skew()
    print(f"  Skewness: {skew:.3f}  (>1 = right-skewed; log-transform may help)")

    print("\n── Per-crop sample counts ──────────────────────────────")
    print(df.groupby("Item").size().sort_values(ascending=False).to_string())

    print("\n── Numerical feature correlations with target ──────────")
    num_cols = ["average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp", "Year"]
    corr = df[num_cols + [TARGET]].corr()[TARGET].drop(TARGET).round(3)
    print(corr.to_string())


def save_eda_summary(df: pd.DataFrame) -> None:
    """Write per-crop descriptive stats to CSV for the paper's appendix."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = (
        df.groupby("Item")[TARGET]
        .agg(["count", "mean", "std", "min", "max"])
        .rename(columns={"count": "n", "mean": "mean_yield",
                         "std": "std_yield", "min": "min_yield", "max": "max_yield"})
        .round(1)
    )
    out = os.path.join(RESULTS_DIR, "eda_summary.csv")
    summary.to_csv(out)
    print(f"\n  EDA summary saved → {out}")


def plot_distributions(df: pd.DataFrame) -> None:
    """
    Two-panel figure:
      left  — raw yield distribution per crop (violin)
      right — log-transformed yield (shows why RF handles skew better than LR)
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    order = df.groupby("Item")[TARGET].median().sort_values(ascending=False).index

    sns.violinplot(data=df, x="Item", y=TARGET, order=order,
                   ax=axes[0], palette="muted", cut=0, linewidth=0.8)
    axes[0].set_title("Yield distribution per crop (raw)", fontsize=12)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("hg/ha yield")
    axes[0].tick_params(axis="x", rotation=30)

    df["log_yield"] = np.log1p(df[TARGET])
    sns.violinplot(data=df, x="Item", y="log_yield", order=order,
                   ax=axes[1], palette="muted", cut=0, linewidth=0.8)
    axes[1].set_title("Yield distribution per crop (log-scale)", fontsize=12)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("log(hg/ha yield + 1)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_yield_distributions.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Distribution figure saved → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run(verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    df       : cleaned, encoded DataFrame
    encoders : {'Area': LabelEncoder, 'Item': LabelEncoder}
    """
    print("\n[01] Loading and preprocessing data …")
    df = load_and_clean(DATA_PATH)
    df, encoders = encode_categoricals(df)

    if verbose:
        print_eda(df)
        save_eda_summary(df)
        plot_distributions(df)

    print("\n[01] Preprocessing complete.\n")
    return df, encoders


if __name__ == "__main__":
    run(verbose=True)
