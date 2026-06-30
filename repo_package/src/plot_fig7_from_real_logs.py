"""
Regenerates Figure 7 (per-crop yield projection, 2026-2029) from the ACTUAL
RF prediction logs saved during the original experiment.

This script reads the real predicted_yield values directly from the JSON
logs (RF_predictions_log_<crop>.json) -- no estimation, no re-inference,
no placeholder data. The numbers plotted are exactly what your trained RF
model produced.

Font sizes increased to satisfy the IEEE footnote-size requirement (R1-8).

USAGE:
    Place this script in the same folder as a subfolder called
    "RF_predictions_all_crops" containing the JSON log files, then run:

        python plot_fig7_from_real_logs.py
"""

import json
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# ── Style: increase all font sizes for IEEE two-column legibility ──────────
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "font.family": "serif",
})

# ── Configuration ────────────────────────────────────────────────────────
LOG_DIR = "RF_predictions_all_crops"   # folder containing the JSON logs
CROPS_TO_PLOT = ["rice_paddy", "soybeans", "sweet_potatoes", "wheat", "yams"]
CROP_DISPLAY_NAMES = {
    "rice_paddy": "Rice Paddy",
    "soybeans": "Soybeans",
    "sweet_potatoes": "Sweet Potatoes",
    "wheat": "Wheat",
    "yams": "Yams",
}
sensors = ["sensor01", "sensor02", "sensor03", "sensor04"]
sensor_colors = {
    "sensor01": "#1f77b4",
    "sensor02": "#ff7f0e",
    "sensor03": "#2ca02c",
    "sensor04": "#d62728",
}


def load_crop_predictions(crop_key):
    """Reads RF_predictions_log_<crop>.json and returns {sensor: [yield_2026..2029]}."""
    path = os.path.join(LOG_DIR, f"RF_predictions_log_{crop_key}.json")
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Organise by sensor -> {year: predicted_yield}
    by_sensor = {s: {} for s in sensors}
    for r in records:
        by_sensor[r["sensor_id"]][int(r["year"])] = r["predicted_yield"]

    years = sorted({int(r["year"]) for r in records})
    series = {s: [by_sensor[s][y] for y in years] for s in sensors}
    return years, series


# ── Load all crop data from the real logs ───────────────────────────────────
data = {}
years_ref = None
for crop_key in CROPS_TO_PLOT:
    years, series = load_crop_predictions(crop_key)
    data[crop_key] = series
    years_ref = years  # assume consistent across crops

# ── Build figure: 2 rows x 3 cols grid (5 crops + 1 empty cell) ─────────────
n_crops = len(CROPS_TO_PLOT)
n_cols = 3
n_rows = int(np.ceil(n_crops / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 2.6 * n_rows))
axes = np.array(axes).reshape(-1)  # flatten for easy iteration

for idx, crop_key in enumerate(CROPS_TO_PLOT):
    ax = axes[idx]
    crop_data = data[crop_key]

    for sensor in sensors:
        ax.plot(
            years_ref,
            crop_data[sensor],
            color=sensor_colors[sensor],
            linewidth=1.8,
            label=sensor,
        )

    mean_vals = np.mean([crop_data[s] for s in sensors], axis=0)
    ax.plot(
        years_ref,
        mean_vals,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="Mean",
    )

    ax.set_title(CROP_DISPLAY_NAMES[crop_key], fontweight="bold", pad=6)
    ax.set_xlabel("Year")
    ax.set_xticks(years_ref)
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(True, alpha=0.3)

    if idx % n_cols == 0:
        ax.set_ylabel("Predicted yield (hg/ha)")

# Hide unused subplot(s)
for idx in range(n_crops, len(axes)):
    axes[idx].axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower right",
    bbox_to_anchor=(0.98, 0.08),
    ncol=1,
    frameon=True,
    fontsize=10,
)

plt.tight_layout(rect=[0, 0, 1, 1])

plt.savefig("fig_yield_projection_revised.pdf", bbox_inches="tight", dpi=300)
plt.savefig("fig_yield_projection_revised.png", bbox_inches="tight", dpi=300)

plt.show()

print("Saved: fig_yield_projection_revised.pdf / .png")
