"""
Regenerates the input trajectory figure (avg. temperature, annual rainfall,
pesticides, 2026-2029, per sensor) from the ACTUAL RF prediction logs.

The input feature values (avg_temp, rain, pesticides) are identical across
all crop JSON logs (they are the model inputs, not outputs), so any single
crop's log file is sufficient as the data source.

Font sizes increased to satisfy IEEE footnote-size requirement (R1-8).

USAGE:
    Place this script in the same folder as a subfolder called
    "RF_predictions_all_crops" containing the JSON log files, then run:

        python plot_input_trajectory_from_real_logs.py
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# ── Style: increase all font sizes for IEEE two-column legibility ──────────
plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10.5,
    "font.family": "serif",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
})

# ── Configuration ────────────────────────────────────────────────────────
LOG_DIR = "RF_predictions_all_crops"
SOURCE_CROP = "rice_paddy"   # any crop works; inputs are identical across crops
sensors = ["sensor01", "sensor02", "sensor03", "sensor04"]
sensor_colors = {
    "sensor01": "#1f77b4",
    "sensor02": "#ff7f0e",
    "sensor03": "#2ca02c",
    "sensor04": "#d62728",
}

FEATURES = [
    ("avg_temp", "Avg. temperature (\u00b0C)"),
    ("rain", "Annual rainfall (mm)"),
    ("pesticides", "Pesticides (tonnes)"),
]


def load_trajectory(crop_key):
    """Reads RF_predictions_log_<crop>.json and returns
    {feature: {sensor: [values_2026..2029]}} plus sorted years list."""
    path = os.path.join(LOG_DIR, f"RF_predictions_log_{crop_key}.json")
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    years = sorted({int(r["year"]) for r in records})

    by_feature = {feat: {s: {} for s in sensors} for feat, _ in FEATURES}
    for r in records:
        y = int(r["year"])
        s = r["sensor_id"]
        for feat, _ in FEATURES:
            by_feature[feat][s][y] = r[feat]

    series = {
        feat: {s: [by_feature[feat][s][y] for y in years] for s in sensors}
        for feat, _ in FEATURES
    }
    return years, series


years, series = load_trajectory(SOURCE_CROP)

# ── Build figure: 1 row x 3 feature panels ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))

for ax, (feat, ylabel) in zip(axes, FEATURES):
    for sensor in sensors:
        ax.plot(
            years,
            series[feat][sensor],
            marker="o",
            markersize=2.8,
            color=sensor_colors[sensor],
            linewidth=1.2,
            label=sensor,
        )
    ax.set_xlabel("Year", labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_xticks(years)
    ax.tick_params(axis="both", which="major", pad=3)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

fig.subplots_adjust(wspace=0.55)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=len(labels),
    bbox_to_anchor=(0.5, -0.13),
    frameon=False,
    title="Sensor",
    handlelength=1.8,
    columnspacing=1.2,
)

plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig("fig_input_trajectories_revised.pdf", bbox_inches="tight", dpi=300)
plt.savefig("fig_input_trajectories_revised.png", bbox_inches="tight", dpi=300)

plt.show()

print("Saved: fig_input_trajectories_revised.pdf / .png")
