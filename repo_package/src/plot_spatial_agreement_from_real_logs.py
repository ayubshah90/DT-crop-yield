"""
Regenerates Figure 8 (spatial agreement heatmap: CV across 4 sensors,
per crop-year cell) from the ACTUAL RF prediction logs.

CV is computed directly from the real predicted_yield values stored in
the JSON logs -- exact match to the original chart, no estimation.

USAGE:
    Place this script in the same folder as a subfolder called
    "RF_predictions_all_crops" containing the JSON log files, then run:

        python plot_spatial_agreement_from_real_logs.py
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Style: moderate, non-bold fonts sized for single-column IEEE width ──────
plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "serif",
    "axes.labelweight": "normal",
})

# ── Configuration ────────────────────────────────────────────────────────
LOG_DIR = "RF_predictions_all_crops"
CROP_ORDER = ["yams", "soybeans", "wheat", "sweet_potatoes", "rice_paddy"]  # top to bottom
CROP_DISPLAY_NAMES = {
    "yams": "Yams",
    "soybeans": "Soybeans",
    "wheat": "Wheat",
    "sweet_potatoes": "Sweet Potatoes",
    "rice_paddy": "Rice Paddy",
}
sensors = ["sensor01", "sensor02", "sensor03", "sensor04"]


def load_cv_matrix():
    """Returns (years, crop_labels, cv_matrix) with cv_matrix shape (n_crops, n_years)."""
    years = None
    cv_rows = []

    for crop in CROP_ORDER:
        path = os.path.join(LOG_DIR, f"RF_predictions_log_{crop}.json")
        with open(path, "r") as f:
            records = [json.loads(line) for line in f if line.strip()]

        by_year = {}
        for r in records:
            y = int(r["year"])
            by_year.setdefault(y, []).append(r["predicted_yield"])

        if years is None:
            years = sorted(by_year.keys())

        row = []
        for y in years:
            vals = np.array(by_year[y])
            cv = np.std(vals, ddof=0) / np.mean(vals) * 100.0
            row.append(cv)
        cv_rows.append(row)

    return years, [CROP_DISPLAY_NAMES[c] for c in CROP_ORDER], np.array(cv_rows)


years, crop_labels, cv_matrix = load_cv_matrix()

# ── Plot heatmap ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 3.6))

im = ax.imshow(cv_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0)

ax.set_xticks(range(len(years)))
ax.set_xticklabels(years)
ax.set_yticks(range(len(crop_labels)))
ax.set_yticklabels(crop_labels)
ax.set_xlabel("Year", labelpad=4)

# Annotate each cell with its value
norm = im.norm
cmap = im.cmap
for i in range(cv_matrix.shape[0]):
    for j in range(cv_matrix.shape[1]):
        val = cv_matrix[i, j]
        # choose text colour based on cell luminance for contrast
        rgba = cmap(norm(val))
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "white" if luminance < 0.55 else "black"
        ax.text(
            j, i, f"{val:.3f}",
            ha="center", va="center",
            fontsize=9.5, color=text_color,
        )

# Light gridlines between cells
ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(crop_labels), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.2)
ax.tick_params(which="minor", bottom=False, left=False)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("CV across 4 sensors (%)", fontsize=10.5)
cbar.ax.tick_params(labelsize=9.5)

plt.tight_layout()

plt.savefig("fig_spatial_agreement_revised.pdf", bbox_inches="tight", dpi=300)
plt.savefig("fig_spatial_agreement_revised.png", bbox_inches="tight", dpi=300)

plt.show()

print("Saved: fig_spatial_agreement_revised.pdf / .png")
