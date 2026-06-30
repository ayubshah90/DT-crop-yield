"""
Rebuilds the DT message round-trip swimlane diagram (sensor -> MQTT ->
inference -> Ditto Thing update) with improved, non-overlapping typography.

NOTE: This figure is illustrative/annotated (event labels and approximate
timing), not derived from a numeric dataset -- the same is true of the
original chart. The event positions (t0, t1, t2) and the example
predicted_yield value are kept identical to the original figure; only the
visual styling (font size, spacing, line weight) has been improved.

If you have an actual latency-logging script that records real per-message
wall-clock timestamps, send it and this can be regenerated from real data
instead.

USAGE:
    python plot_dt_roundtrip_revised.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 12.5,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "font.family": "serif",
    "axes.labelweight": "normal",
    "axes.titleweight": "normal",
})

# ── Event data (kept identical to original figure) ─────────────────────────
lanes = ["Sensor", "Inference service", "Eclipse Ditto"]
lane_colors = ["#dde7ef", "#e1ede1", "#f6e1e1"]
lane_text_colors = ["#1f5c8a", "#1a6b3a", "#a13a3a"]

events = [
    {"x": 0.0,  "lane": 0, "label_top": "broker + queue",
     "label_main": "$t_0$: publish\n`sensors/sensor01/telemetry`"},
    {"x": 4.0,  "lane": 1, "label_top": "predicted_yield = 37,374",
     "label_main": "$t_1$: consume\nRF inference"},
    {"x": 22.5, "lane": 2, "label_top": None,
     "label_main": "$t_2$: Thing PATCH applied"},
]

ditto_adapter_label_x = 14.5
ditto_adapter_label_lane = 1

# ── Build figure ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 4.2))

x_min, x_max = -0.8, 26.5
n_lanes = len(lanes)

# Draw lane background bands (start exactly at x=0, the axis line)
for i, color in enumerate(lane_colors):
    ax.axhspan(n_lanes - 1 - i - 0.5, n_lanes - 1 - i + 0.5,
               xmin=0, xmax=1, color=color, zorder=0)

# Lane labels: placed OUTSIDE the plot area (left of the y-axis line),
# x in axes-fraction (negative = left margin), y in data coordinates
# (so each label lines up with its lane's vertical centre), matching the
# original layout where labels sit outside the bands, not inside them.
for i, (lane, tcolor) in enumerate(zip(lanes, lane_text_colors)):
    ax.text(-0.012, n_lanes - 1 - i,
            lane, ha="right", va="center", fontsize=12,
            fontweight="bold", color=tcolor, zorder=5,
            transform=ax.get_yaxis_transform())

# "Ditto adapter" annotation in the inference-service lane
ax.text(ditto_adapter_label_x, n_lanes - 1 - ditto_adapter_label_lane + 0.18,
        "Ditto adapter", ha="center", va="center",
        fontsize=9.5, style="italic", color="#555555",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc", lw=0.6),
        zorder=4)

# Plot event markers and connecting arrows
xs = [e["x"] for e in events]
ys = [n_lanes - 1 - e["lane"] for e in events]

for i in range(len(events) - 1):
    ax.annotate(
        "", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
        arrowprops=dict(arrowstyle="-|>", color="#777777", lw=1.3,
                         shrinkA=10, shrinkB=10),
        zorder=3,
    )

for x, y in zip(xs, ys):
    ax.plot(x, y, marker="o", markersize=11, markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=1.6, zorder=6)

# Event text boxes (offsets tuned to avoid overlapping the title and lane labels)
text_offsets = [(0.5, 0.32), (0.5, 0.32), (-4.6, 0.42)]
for e, (dx, dy) in zip(events, text_offsets):
    x = e["x"]
    y = n_lanes - 1 - e["lane"]

    if e["label_top"]:
        ax.text(
            x + dx, y + dy + 0.30, e["label_top"],
            ha="left", va="bottom", fontsize=9.3, style="italic",
            color="#444444",
            bbox=dict(boxstyle="round,pad=0.22", fc="#f2f2f2", ec="#cccccc", lw=0.5),
            zorder=7,
        )

    ax.text(
        x + dx, y + dy, e["label_main"],
        ha="left", va="top" if dy > 0 else "center",
        fontsize=10, family="monospace" if "`" in e["label_main"] or "consume" in e["label_main"] else "serif",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", lw=0.8),
        zorder=7,
    )

ax.set_xlim(x_min, x_max)
ax.set_ylim(-0.6, n_lanes - 0.5 + 0.55)
ax.set_xlabel("Wall-clock time since publish (ms)", labelpad=6)
ax.set_xticks(range(0, 21, 5))
ax.set_yticks([])
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_linewidth(0.8)
ax.spines["left"].set_linewidth(0.8)
ax.spines["left"].set_visible(True)

ax.set_title(
    r"DT message round-trip: sensor $\rightarrow$ MQTT $\rightarrow$ inference $\rightarrow$ Ditto Thing update",
    pad=16, fontsize=12,
)

plt.tight_layout()
fig.subplots_adjust(left=0.18)

plt.savefig("fig_dt_roundtrip_revised.pdf", bbox_inches="tight", dpi=300)
plt.savefig("fig_dt_roundtrip_revised.png", bbox_inches="tight", dpi=300)

plt.show()

print("Saved: fig_dt_roundtrip_revised.pdf / .png")
