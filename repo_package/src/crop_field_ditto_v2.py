"""
Crop Field Digital Twin — Eclipse Ditto Integration
=====================================================
Polls 4 sensor twins from Eclipse Ditto every 25 seconds.
Run:  pip install matplotlib numpy requests
      python crop_field_ditto_v2.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive: renders to file without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec
import os
import requests, threading, time
from datetime import datetime

# ── Ditto config ──────────────────────────────────────────────────
DITTO_BASE_URL  = "http://localhost:8080"
DITTO_AUTH      = ("ditto", "ditto")
POLL_INTERVAL   = 25
SENSOR_TWINS = {
    "Sensor_ID_01": "org.example:sensor-temperature-01",
    "Sensor_ID_02": "org.example:sensor-temperature-02",
    "Sensor_ID_03": "org.example:sensor-humidity-01",
    "Sensor_ID_04": "org.example:sensor-humidity-02",
}
DITTO_FEATURE_PATH = {
    "Sensor_ID_01": ("temperature", "value"),
    "Sensor_ID_02": ("temperature", "value"),
    "Sensor_ID_03": ("humidity",    "value"),
    "Sensor_ID_04": ("humidity",    "value"),
}

# ── Field data ────────────────────────────────────────────────────
np.random.seed(42)
ROWS, COLS = 20, 40

field_base = np.random.choice(
    [0.3, 0.5, 0.7, 0.9, 1.0], size=(ROWS, COLS),
    p=[0.05, 0.10, 0.20, 0.40, 0.25]
)
field_base[0:10,  0:20]  = np.clip(field_base[0:10,  0:20]  + 0.12, 0, 1)
field_base[0:10,  20:40] = np.clip(field_base[0:10,  20:40] + 0.05, 0, 1)
field_base[10:20, 0:20]  = np.clip(field_base[10:20, 0:20]  - 0.08, 0, 1)
field_base[10:20, 20:40] = np.clip(field_base[10:20, 20:40] + 0.03, 0, 1)
field_base[4:7,   10:15] = np.random.uniform(0.1, 0.3, (3, 5))
field_base[12:15, 25:32] = np.random.uniform(0.15, 0.35, (3, 7))

# ── Zones & sensors ───────────────────────────────────────────────
ZONES = [
    dict(r0=0,  r1=9,  c0=0,  c1=19, name="Zone A", method="Drip Irrigation",
         color="#00bcd4",
         bullets=["Precision root-level delivery",
                  "Lowest water waste (~10%)",
                  "Best crop health output"]),
    dict(r0=0,  r1=9,  c0=20, c1=39, name="Zone B", method="Sprinkler System",
         color="#8bc34a",
         bullets=["Overhead spray coverage",
                  "Moderate efficiency (~65%)",
                  "Suited for dense canopy rows"]),
    dict(r0=10, r1=19, c0=0,  c1=19, name="Zone C", method="Flood / Furrow",
         color="#ff9800",
         bullets=["Surface flooding method",
                  "Higher water usage (~50%)",
                  "Risk of stress in clay soil"]),
    dict(r0=10, r1=19, c0=20, c1=39, name="Zone D", method="Micro-spray",
         color="#9c27b0",
         bullets=["Fine mist near canopy",
                  "Good humidity regulation",
                  "Mid-range water efficiency"]),
]

SENSORS = [
    dict(id="Sensor_ID_01", type="Temperature", row=3,  col=8,  color="#FF5722", marker="^"),
    dict(id="Sensor_ID_02", type="Temperature", row=14, col=30, color="#FF5722", marker="^"),
    dict(id="Sensor_ID_03", type="Humidity",    row=8,  col=22, color="#1E88E5", marker="o"),
    dict(id="Sensor_ID_04", type="Humidity",    row=17, col=5,  color="#1E88E5", marker="o"),
]

HEALTH_ROWS = [
    ("#d4a017", "Poor",      "0.0-0.2", "Severe stress / drought"),
    ("#a8d08d", "Low",       "0.3-0.4", "Suboptimal crop growth"),
    ("#4caf50", "Moderate",  "0.5-0.6", "Average crop vigour"),
    ("#2e7d32", "Good",      "0.7-0.8", "Healthy growth rate"),
    ("#1b5e20", "Excellent", "0.9-1.0", "Peak productivity"),
]

sensor_values = {s["id"]: None for s in SENSORS}
poll_status   = ["Eclipse Ditto  ·  simulated values active"]

# ── Ditto polling thread ──────────────────────────────────────────
def poll_ditto():
    while True:
        ok = 0
        for s in SENSORS:
            feat, prop = DITTO_FEATURE_PATH[s["id"]]
            url = (f"{DITTO_BASE_URL}/api/2/things/{SENSOR_TWINS[s['id']]}"
                   f"/features/{feat}/properties/{prop}")
            try:
                r = requests.get(url, auth=DITTO_AUTH, timeout=5)
                r.raise_for_status()
                sensor_values[s["id"]] = round(float(r.json()), 1)
                ok += 1
            except Exception as e:
                print(f"[Ditto] {s['id']}: {e}")
        ts = datetime.now().strftime("%H:%M:%S")
        if ok == len(SENSORS):
            poll_status[0] = f"Eclipse Ditto  |  Live  |  last sync {ts}"
        elif ok > 0:
            poll_status[0] = f"Eclipse Ditto  |  Partial ({ok}/{len(SENSORS)})  |  {ts}"
        else:
            poll_status[0] = f"Eclipse Ditto  |  unreachable  |  simulated  |  {ts}"
            for s in SENSORS:
                sensor_values[s["id"]] = round(
                    np.random.uniform(28, 38.5) if s["type"] == "Temperature"
                    else np.random.uniform(45, 78), 1)
        time.sleep(POLL_INTERVAL)

threading.Thread(target=poll_ditto, daemon=True).start()

# ── Colormap ──────────────────────────────────────────────────────
cmap = LinearSegmentedColormap.from_list(
    "crop_health",
    ["#d4a017", "#a8d08d", "#4caf50", "#2e7d32", "#1b5e20"], N=256
)

# ─────────────────────────────────────────────────────────────────
#  FIGURE LAYOUT
# ─────────────────────────────────────────────────────────────────
BG    = "#0a1628"
PANEL = "#0d1f35"
BORDER= "#1e3a5f"
MONO  = "monospace"

fig = plt.figure(figsize=(22, 12), facecolor=BG)
gs  = GridSpec(1, 2, figure=fig,
               left=0.01, right=0.99, top=0.94, bottom=0.05,
               wspace=0.025, width_ratios=[3.2, 1])

ax_field = fig.add_subplot(gs[0, 0])
ax_field.set_facecolor(BG)
for sp in ax_field.spines.values():
    sp.set_edgecolor(BORDER)

ax_panel = fig.add_subplot(gs[0, 1])
ax_panel.set_facecolor(PANEL)
ax_panel.set_xlim(0, 1)
ax_panel.set_ylim(0, 1)
ax_panel.axis("off")
# draw visible border on panel
for sp in ax_panel.spines.values():
    sp.set_visible(True)
    sp.set_edgecolor(BORDER)
    sp.set_linewidth(1.2)

# ── Field image ───────────────────────────────────────────────────
img_plot = ax_field.imshow(field_base, cmap=cmap, vmin=0, vmax=1,
                           aspect="auto", alpha=0.93, zorder=1)

cax = inset_axes(ax_field, width="2%", height="35%",
                 loc="lower right", borderpad=1.5)
cbar = fig.colorbar(img_plot, cax=cax)
cbar.set_label("Health Index", fontsize=9, color="white", labelpad=6)
cbar.set_ticks([0.1, 0.3, 0.5, 0.7, 0.9])
cbar.set_ticklabels(["Poor", "Low", "Med", "Good", "Excel"])
cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white", labelsize=8)
cbar.outline.set_edgecolor("#ffffff50")

ax_field.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
ax_field.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
ax_field.grid(which="minor", color="white", linewidth=0.18, alpha=0.15, zorder=2)
ax_field.tick_params(which="both", bottom=False, left=False,
                     labelbottom=False, labelleft=False)

# Zone overlays on field
for z in ZONES:
    r0, r1, c0, c1, bc = z["r0"], z["r1"], z["c0"], z["c1"], z["color"]
    ax_field.add_patch(mpatches.FancyBboxPatch(
        (c0-0.5, r0-0.5), (c1-c0+1), (r1-r0+1),
        boxstyle="round,pad=0.1",
        facecolor=bc+"28", edgecolor=bc,
        linewidth=1.8, linestyle="--", zorder=3, alpha=0.9
    ))
    ax_field.text(c0+0.5, r0+0.9, z["name"],
                  ha="left", va="top", fontsize=11, color=bc,
                  fontweight="bold", alpha=0.98, zorder=4, fontfamily=MONO)
    ax_field.text(c0+0.5, r0+2.3, f"({z['method']})",
                  ha="left", va="top", fontsize=9, color=bc,
                  alpha=0.90, zorder=4, fontfamily=MONO)

# Sensors
sensor_artists = {}
for s in SENSORS:
    r, c = s["row"], s["col"]
    g1, = ax_field.plot(c, r, "o", ms=22, color=s["color"], alpha=0.12, zorder=5)
    g2, = ax_field.plot(c, r, "o", ms=14, color=s["color"], alpha=0.22, zorder=5)
    ax_field.plot(c, r, s["marker"], ms=10, color=s["color"],
                  markeredgecolor="white", markeredgewidth=1.3, zorder=6)
    ox = 2.2 if c < COLS - 8 else -2.2
    oy = -2.5 if r > 3 else 2.5
    ha = "left" if c < COLS - 8 else "right"
    ann = ax_field.annotate(
        f"{s['id']}\n{s['type']}",
        xy=(c, r), xytext=(c+ox, r+oy),
        fontsize=9, color="white", fontweight="bold",
        ha=ha, va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=s["color"],
                  edgecolor="white", linewidth=1.2, alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="white", lw=1.0, alpha=0.7),
        zorder=7,
    )
    sensor_artists[s["id"]] = dict(ann=ann, g1=g1, g2=g2)

ax_field.set_title("Crop Field  —  Digital Twin Demo",
                   fontsize=14, fontweight="bold", color="white",
                   pad=10, fontfamily=MONO)

# ─────────────────────────────────────────────────────────────────
#  RIGHT PANEL — drawn using ax_panel.transAxes coordinates
#  Strategy: measure all content heights first, then place top-down
#  with exact fixed gaps so nothing overlaps.
# ─────────────────────────────────────────────────────────────────
T = ax_panel.transAxes

# Heights (all in axes fraction units, tuned to fit in 0→1)
H_SECTION_BAR   = 0.036   # height of the colored header bar
H_SECTION_GAP   = 0.014   # gap below header bar before content
H_ZONE_TITLE    = 0.028   # zone name line
H_ZONE_METHOD   = 0.024   # zone method line
H_ZONE_BULLET   = 0.023   # each bullet line
H_ZONE_GAP      = 0.012   # gap between zone cards
H_BETWEEN_SECTS = 0.018   # gap between major sections
H_SENSOR_TITLE  = 0.028
H_SENSOR_BULLET = 0.023
H_SENSOR_GAP    = 0.014
H_HEALTH_ROW    = 0.052   # height of each health index row

N_ZONE_BULLETS  = 3
N_SENSOR_BULLETS= 3
N_HEALTH_ROWS   = 5

# Pre-calculate total height used:
zone_card_h = (H_ZONE_TITLE + H_ZONE_METHOD +
               N_ZONE_BULLETS * H_ZONE_BULLET + H_ZONE_GAP)
sensor_block_h = (H_SENSOR_TITLE + N_SENSOR_BULLETS * H_SENSOR_BULLET + H_SENSOR_GAP)

total = (
    H_SECTION_BAR + H_SECTION_GAP +          # header 1
    4 * zone_card_h +                          # 4 zones
    H_BETWEEN_SECTS +
    H_SECTION_BAR + H_SECTION_GAP +           # header 2
    2 * sensor_block_h +                       # 2 sensor types
    H_BETWEEN_SECTS +
    H_SECTION_BAR + H_SECTION_GAP +           # header 3
    N_HEALTH_ROWS * H_HEALTH_ROW               # health rows
)

# Scale so content fills ~96% of the axes height
SCALE  = 0.96 / total
TOP_Y  = 0.985   # where we start drawing (just below top edge)

y = TOP_Y


def draw_header(y, title):
    """Draw a section header. Returns y at bottom of bar."""
    bot = y - H_SECTION_BAR * SCALE
    ax_panel.add_patch(mpatches.Rectangle(
        (0.0, bot), 1.0, H_SECTION_BAR * SCALE,
        facecolor="#112240", edgecolor="none",
        transform=T, clip_on=False, zorder=2
    ))
    ax_panel.text(0.5, y - (H_SECTION_BAR * SCALE * 0.35),
                  title, ha="center", va="top",
                  fontsize=9.5, fontweight="bold", color="#80cbc4",
                  fontfamily=MONO, transform=T, zorder=3)
    return bot - H_SECTION_GAP * SCALE


def draw_zone(y, z):
    """Draw one zone card. Returns y at bottom."""
    bc = z["color"]
    card_h = zone_card_h * SCALE
    ax_panel.add_patch(mpatches.FancyBboxPatch(
        (0.025, y - card_h), 0.950, card_h,
        boxstyle="round,pad=0.008",
        facecolor=bc + "1a", edgecolor=bc,
        linewidth=1.0, transform=T, clip_on=False, zorder=2
    ))
    cy = y - 0.005
    ax_panel.text(0.055, cy, z["name"],
                  ha="left", va="top", fontsize=10,
                  color=bc, fontweight="bold",
                  fontfamily=MONO, transform=T, zorder=3)
    cy -= H_ZONE_TITLE * SCALE
    ax_panel.text(0.055, cy, z["method"],
                  ha="left", va="top", fontsize=9,
                  color="white", fontfamily=MONO, transform=T, zorder=3)
    cy -= H_ZONE_METHOD * SCALE
    for b in z["bullets"]:
        ax_panel.text(0.065, cy, f"· {b}",
                      ha="left", va="top", fontsize=8.5,
                      color="white", fontfamily=MONO, transform=T, zorder=3)
        cy -= H_ZONE_BULLET * SCALE
    return y - card_h - H_ZONE_GAP * SCALE


def draw_sensor(y, marker, color, title, bullets):
    """Draw one sensor entry. Returns y at bottom."""
    ax_panel.text(0.055, y, f"{marker}  {title}",
                  ha="left", va="top", fontsize=10,
                  color=color, fontweight="bold",
                  fontfamily=MONO, transform=T)
    y -= H_SENSOR_TITLE * SCALE
    for b in bullets:
        ax_panel.text(0.075, y, f"· {b}",
                      ha="left", va="top", fontsize=8.5,
                      color="white", fontfamily=MONO, transform=T)
        y -= H_SENSOR_BULLET * SCALE
    return y - H_SENSOR_GAP * SCALE


def draw_health(y, color, label, rng, desc):
    """Draw one health index row. Returns y at bottom."""
    row_h = H_HEALTH_ROW * SCALE
    swatch_h = row_h * 0.60
    ax_panel.add_patch(mpatches.FancyBboxPatch(
        (0.04, y - swatch_h), 0.07, swatch_h,
        boxstyle="round,pad=0.003",
        facecolor=color, edgecolor="none",
        transform=T, clip_on=False
    ))
    ax_panel.text(0.13, y - 0.002, f"{label}  {rng}",
                  ha="left", va="top", fontsize=10,
                  color="white", fontweight="bold",
                  fontfamily=MONO, transform=T)
    ax_panel.text(0.13, y - swatch_h * 0.55, desc,
                  ha="left", va="top", fontsize=8.5,
                  color="white", fontfamily=MONO, transform=T)
    return y - row_h


# ── Draw all sections ─────────────────────────────────────────────
y = draw_header(y, "◈  IRRIGATION ZONES")
for z in ZONES:
    y = draw_zone(y, z)

y -= H_BETWEEN_SECTS * SCALE
y = draw_header(y, "◈  SENSOR TYPES")
y = draw_sensor(y, "▲", "#FF5722", "Temperature Sensor",
                ["Measures ambient/canopy temp (C)",
                 "Alerts on heat-stress thresholds",
                 "2 sensors: ID_01 (Zone A), ID_02 (Zone D)"])
y = draw_sensor(y, "●", "#1E88E5", "Humidity Sensor",
                ["Monitors relative humidity (%)",
                 "Informs irrigation scheduling",
                 "2 sensors: ID_03 (Zone B), ID_04 (Zone C)"])

y -= H_BETWEEN_SECTS * SCALE
y = draw_header(y, "◈  CROP HEALTH INDEX")
for row in HEALTH_ROWS:
    y = draw_health(y, *row)

# ── Status bar ────────────────────────────────────────────────────
status_txt = fig.text(
    0.375, 0.012, poll_status[0],
    ha="center", fontsize=8, color="white", fontfamily=MONO
)

# ─────────────────────────────────────────────────────────────────
#  SAVE TO FILE (non-interactive)
# ─────────────────────────────────────────────────────────────────
output_dir = os.path.dirname(os.path.abspath(__file__))
timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"crop_field_ditto_{timestamp}.png")

plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"[Saved] {output_path}")
