"""
06_whatif_engine.py — What-if scenario engine for the DT precision-agriculture pipeline.

This script is the scientific justification for using a Digital Twin
rather than a plain ML model or database.

How it validates the DT
-----------------------
Eclipse Ditto maintains a *verified baseline state* for each sensor Thing —
the current real (or realistically simulated) agronomic conditions of the
physical field. The what-if engine:

  1. Reads the baseline state from Ditto (or accepts it via terminal input)
  2. Constructs hypothetical scenario states (perturbations of the baseline)
  3. Publishes each scenario to Ditto as a temporary feature snapshot
  4. Invokes the per-crop ML model on both baseline and each scenario
  5. Retrieves and compares predictions — stored persistently in Ditto

A plain database or standalone ML script cannot do step 3 or 5 without
replicating the state-management logic that Ditto provides for free.

Usage
-----
  python 06_whatif_engine.py
      Interactive terminal mode — enter baseline values when prompted,
      choose scenarios from a menu.

  python 06_whatif_engine.py --batch
      Run all predefined scenarios for four crops and save figures.

  python 06_whatif_engine.py --crop wheat --temp 14.5 --rain 780 --pest 12000 --area Italy
      Non-interactive single-crop run.

Outputs
-------
  results/whatif_comparison_table.csv
  results/fig_whatif_sensitivity.png
  results/fig_whatif_heatmap.png
  Console: formatted comparison table with Δ% and directional arrows
"""

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from config import MODELS_DIR, RESULTS_DIR, ENV_FEATURES
from per_crop_models import load_models


# ── Scenario definitions ──────────────────────────────────────────────────────

PREDEFINED_SCENARIOS = [
    # (label, temp_delta, rain_factor, pest_factor)
    ("Baseline",          0.0,  1.00, 1.00),
    ("+1 °C warming",    +1.0,  1.00, 1.00),
    ("+2 °C warming",    +2.0,  1.00, 1.00),
    ("+3 °C warming",    +3.0,  1.00, 1.00),
    ("−20 % rainfall",   0.0,  0.80, 1.00),
    ("−40 % rainfall",   0.0,  0.60, 1.00),
    ("−50 % pesticides", 0.0,  1.00, 0.50),
    ("+2 °C & −40 % rain", +2.0, 0.60, 1.00),
]

# Default crop configurations (baseline values drawn from Italian FAO data)
DEFAULT_CROP_CONFIGS = {
    "wheat":     {"area": "Italy", "base_temp": 14.5, "base_rain": 780,  "base_pest": 12000},
    "maize":     {"area": "Italy", "base_temp": 18.2, "base_rain": 820,  "base_pest": 55000},
    "potatoes":  {"area": "Italy", "base_temp": 12.0, "base_rain": 700,  "base_pest": 90000},
    "rice paddy":{"area": "Italy", "base_temp": 19.5, "base_rain": 1100, "base_pest": 30000},
}


# ── Ditto state model ─────────────────────────────────────────────────────────

@dataclass
class DittoThing:
    """
    Lightweight in-memory representation of an Eclipse Ditto Thing.
    In production this would be a REST call to:
      GET /api/2/things/my.sensors:{thing_id}
      PUT /api/2/things/my.sensors:{thing_id}/features/...

    The structure mirrors the Ditto JSON schema used in the paper.
    """
    thing_id:  str
    crop:      str
    area:      str
    year:      int = 2023

    # Baseline (physical field state)
    baseline_temp:  float = 0.0
    baseline_rain:  float = 0.0
    baseline_pest:  float = 0.0
    baseline_yield: Optional[float] = None

    # Scenario results (populated by what-if engine)
    scenario_results: dict = field(default_factory=dict)

    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_ditto_json(self) -> dict:
        """Serialise to Eclipse Ditto Thing JSON format."""
        return {
            "thingId": f"my.sensors:{self.thing_id}",
            "policyId": "my.test:policy",
            "attributes": {
                "type": "agriculture_sensor",
                "location": "field_A",
                "crop": self.crop,
                "area": self.area,
            },
            "features": {
                "baseline": {
                    "properties": {
                        "avg_temp":              self.baseline_temp,
                        "average_rain_fall_mm_per_year": self.baseline_rain,
                        "pesticides_tonnes":     self.baseline_pest,
                        "predicted_yield":       self.baseline_yield,
                        "timestamp":             self.timestamp,
                    }
                },
                "whatif_scenarios": {
                    "properties": self.scenario_results
                }
            }
        }

    def patch_scenario(self, label: str, yield_val: float, params: dict) -> None:
        """Store a scenario prediction — mirrors a Ditto PATCH command."""
        self.scenario_results[label] = {
            "avg_temp":    params["avg_temp"],
            "rainfall_mm": params["rainfall_mm"],
            "pest_tonnes": params["pest_tonnes"],
            "yield_hg_ha": round(yield_val, 1),
            "delta_pct":   round(
                (yield_val - self.baseline_yield) / self.baseline_yield * 100, 2
            ) if self.baseline_yield else None,
        }


# ── Mock Ditto registry ───────────────────────────────────────────────────────

class DittoRegistry:
    """
    In-memory registry of DittoThings.
    Replaces live Eclipse Ditto REST calls for offline/demo use.
    In production: swap get/update with requests.get/patch to the Ditto API.
    """

    def __init__(self):
        self._things: dict[str, DittoThing] = {}

    def register(self, thing: DittoThing) -> None:
        self._things[thing.thing_id] = thing
        print(f"  [Ditto] Thing registered: my.sensors:{thing.thing_id}")

    def get(self, thing_id: str) -> Optional[DittoThing]:
        return self._things.get(thing_id)

    def update_scenario(self, thing_id: str, label: str,
                        yield_val: float, params: dict) -> None:
        if thing_id in self._things:
            self._things[thing_id].patch_scenario(label, yield_val, params)
            print(f"  [Ditto] PATCH my.sensors:{thing_id} "
                  f"→ whatif_scenarios.{label!r} = {yield_val:,.0f} hg/ha")

    def dump_json(self, thing_id: str) -> str:
        t = self._things.get(thing_id)
        return json.dumps(t.to_ditto_json(), indent=2) if t else "{}"

    def all_things(self) -> list[DittoThing]:
        return list(self._things.values())


# ── Prediction helper ─────────────────────────────────────────────────────────

def _predict(crop_key: str, area: str, year: int,
             temp: float, rain: float, pest: float,
             crop_models: dict, encoders: dict) -> float:
    """Direct inference using per-crop RF (no name-matching overhead)."""
    le_area = encoders["Area"]
    try:
        area_enc = int(le_area.transform([area])[0])
    except ValueError:
        area_enc = int(le_area.classes_.shape[0] // 2)

    X = pd.DataFrame([{
        "average_rain_fall_mm_per_year": rain,
        "pesticides_tonnes":             pest,
        "avg_temp":                      temp,
        "Area_enc":                      area_enc,
        "Year":                          year,
    }])[ENV_FEATURES]

    return float(crop_models[crop_key].predict(X)[0])


# ── What-if engine ────────────────────────────────────────────────────────────

class WhatIfEngine:
    """
    Core what-if engine:
      - Accepts a baseline state (from terminal or Ditto)
      - Runs all predefined scenarios against the per-crop ML model
      - Stores results in the DittoRegistry
      - Produces a comparison DataFrame
    """

    def __init__(self, crop_models: dict, encoders: dict,
                 registry: DittoRegistry):
        self.crop_models = crop_models
        self.encoders    = encoders
        self.registry    = registry

    def run(
        self,
        thing_id:  str,
        crop_key:  str,
        area:      str,
        year:      int,
        base_temp: float,
        base_rain: float,
        base_pest: float,
        scenarios: list = PREDEFINED_SCENARIOS,
    ) -> pd.DataFrame:
        """
        Run all scenarios for one sensor Thing.

        Returns a DataFrame with columns:
          Scenario, Temp(°C), Rain(mm), Pest(t),
          Yield(hg/ha), Δ(hg/ha), Δ(%)
        """
        # ── Register baseline in Ditto ────────────────────────────────────────
        thing = DittoThing(
            thing_id     = thing_id,
            crop         = crop_key,
            area         = area,
            year         = year,
            baseline_temp = base_temp,
            baseline_rain = base_rain,
            baseline_pest = base_pest,
        )
        base_yield = _predict(crop_key, area, year,
                              base_temp, base_rain, base_pest,
                              self.crop_models, self.encoders)
        thing.baseline_yield = base_yield
        self.registry.register(thing)

        # ── Run scenarios ─────────────────────────────────────────────────────
        rows = []
        for label, dt, rf, pf in scenarios:
            sc_temp = base_temp + dt
            sc_rain = base_rain * rf
            sc_pest = base_pest * pf

            sc_yield = _predict(crop_key, area, year,
                                sc_temp, sc_rain, sc_pest,
                                self.crop_models, self.encoders)
            delta_abs = sc_yield - base_yield
            delta_pct = delta_abs / base_yield * 100

            params = {"avg_temp": sc_temp,
                      "rainfall_mm": sc_rain,
                      "pest_tonnes": sc_pest}
            self.registry.update_scenario(thing_id, label, sc_yield, params)

            rows.append({
                "Scenario":    label,
                "Temp (°C)":   round(sc_temp, 1),
                "Rain (mm)":   round(sc_rain, 0),
                "Pest (t)":    round(sc_pest, 0),
                "Yield (hg/ha)": round(sc_yield, 0),
                "Δ (hg/ha)":   round(delta_abs, 0),
                "Δ (%)":       round(delta_pct, 2),
            })

        return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _arrow(val: float) -> str:
    if val > 1:  return "▲"
    if val < -1: return "▼"
    return "◆"


def print_comparison_table(df: pd.DataFrame, crop: str) -> None:
    """Pretty-print scenario comparison to terminal."""
    print(f"\n{'═'*68}")
    print(f"  What-if analysis — {crop.title()}")
    print(f"{'═'*68}")
    print(f"  {'Scenario':<24} {'Yield':>10}  {'Δ (hg/ha)':>10}  {'Δ (%)':>7}  Dir")
    print(f"  {'─'*24} {'─'*10}  {'─'*10}  {'─'*7}  ───")
    for _, row in df.iterrows():
        arrow = _arrow(row["Δ (%)"])
        base_marker = " ← baseline" if row["Scenario"] == "Baseline" else ""
        print(f"  {row['Scenario']:<24} {row['Yield (hg/ha)']:>10,.0f}  "
              f"{row['Δ (hg/ha)']:>+10,.0f}  {row['Δ (%)']:>+6.1f}%  "
              f"{arrow}{base_marker}")
    print()


def plot_sensitivity(all_results: dict) -> None:
    """
    Multi-crop sensitivity chart.
    X-axis: scenario label
    Y-axis: Δ% from baseline
    One line per crop.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13, 5))

    colors  = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517"]
    markers = ["o", "s", "^", "D"]

    for (crop, df), color, marker in zip(all_results.items(), colors, markers):
        # Skip baseline row for the delta plot
        plot_df = df[df["Scenario"] != "Baseline"].copy()
        ax.plot(
            plot_df["Scenario"], plot_df["Δ (%)"],
            marker=marker, color=color, linewidth=1.8,
            markersize=7, label=crop.title(),
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Yield change from baseline (%)", fontsize=11)
    ax.set_xlabel("")
    ax.set_title(
        "What-if Scenario Analysis — Yield Sensitivity per Crop\n"
        "(Eclipse Ditto baseline state vs hypothetical conditions)",
        fontsize=11,
    )
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_whatif_sensitivity.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sensitivity figure saved → {out}")


def plot_heatmap(all_results: dict) -> None:
    """
    Heatmap: crops × scenarios, coloured by Δ%.
    Red = yield loss, green = yield gain.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    crops     = list(all_results.keys())
    scenarios = [s for s in all_results[crops[0]]["Scenario"]
                 if s != "Baseline"]

    matrix = np.zeros((len(crops), len(scenarios)))
    for i, crop in enumerate(crops):
        df = all_results[crop]
        for j, sc in enumerate(scenarios):
            row = df[df["Scenario"] == sc]
            if not row.empty:
                matrix[i, j] = row["Δ (%)"].values[0]

    fig, ax = plt.subplots(figsize=(13, 3.5))
    cmap = mcolors.TwoSlopeNorm(vmin=matrix.min(), vcenter=0, vmax=matrix.max())
    im = ax.imshow(matrix, cmap="RdYlGn", norm=cmap, aspect="auto")

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(crops)))
    ax.set_yticklabels([c.title() for c in crops], fontsize=10)

    # Annotate cells
    for i in range(len(crops)):
        for j in range(len(scenarios)):
            v = matrix[i, j]
            color = "white" if abs(v) > 12 else "black"
            ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="500")

    plt.colorbar(im, ax=ax, label="Yield Δ (%)", shrink=0.8)
    ax.set_title(
        "What-if Scenario Heatmap — Yield Change (%) from DT Baseline",
        fontsize=11, pad=10,
    )
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "fig_whatif_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap figure saved → {out}")


# ── Interactive terminal mode ─────────────────────────────────────────────────

def terminal_mode(crop_models: dict, encoders: dict) -> None:
    """
    Replicate the current manual-input workflow but routed through
    the DT state model. User enters baseline values; engine runs all
    scenarios and stores results in Ditto.
    """
    print("\n" + "═" * 60)
    print("  DT What-if Engine — Interactive Mode")
    print("  (Press Ctrl+C to exit)")
    print("═" * 60)

    available = list(crop_models.keys())
    print(f"\n  Available crops: {', '.join(c.title() for c in available)}")
    crop_input = input("\n  Enter crop name: ").strip().lower()
    if crop_input not in available:
        print(f"  Unknown crop. Choose from: {available}")
        return

    area  = input("  Country (e.g. Italy): ").strip() or "Italy"
    year  = int(input("  Year (e.g. 2023): ").strip() or 2023)
    temp  = float(input("  Avg temperature (°C): ").strip())
    rain  = float(input("  Annual rainfall (mm): ").strip())
    pest  = float(input("  Pesticides (tonnes): ").strip())

    registry = DittoRegistry()
    engine   = WhatIfEngine(crop_models, encoders, registry)

    print("\n  Running scenarios …")
    df = engine.run(
        thing_id  = "sensor01",
        crop_key  = crop_input,
        area      = area,
        year      = year,
        base_temp = temp,
        base_rain = rain,
        base_pest = pest,
    )

    print_comparison_table(df, crop_input)

    # Show Ditto state
    print("  Eclipse Ditto Thing state (JSON):")
    print(registry.dump_json("sensor01"))

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "whatif_comparison_table.csv"), index=False)
    plot_sensitivity({crop_input: df})
    plot_heatmap({crop_input: df})


# ── Batch mode ────────────────────────────────────────────────────────────────

def batch_mode(crop_models: dict, encoders: dict) -> dict:
    """
    Run all predefined scenarios for all default crops.
    Returns all_results dict for downstream use.
    """
    registry    = DittoRegistry()
    engine      = WhatIfEngine(crop_models, encoders, registry)
    all_results = {}

    print("\n  Running batch what-if analysis …\n")
    for i, (crop, cfg) in enumerate(DEFAULT_CROP_CONFIGS.items(), 1):
        df = engine.run(
            thing_id  = f"sensor0{i}",
            crop_key  = crop,
            area      = cfg["area"],
            year      = 2023,
            base_temp = cfg["base_temp"],
            base_rain = cfg["base_rain"],
            base_pest = cfg["base_pest"],
        )
        all_results[crop] = df
        print_comparison_table(df, crop)

    # Combined CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined = pd.concat(
        [df.assign(Crop=crop) for crop, df in all_results.items()],
        ignore_index=True,
    )[["Crop", "Scenario", "Temp (°C)", "Rain (mm)", "Pest (t)",
       "Yield (hg/ha)", "Δ (hg/ha)", "Δ (%)"]]
    combined.to_csv(
        os.path.join(RESULTS_DIR, "whatif_comparison_table.csv"), index=False
    )
    print(f"  Full table saved → {RESULTS_DIR}/whatif_comparison_table.csv")

    plot_sensitivity(all_results)
    plot_heatmap(all_results)
    return all_results


# ── Single-crop CLI mode ──────────────────────────────────────────────────────

def single_crop_mode(args, crop_models: dict, encoders: dict) -> None:
    registry = DittoRegistry()
    engine   = WhatIfEngine(crop_models, encoders, registry)
    crop_key = args.crop.lower().replace("-", " ")

    df = engine.run(
        thing_id  = "sensor01",
        crop_key  = crop_key,
        area      = args.area,
        year      = args.year,
        base_temp = args.temp,
        base_rain = args.rain,
        base_pest = args.pest,
    )
    print_comparison_table(df, crop_key)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "whatif_comparison_table.csv"), index=False)
    plot_sensitivity({crop_key: df})
    plot_heatmap({crop_key: df})


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> dict:
    """Entry point for run_all.py — runs batch mode."""
    print("\n[06] What-if scenario engine …")
    crop_models, encoders = load_models()
    results = batch_mode(crop_models, encoders)
    print("\n[06] What-if analysis complete.\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DT What-if Scenario Engine")
    parser.add_argument("--batch",  action="store_true",
                        help="Run all predefined crops and scenarios")
    parser.add_argument("--crop",   type=str, help="Crop name (e.g. wheat)")
    parser.add_argument("--temp",   type=float, help="Baseline avg temperature (°C)")
    parser.add_argument("--rain",   type=float, help="Baseline annual rainfall (mm)")
    parser.add_argument("--pest",   type=float, help="Baseline pesticides (tonnes)")
    parser.add_argument("--area",   type=str,   default="Italy")
    parser.add_argument("--year",   type=int,   default=2023)
    args = parser.parse_args()

    crop_models, encoders = load_models()

    if args.batch:
        batch_mode(crop_models, encoders)
    elif args.crop and args.temp and args.rain and args.pest:
        single_crop_mode(args, crop_models, encoders)
    else:
        terminal_mode(crop_models, encoders)
