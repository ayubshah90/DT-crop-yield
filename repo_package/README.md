# Digital Twin-Driven Crop-Yield Prediction

Code and artefacts accompanying:

> A. Shah, R. Ntassah, F. Granelli, "A Digital Twin-Driven Framework with
> Machine Learning Integration for Crop-Yield Prediction," SoftCOM 2026.

## Repository structure

```
.
├── data/
│   └── yield_df.csv                  FAO/World Bank crop-yield dataset extract
├── models/
│   ├── <crop>.pkl                    Trained per-crop Random Forest models
│   └── encoders.pkl                  LabelEncoder objects for Area / Item fields
├── results/
│   └── RF_predictions_all_crops/     RF inference logs for the 2026-2030
│                                     counterfactual scenario (per crop, per
│                                     sensor, per year)
├── src/
│   ├── config.py                     Shared configuration (feature lists,
│   │                                 RF hyperparameters, paths)
│   ├── preprocessing.py              Dataset loading and cleaning
│   ├── per_crop_models.py            Per-crop RF training and inference
│   ├── dt_validation.py              KS-test robustness experiment and
│   │                                 seasonal aggregation pipeline demo
│   ├── mqtt_dt_simulator.py /
│   │   05_mqtt_dt_simulator.py       MQTT-driven sensor telemetry simulator
│   ├── 06_interactive_predict.py /
│   │   06_whatif_engine.py           Interactive what-if scenario querying
│   └── plot_*.py                     Figure-generation scripts (read directly
│                                     from results/RF_predictions_all_crops)
└── docs/                             (reserved for supplementary material)
```

## Reproducing the paper's figures

The plotting scripts in `src/plot_*.py` read directly from the JSON
prediction logs in `results/RF_predictions_all_crops/`, so they reproduce
the exact published figures without re-running inference:

```bash
cd src
python plot_fig7_from_real_logs.py
python plot_input_trajectory_from_real_logs.py
python plot_spatial_agreement_from_real_logs.py
python plot_dt_roundtrip_revised.py
```

## Reproducing the trained models

The FAO/World Bank dataset extract (`data/yield_df.csv`) and the
preprocessing/training pipeline (`src/preprocessing.py`,
`src/per_crop_models.py`) are included to retrain the per-crop Random
Forest models from scratch if desired. Trained model artefacts are also
provided directly in `models/` for convenience.

## Requirements

See `requirements.txt`. Tested with Python 3.10+.

## Eclipse Ditto / MQTT configuration

The MQTT simulator (`src/mqtt_dt_simulator.py`) publishes synthetic sensor
telemetry for four virtual field sensors (`my.sensors:sensor01` …
`my.sensors:sensor04`) under a shared Eclipse Ditto policy
(`my.test:policy`), matching the configuration described in Section III-A
of the paper.

## Citation

If you use this code, please cite the SoftCOM 2026 paper (full citation to
be added upon publication).

## License

To be determined by the authors prior to making the repository public.
