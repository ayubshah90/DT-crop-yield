"""
05_mqtt_dt_simulator.py — End-to-end MQTT ↔ Eclipse Ditto ↔ ML pipeline.

What this script demonstrates
------------------------------
  1. Four simulated sensors publish temperature and humidity readings to an
     MQTT broker (using the paho-mqtt library or a local mock).
  2. A message handler (the "ML service") subscribes to sensor topics,
     accumulates readings via SeasonalAggregator, and invokes the per-crop
     Random Forest when a seasonal window is complete.
  3. The prediction is published back to a Ditto-compatible topic so Eclipse
     Ditto can update the predicted_yield feature of the corresponding Thing.

Eclipse Ditto topic convention used
--------------------------------------
  Inbound  (sensor → Ditto):
    sensors/{sensor_id}/telemetry
  Outbound (ML → Ditto PATCH):
    things/my.sensors:{sensor_id}/commands/modify
    (payload: JSON with featurePath = predicted_yield)

Running modes
-------------
  python 05_mqtt_dt_simulator.py --mock
      Run fully offline with a mock MQTT broker (no external service needed).
      Suitable for CI and development. Prints all messages to stdout.

  python 05_mqtt_dt_simulator.py --broker localhost --port 1883
      Connect to a real MQTT broker (e.g., Mosquitto running locally or the
      Eclipse Ditto MQTT adapter endpoint).

Dependencies
------------
  pip install paho-mqtt --break-system-packages
"""

import argparse
import json
import pickle
import os
import time
import threading
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np

warnings.filterwarnings("ignore")

from config import (
    MODELS_DIR, MQTT_TOPIC_TEMPLATE, SEASON_WINDOW_DAYS, SENSOR_NOISE,
)
from dt_validation import SeasonalAggregator        # seasonal window logic
from per_crop_models import predict_yield, load_models  # inference


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class SensorReading:
    """One MQTT message from a field sensor."""
    sensor_id:   str
    crop:        str     # e.g. "Wheat" — must match model keys
    area:        str     # country
    year:        int
    avg_temp:    float   # °C
    rainfall_mm: float   # mm
    pesticides:  float   # tonnes
    timestamp:   str     = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_mqtt_payload(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class YieldPrediction:
    """Outbound MQTT message carrying a yield prediction back to Ditto."""
    sensor_id:     str
    crop:          str
    predicted_yield: float   # hg/ha
    n_days:        int       # how many days were aggregated
    timestamp:     str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_ditto_patch(self) -> dict:
        """
        Format the prediction as an Eclipse Ditto feature PATCH payload.
        POST to:  GET /api/2/things/my.sensors:{sensor_id}/features/predicted_yield
        """
        return {
            "thingId": f"my.sensors:{self.sensor_id}",
            "features": {
                "predicted_yield": {
                    "properties": {
                        "value": self.predicted_yield,
                        "unit":  "hg/ha",
                        "n_days_aggregated": self.n_days,
                        "timestamp": self.timestamp,
                    }
                }
            }
        }


# ── ML service (subscriber side) ─────────────────────────────────────────────

class MLService:
    """
    Subscribes to sensor MQTT topics.
    Accumulates readings per sensor, predicts yield seasonally.
    Publishes predictions back to Ditto-formatted topics.
    """

    def __init__(
        self,
        crop_models: dict,
        encoders: dict,
        publish_fn,   # callable(topic: str, payload: str) — injected
        window_days: int = SEASON_WINDOW_DAYS,
    ):
        self.crop_models = crop_models
        self.encoders    = encoders
        self.publish     = publish_fn
        self.aggregators: dict[str, SeasonalAggregator] = {}
        self.window_days = window_days
        self.prediction_log: list[YieldPrediction] = []

    def _get_aggregator(self, sensor_id: str) -> SeasonalAggregator:
        if sensor_id not in self.aggregators:
            self.aggregators[sensor_id] = SeasonalAggregator(self.window_days)
        return self.aggregators[sensor_id]

    def on_message(self, topic: str, payload: str) -> None:
        """
        Handle one inbound MQTT message.
        Call this from the paho on_message callback.
        """
        try:
            data = json.loads(payload)
            reading = SensorReading(**data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  [MLService] Malformed message on {topic}: {e}")
            return

        agg = self._get_aggregator(reading.sensor_id)
        result = agg.ingest({
            "avg_temp":                      reading.avg_temp,
            "average_rain_fall_mm_per_year": reading.rainfall_mm,
            "pesticides_tonnes":             reading.pesticides,
        })

        if result is None:
            # Window not yet full
            print(f"  [MLService] {reading.sensor_id} day {agg.days_accumulated:>3}"
                  f"/{self.window_days}  temp={reading.avg_temp:.1f}°C")
            return

        # Seasonal window complete → predict
        try:
            yield_pred = predict_yield(
                crop_name         = reading.crop,
                area              = reading.area,
                year              = reading.year,
                avg_temp          = result["avg_temp"],
                rainfall_mm       = result["average_rain_fall_mm_per_year"],
                pesticides_tonnes = result["pesticides_tonnes"],
                crop_models       = self.crop_models,
                encoders          = self.encoders,
            )
        except ValueError as e:
            print(f"  [MLService] Prediction error: {e}")
            agg.reset()
            return

        prediction = YieldPrediction(
            sensor_id      = reading.sensor_id,
            crop           = reading.crop,
            predicted_yield= round(yield_pred, 1),
            n_days         = result["n_days"],
        )
        self.prediction_log.append(prediction)

        # Publish result (back to Ditto via MQTT)
        out_topic = f"things/my.sensors:{reading.sensor_id}/commands/modify"
        self.publish(out_topic, json.dumps(prediction.to_ditto_patch()))

        print(f"\n  ★ [MLService] Prediction for {reading.sensor_id} ({reading.crop})"
              f"\n      seasonal avg_temp  = {result['avg_temp']:.2f} °C"
              f"\n      seasonal rainfall  = {result['average_rain_fall_mm_per_year']:.1f} mm"
              f"\n      predicted yield    = {yield_pred:,.0f} hg/ha"
              f"\n      Ditto PATCH        → {out_topic}\n")

        agg.reset()


# ── Mock MQTT broker ──────────────────────────────────────────────────────────

class MockBroker:
    """
    In-memory pub/sub broker — no network needed.
    Subscribers register via subscribe(topic, callback).
    """

    def __init__(self):
        self._subscribers: dict[str, list] = {}
        self._lock = threading.Lock()

    def subscribe(self, topic: str, callback):
        with self._lock:
            self._subscribers.setdefault(topic, []).append(callback)

    def publish(self, topic: str, payload: str):
        with self._lock:
            callbacks = list(self._subscribers.get(topic, []))
        for cb in callbacks:
            cb(topic, payload)


# ── Sensor simulator ──────────────────────────────────────────────────────────

class SensorSimulator:
    """
    Simulates MQTT messages from four field sensors over one growing season.
    Publishes one reading per sensor per day (condensed to ~10 ms intervals
    so the demo runs quickly).
    """

    SENSOR_CONFIG = [
        {"sensor_id": "sensor01", "crop": "Wheat",       "area": "Italy", "year": 2023,
         "base_temp": 14.5, "base_rain": 780,  "base_pest": 12000},
        {"sensor_id": "sensor02", "crop": "Maize",       "area": "Italy", "year": 2023,
         "base_temp": 18.2, "base_rain": 820,  "base_pest": 55000},
        {"sensor_id": "sensor03", "crop": "Potatoes",    "area": "Italy", "year": 2023,
         "base_temp": 12.0, "base_rain": 700,  "base_pest": 90000},
        {"sensor_id": "sensor04", "crop": "Rice, paddy", "area": "Italy", "year": 2023,
         "base_temp": 19.5, "base_rain": 1100, "base_pest": 30000},
    ]

    def __init__(self, broker, days: int = SEASON_WINDOW_DAYS + 10, seed: int = 42):
        self.broker = broker
        self.days   = days
        self.rng    = np.random.default_rng(seed)

    def _make_reading(self, cfg: dict, day: int) -> SensorReading:
        return SensorReading(
            sensor_id   = cfg["sensor_id"],
            crop        = cfg["crop"],
            area        = cfg["area"],
            year        = cfg["year"],
            avg_temp    = cfg["base_temp"]   + self.rng.normal(0, SENSOR_NOISE["avg_temp"]),
            rainfall_mm = cfg["base_rain"]   + self.rng.normal(0, SENSOR_NOISE["average_rain_fall_mm_per_year"]),
            pesticides  = cfg["base_pest"]   * (1 + self.rng.normal(0, SENSOR_NOISE["pesticides_tonnes"])),
        )

    def run(self, speed: float = 0.01) -> None:
        """
        Publish sensor readings for `self.days` days.

        Parameters
        ----------
        speed : seconds to sleep between days (0.01 = fast demo, 86400 = real-time)
        """
        print(f"\n[Simulator] Streaming {self.days} days × "
              f"{len(self.SENSOR_CONFIG)} sensors …\n")
        for day in range(1, self.days + 1):
            for cfg in self.SENSOR_CONFIG:
                reading = self._make_reading(cfg, day)
                topic   = MQTT_TOPIC_TEMPLATE.format(sensor_id=cfg["sensor_id"])
                self.broker.publish(topic, reading.to_mqtt_payload())
            time.sleep(speed)

        print("[Simulator] Stream complete.")


# ── Real MQTT runner (paho-mqtt) ──────────────────────────────────────────────

def run_with_real_broker(host: str, port: int, ml_service: MLService) -> None:
    """
    Connect to a real MQTT broker (e.g., Mosquitto or Eclipse Ditto's adapter).
    Requires: pip install paho-mqtt
    """
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        raise ImportError("Install paho-mqtt:  pip install paho-mqtt --break-system-packages")

    def on_connect(client, userdata, flags, rc):
        print(f"[MQTT] Connected to {host}:{port} (rc={rc})")
        client.subscribe("sensors/+/telemetry")

    def on_message(client, userdata, msg):
        ml_service.on_message(msg.topic, msg.payload.decode())

    def publish_fn(topic: str, payload: str):
        client.publish(topic, payload)

    ml_service.publish = publish_fn

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, keepalive=60)
    client.loop_forever()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DT crop-yield MQTT simulator")
    parser.add_argument("--mock",   action="store_true", default=True,
                        help="Use mock broker (default; no external service needed)")
    parser.add_argument("--broker", default="localhost",
                        help="Real MQTT broker hostname")
    parser.add_argument("--port",   type=int, default=1883,
                        help="Real MQTT broker port")
    parser.add_argument("--days",   type=int, default=SEASON_WINDOW_DAYS + 10,
                        help="Number of simulated days to stream")
    args = parser.parse_args()

    print("\n[05] Loading per-crop models …")
    try:
        crop_models, encoders = load_models()
        print(f"  Loaded {len(crop_models)} crop models from '{MODELS_DIR}/'")
    except FileNotFoundError:
        print("  Models not found — training now (run 03_per_crop_models.py first)")
        from per_crop_models import run as train_crops
        crop_models, encoders = train_crops()

    # ── Mock mode ─────────────────────────────────────────────────────────────
    if args.mock:
        broker = MockBroker()

        def publish_fn(topic: str, payload: str):
            # Echo Ditto PATCH back to console (would go to Ditto in production)
            data = json.loads(payload)
            print(f"  [Ditto PATCH] thingId={data['thingId']}  "
                  f"yield={data['features']['predicted_yield']['properties']['value']:,.0f} hg/ha")

        ml_service = MLService(crop_models, encoders, publish_fn)

        # Subscribe the ML service to all sensor topics
        for cfg in SensorSimulator.SENSOR_CONFIG:
            topic = MQTT_TOPIC_TEMPLATE.format(sensor_id=cfg["sensor_id"])
            broker.subscribe(topic, ml_service.on_message)

        simulator = SensorSimulator(broker, days=args.days)
        simulator.run(speed=0.005)  # fast demo

        # Summary
        print("\n── Prediction log ───────────────────────────────────────")
        for p in ml_service.prediction_log:
            print(f"  {p.sensor_id}  {p.crop:<18}"
                  f"  yield={p.predicted_yield:>10,.1f} hg/ha"
                  f"  (agg over {p.n_days} days)")

    # ── Real broker mode ──────────────────────────────────────────────────────
    else:
        ml_service = MLService(crop_models, encoders, publish_fn=lambda t, p: None)
        run_with_real_broker(args.broker, args.port, ml_service)

    print("\n[05] MQTT simulation complete.\n")


if __name__ == "__main__":
    main()
