"""
06_interactive_predict.py — Interactive terminal REPL for crop-yield prediction
via the MQTT / Eclipse Ditto pipeline.

You type the input parameters at the prompt; the script:
  1. Publishes them as a SensorReading on the inbound MQTT topic
       sensors/{sensor_id}/telemetry
  2. An ML handler receives the message, calls the per-crop Random Forest
     directly (no seasonal aggregation — one input == one prediction),
     and publishes a Ditto-formatted PATCH on the outbound topic
       things/my.sensors:{sensor_id}/commands/modify
  3. The script subscribes to that outbound topic and prints the predicted
     yield to the terminal.

Modes
-----
  python 06_interactive_predict.py --mock
      Runs everything in-process with the MockBroker from script 05.
      No external services needed; useful for development.

  python 06_interactive_predict.py --broker localhost --port 1883
      Connects to a real MQTT broker (e.g., Mosquitto bridged to Eclipse
      Ditto via Ditto's "connections" feature). Requires paho-mqtt.

Eclipse Ditto wiring
--------------------
  The outbound payload is a Ditto-compatible feature update for the Thing
  "my.sensors:{sensor_id}". To make Ditto consume it, configure a Ditto
  MQTT connection that maps `things/+/commands/modify` to the Ditto
  `commands/modify` channel. See Ditto docs:
  https://www.eclipse.dev/ditto/connectivity-mqtt.html

Dependencies
------------
  pip install paho-mqtt --break-system-packages   # only for non-mock mode
"""

import argparse
import json
import sys
import threading
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

from config import MODELS_DIR, MQTT_TOPIC_TEMPLATE
from per_crop_models import predict_yield, load_models

# Reuse the dataclasses and MockBroker we already defined in script 05
from mqtt_dt_simulator import SensorReading, YieldPrediction, MockBroker


# ── Inbound message handler (the "ML service") ───────────────────────────────

class ImmediatePredictor:
    """
    Variant of MLService that predicts on every reading (no aggregation).
    Used for interactive single-shot inference.
    """

    def __init__(self, crop_models: dict, encoders: dict, publish_fn):
        self.crop_models = crop_models
        self.encoders    = encoders
        self.publish     = publish_fn

    def on_message(self, topic: str, payload: str) -> None:
        try:
            data    = json.loads(payload)
            reading = SensorReading(**data)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  [Predictor] Malformed message on {topic}: {e}")
            return

        try:
            yield_hg_ha = predict_yield(
                crop_name         = reading.crop,
                area              = reading.area,
                year              = reading.year,
                avg_temp          = reading.avg_temp,
                rainfall_mm       = reading.rainfall_mm,
                pesticides_tonnes = reading.pesticides,
                crop_models       = self.crop_models,
                encoders          = self.encoders,
            )
        except ValueError as e:
            print(f"  [Predictor] {e}")
            return

        prediction = YieldPrediction(
            sensor_id       = reading.sensor_id,
            crop            = reading.crop,
            predicted_yield = round(yield_hg_ha, 1),
            n_days          = 1,
        )
        out_topic = f"things/my.sensors:{reading.sensor_id}/commands/modify"
        self.publish(out_topic, json.dumps(prediction.to_ditto_patch()))


# ── Result listener (subscribes to the Ditto-formatted outbound topic) ───────

class ResultListener:
    """Prints predictions as they come back from the ML service."""

    def __init__(self):
        self.last_event = threading.Event()
        self.last_payload = None

    def on_message(self, topic: str, payload: str) -> None:
        try:
            data = json.loads(payload)
            props = data["features"]["predicted_yield"]["properties"]
        except (json.JSONDecodeError, KeyError):
            print(f"  [Listener] Unexpected payload on {topic}")
            return

        thing_id = data.get("thingId", "?")
        value    = props["value"]
        unit     = props.get("unit", "hg/ha")
        print(f"\n  ★ Ditto PATCH received")
        print(f"      topic   : {topic}")
        print(f"      thingId : {thing_id}")
        print(f"      yield   : {value:,.1f} {unit}\n")

        self.last_payload = data
        self.last_event.set()


# ── Terminal input ────────────────────────────────────────────────────────────

def prompt(label: str, default=None, cast=str):
    """Prompt for one value, with optional default and type cast."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"  {label}{suffix}: ").strip()
        if raw == "" and default is not None:
            return default
        if raw == "":
            print("    (required)")
            continue
        try:
            return cast(raw)
        except ValueError:
            print(f"    not a valid {cast.__name__}, try again")


def collect_reading(available_crops: list[str]) -> SensorReading | None:
    """
    Prompt the user for one set of parameters. Returns None if the user
    types 'q' / 'quit' to exit.
    """
    print("\n" + "─" * 60)
    print("  New prediction request — enter parameters (or 'q' to quit)")
    print("─" * 60)

    raw = input("  sensor_id [sensor01]: ").strip().lower()
    if raw in ("q", "quit", "exit"):
        return None
    sensor_id = raw or "sensor01"

    print(f"  available crops: {', '.join(available_crops)}")
    crop = prompt("crop", default="Wheat")
    if crop not in available_crops:
        print(f"  WARN: '{crop}' is not in the trained models. "
              f"Pick one of: {available_crops}")
        return collect_reading(available_crops)

    area = prompt("area (country)", default="Italy")
    year = prompt("year", default=2023, cast=int)
    avg_temp    = prompt("avg_temp (°C)", default=15.0, cast=float)
    rainfall_mm = prompt("rainfall_mm",    default=800.0, cast=float)
    pesticides  = prompt("pesticides (tonnes)", default=12000.0, cast=float)

    return SensorReading(
        sensor_id   = sensor_id,
        crop        = crop,
        area        = area,
        year        = year,
        avg_temp    = avg_temp,
        rainfall_mm = rainfall_mm,
        pesticides  = pesticides,
        timestamp   = datetime.now(timezone.utc).isoformat(),
    )


# ── Mock-mode runner (in-process) ────────────────────────────────────────────

def run_mock(crop_models, encoders, available_crops):
    broker = MockBroker()

    listener = ResultListener()
    predictor = ImmediatePredictor(
        crop_models, encoders, publish_fn=broker.publish
    )

    # ML service subscribes to ALL sensor topics (wildcard not supported in
    # MockBroker — we subscribe per known sensor_id when first seen).
    subscribed_sensors = set()

    def ensure_subscriptions(sensor_id: str):
        if sensor_id in subscribed_sensors:
            return
        in_topic  = MQTT_TOPIC_TEMPLATE.format(sensor_id=sensor_id)
        out_topic = f"things/my.sensors:{sensor_id}/commands/modify"
        broker.subscribe(in_topic,  predictor.on_message)
        broker.subscribe(out_topic, listener.on_message)
        subscribed_sensors.add(sensor_id)

    print("\n[Mock mode] No external broker. All messages stay in-process.\n"
          "Type 'q' at the sensor_id prompt to exit.")

    while True:
        reading = collect_reading(available_crops)
        if reading is None:
            print("\nExiting.")
            return

        ensure_subscriptions(reading.sensor_id)
        topic = MQTT_TOPIC_TEMPLATE.format(sensor_id=reading.sensor_id)
        listener.last_event.clear()
        broker.publish(topic, reading.to_mqtt_payload())

        # MockBroker is synchronous, so the prediction has already arrived
        # by the time publish() returns. The listener.on_message printed it.
        if not listener.last_event.is_set():
            print("  (no prediction returned)")


# ── Real-broker runner (paho-mqtt) ───────────────────────────────────────────

def run_real(host: str, port: int, crop_models, encoders, available_crops):
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        sys.exit("Install paho-mqtt:  pip install paho-mqtt "
                 "--break-system-packages")

    listener  = ResultListener()
    predictor = ImmediatePredictor(crop_models, encoders, publish_fn=None)

    def on_connect(client, userdata, flags, rc):
        print(f"[MQTT] Connected to {host}:{port} (rc={rc})")
        client.subscribe("sensors/+/telemetry")
        client.subscribe("things/my.sensors:+/commands/modify")

    def on_message(client, userdata, msg):
        topic   = msg.topic
        payload = msg.payload.decode()
        if topic.startswith("sensors/"):
            predictor.on_message(topic, payload)
        elif topic.startswith("things/"):
            listener.on_message(topic, payload)

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(host, port, keepalive=60)

    # Inject the publish function now that the client exists
    predictor.publish = lambda t, p: client.publish(t, p)

    # paho's network loop runs in a background thread so we can do input()
    client.loop_start()

    print(f"\n[Real mode] Connected to {host}:{port}.\n"
          "If you have Eclipse Ditto bridged to this broker, predictions\n"
          "will appear both here and in Ditto as 'predicted_yield' feature\n"
          "updates on Things 'my.sensors:<id>'.")

    try:
        while True:
            reading = collect_reading(available_crops)
            if reading is None:
                break
            topic = MQTT_TOPIC_TEMPLATE.format(sensor_id=reading.sensor_id)
            listener.last_event.clear()
            client.publish(topic, reading.to_mqtt_payload())

            # Wait up to 5 s for the round-trip
            if not listener.last_event.wait(timeout=5):
                print("  (no response within 5 s — is the ML service running?)")
    finally:
        client.loop_stop()
        client.disconnect()
        print("\nDisconnected.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive crop-yield prediction over MQTT/Ditto"
    )
    parser.add_argument("--mock",   action="store_true",
                        help="Use in-process mock broker (default if no host)")
    parser.add_argument("--broker", default=None,
                        help="Real MQTT broker hostname")
    parser.add_argument("--port",   type=int, default=1883,
                        help="Real MQTT broker port (default: 1883)")
    args = parser.parse_args()

    print("\n[06] Loading per-crop models …")
    try:
        crop_models, encoders = load_models()
        print(f"  Loaded {len(crop_models)} crop models from '{MODELS_DIR}/'")
    except FileNotFoundError:
        sys.exit("  No models found. Run 03_per_crop_models.py first.")

    available_crops = sorted(crop_models.keys())

    if args.broker is None or args.mock:
        run_mock(crop_models, encoders, available_crops)
    else:
        run_real(args.broker, args.port, crop_models, encoders, available_crops)


if __name__ == "__main__":
    main()
