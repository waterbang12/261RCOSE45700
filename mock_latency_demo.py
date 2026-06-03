import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from arduino_actuator import ArduinoActuator
from latency_profiler import LatencyProfiler

DEFAULT_ARDUINO_PORT = "COM3"
DEFAULT_ARDUINO_BAUD = 9600
DEFAULT_EVENT_INTERVAL_SEC = 3.0
DEFAULT_CSV_PATH = "latency_results.csv"
DEFAULT_JSON_PATH = "latency_results.json"
DEFAULT_AGENT_MODEL = "gpt-5-mini"

AGENT_SYSTEM_PROMPT = """
You are the AI decision agent for an unmanned-store environment-control system.
Given one detection event, decide whether the physical environment actuator should receive ALERT or NORMAL.

Use ALERT when the event indicates customer discomfort, heat/sweat discomfort, occlusion, abnormal behavior,
or unsafe environment state with confidence >= 0.70.
Use NORMAL only when confidence is low or no environment-control action is needed.

Return JSON only:
{
  "decision": "ALERT" | "NORMAL",
  "reason": "short reason"
}
"""


def env_enabled(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_environment() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        print("[ENV] .env not found. Using safe defaults for latency demo.")
        return
    load_dotenv(env_path)


def build_mock_event(event_number: int) -> dict:
    return {
        "event_id": f"event_{event_number:04d}",
        "type": "mock_store_environment_event",
        "store_zone": "entrance_zone",
        "timestamp": time.time(),
        "confidence": 0.90,
        "source": "mock_camera_or_sensor",
    }


def mock_detection_loop(
    signal_queue: queue.Queue,
    profiler: LatencyProfiler,
    event_count: int,
    interval_sec: float,
) -> None:
    for event_number in range(1, event_count + 1):
        event = build_mock_event(event_number)
        event_id = event["event_id"]

        profiler.set_meta(event_id, "event_type", event["type"])
        profiler.set_meta(event_id, "confidence", event["confidence"])
        profiler.set_meta(event_id, "store_zone", event["store_zone"])
        profiler.mark(event_id, "detection_created")

        signal_queue.put(event)
        profiler.mark(event_id, "signal_queued")

        print(f"[MOCK DETECTION] Store event generated: {event_id}")

        if event_number < event_count:
            time.sleep(interval_sec)


def state_machine_loop(
    signal_queue: queue.Queue,
    arduino: ArduinoActuator,
    profiler: LatencyProfiler,
    stop_event: threading.Event,
    agent_model: str,
    use_openai_agent: bool,
) -> None:
    while not stop_event.is_set() or not signal_queue.empty():
        try:
            event = signal_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        event_id = event["event_id"]
        profiler.mark(event_id, "state_machine_received")
        print(f"[STATE MACHINE] Received store event: {event_id}")

        profiler.mark(event_id, "agent_decision_started")
        agent_result = decide_with_openai(event, agent_model, use_openai_agent)
        profiler.mark(event_id, "agent_decision_finished")

        decision = agent_result["decision"]
        profiler.set_meta(event_id, "agent_decision", decision)
        profiler.set_meta(event_id, "agent_reason", agent_result["reason"])
        profiler.set_meta(event_id, "agent_source", agent_result["source"])
        print(
            f"[AGENT] Decision: {decision} / {agent_result['reason']} "
            f"({agent_result['source']})"
        )

        if decision == "ALERT":
            arduino_result = arduino.alert()
        else:
            arduino_result = arduino.normal()

        profiler.set_meta(event_id, "actuator_send_started", arduino_result["send_start"])
        profiler.set_meta(event_id, "actuator_response_received", arduino_result["response_received"])
        profiler.set_meta(event_id, "actuator_round_trip_ms", arduino_result["round_trip_ms"])
        profiler.finalize(event_id)

        total_ms = profiler.records[event_id]["total_detection_to_actuator_ms"]
        actuator_ms = profiler.records[event_id]["actuator_round_trip_ms"]
        print(f"[LATENCY] {event_id} total={total_ms:.1f}ms actuator_round_trip={actuator_ms:.1f}ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure mock detection to Arduino actuator latency."
    )
    parser.add_argument("--events", type=int, default=5, help="Number of mock events to generate.")
    parser.add_argument("--interval", type=float, default=None, help="Seconds between mock events.")
    parser.add_argument(
        "--arduino-mock",
        action="store_true",
        help="Force mock Arduino mode even if Arduino is connected.",
    )
    parser.add_argument(
        "--agent-mock",
        action="store_true",
        help="Force local rule agent mode instead of calling OpenAI.",
    )
    return parser.parse_args()


def decide_with_openai(event: dict, model: str, use_openai: bool) -> dict:
    if not use_openai:
        return fallback_decision(event, "agent mock mode")

    if not os.getenv("OPENAI_API_KEY"):
        print("[AGENT] OPENAI_API_KEY is not set. Falling back to local rule decision.")
        return fallback_decision(event, "missing OpenAI API key")

    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(event, ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        decision = str(data.get("decision", "NORMAL")).strip().upper()
        if decision not in {"ALERT", "NORMAL"}:
            decision = "NORMAL"
        return {
            "decision": decision,
            "reason": data.get("reason", "OpenAI agent decision"),
            "source": f"openai:{model}",
        }
    except Exception as exc:
        print(f"[AGENT] OpenAI decision failed: {exc}")
        print("[AGENT] Falling back to local rule decision.")
        return fallback_decision(event, "OpenAI error")


def fallback_decision(event: dict, reason_prefix: str) -> dict:
    confidence = float(event.get("confidence", 0.0))
    decision = "ALERT" if confidence >= 0.7 else "NORMAL"
    return {
        "decision": decision,
        "reason": f"{reason_prefix}; confidence={confidence:.2f}",
        "source": "local-rule",
    }


def main() -> None:
    args = parse_args()
    load_environment()

    arduino_enabled = env_enabled(os.getenv("ARDUINO_ENABLED"), default=True) and not args.arduino_mock
    arduino_port = os.getenv("ARDUINO_PORT", DEFAULT_ARDUINO_PORT)
    arduino_baud = int(os.getenv("ARDUINO_BAUD", str(DEFAULT_ARDUINO_BAUD)))
    mock_camera_enabled = env_enabled(os.getenv("MOCK_CAMERA_ENABLED"), default=True)
    interval_sec = args.interval
    if interval_sec is None:
        interval_sec = float(os.getenv("MOCK_EVENT_INTERVAL_SEC", str(DEFAULT_EVENT_INTERVAL_SEC)))
    csv_path = os.getenv("LATENCY_OUTPUT_CSV", DEFAULT_CSV_PATH)
    json_path = os.getenv("LATENCY_OUTPUT_JSON", DEFAULT_JSON_PATH)
    agent_model = os.getenv(
        "LATENCY_AGENT_MODEL",
        os.getenv("ORCHESTRATOR_MODEL", os.getenv("CUSTOMER_MODEL", DEFAULT_AGENT_MODEL)),
    )
    use_openai_agent = env_enabled(os.getenv("OPENAI_AGENT_ENABLED"), default=True)
    if args.agent_mock:
        use_openai_agent = False

    if not mock_camera_enabled:
        print("[MOCK DETECTION] MOCK_CAMERA_ENABLED=0, but camera is absent. Enabling mock events.")
    if use_openai_agent:
        print(f"[AGENT] OpenAI agent enabled. model={agent_model}")
    else:
        print("[AGENT] OpenAI agent disabled. Using local rule decision.")

    signal_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    profiler = LatencyProfiler()
    arduino = ArduinoActuator(
        port=arduino_port,
        baudrate=arduino_baud,
        enabled=arduino_enabled,
    )

    state_thread = threading.Thread(
        target=state_machine_loop,
        args=(signal_queue, arduino, profiler, stop_event, agent_model, use_openai_agent),
        name="state-machine",
        daemon=True,
    )

    try:
        state_thread.start()
        mock_detection_loop(signal_queue, profiler, args.events, interval_sec)

        while len(profiler.completed_event_ids) < args.events:
            time.sleep(0.05)
    finally:
        stop_event.set()
        state_thread.join(timeout=2.0)
        print("[ACTUATOR] Sending NORMAL before shutdown")
        arduino.normal()
        arduino.close()

    profiler.save_csv(csv_path)
    profiler.save_json(json_path)
    print(f"[LATENCY] CSV saved: {csv_path}")
    print(f"[LATENCY] JSON saved: {json_path}")
    profiler.print_summary()


if __name__ == "__main__":
    main()
