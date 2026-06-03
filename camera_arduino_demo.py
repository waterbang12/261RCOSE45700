"""Tapo TC70 camera demo with Arduino USB serial actuator control."""

import os
import queue
import sys
import threading
import time

import cv2
from dotenv import load_dotenv

from arduino_actuator import ArduinoActuator

FRAME_SIZE = (640, 360)
FRAME_QUEUE_SIZE = 2
WINDOW_NAME = "Tapo TC70 Arduino Demo"
ACTUATOR_COOLDOWN_SEC = 3


def camera_capture_loop(rtsp_url: str, frame_queue: queue.Queue, stop_event: threading.Event) -> None:
    """Read RTSP frames, resize them, and keep only the latest frames."""
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[CAMERA] Failed to open RTSP stream.")
        print("[CAMERA] Check TAPO_RTSP_URL in .env (username, password, camera IP, stream2).")
        stop_event.set()
        return

    print("[CAMERA] Stream opened")

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[CAMERA] Failed to read frame, retrying...")
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, FRAME_SIZE)

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                frame_queue.put_nowait(frame)
            except queue.Full:
                pass
    finally:
        cap.release()
        print("[CAMERA] Stream released")


def state_machine_loop(
    signal_queue: queue.Queue,
    arduino: ArduinoActuator,
    stop_event: threading.Event,
) -> None:
    """Receive mock AI events and control the Arduino actuator."""
    last_alert_time = 0.0

    while not stop_event.is_set():
        try:
            event = signal_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        print(f"[STATE MACHINE] Received event: {event}")

        confidence = float(event.get("confidence", 0.0))
        if confidence < 0.7:
            print("[STATE MACHINE] Confidence below threshold, ignoring event")
            continue

        print("[AGENT] Decision: environment cooling / alert action required")

        now = time.time()
        if now - last_alert_time < ACTUATOR_COOLDOWN_SEC:
            print("[ACTUATOR] Cooldown active, skipping duplicate ALERT")
            continue

        print("[ACTUATOR] Sending ALERT to Arduino")
        arduino.alert()
        last_alert_time = now


def console_control_loop(control_queue: queue.Queue, stop_event: threading.Event) -> None:
    """Allow terminal controls when the OpenCV window is not focused."""
    while not stop_event.is_set():
        try:
            command = input().strip().lower()
        except EOFError:
            break
        except KeyboardInterrupt:
            control_queue.put("q")
            break

        if command in {"e", "r", "s", "q"}:
            control_queue.put(command)
        elif command:
            print("[CONTROL] Unknown command. Use e, r, s, or q.")


def _env_enabled(value: str | None) -> bool:
    return (value or "1").strip().lower() in {"1", "true", "yes", "on"}


def push_mock_event(signal_queue: queue.Queue) -> None:
    event = {
        "type": "mock_occlusion_or_sweat_event",
        "camera_id": "bay_1",
        "timestamp": time.time(),
        "confidence": 0.90,
        "source": "tapo_tc70",
    }
    signal_queue.put(event)
    print(f"[DETECTION] Event pushed: {event}")


def main() -> None:
    load_dotenv()

    rtsp_url = os.getenv("TAPO_RTSP_URL")
    arduino_enabled = _env_enabled(os.getenv("ARDUINO_ENABLED"))
    arduino_port = os.getenv("ARDUINO_PORT", "COM3")
    arduino_baud = int(os.getenv("ARDUINO_BAUD", "9600"))

    if not rtsp_url:
        print("[CAMERA] TAPO_RTSP_URL is not set.")
        print("[CAMERA] Copy .env.example to .env and fill in your camera RTSP URL.")
        sys.exit(1)

    arduino = ArduinoActuator(
        port=arduino_port,
        baudrate=arduino_baud,
        enabled=arduino_enabled,
    )

    frame_queue: queue.Queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    signal_queue: queue.Queue = queue.Queue()
    control_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    camera_thread = threading.Thread(
        target=camera_capture_loop,
        args=(rtsp_url, frame_queue, stop_event),
        name="camera-capture",
        daemon=True,
    )
    state_thread = threading.Thread(
        target=state_machine_loop,
        args=(signal_queue, arduino, stop_event),
        name="state-machine",
        daemon=True,
    )
    console_thread = threading.Thread(
        target=console_control_loop,
        args=(control_queue, stop_event),
        name="console-control",
        daemon=True,
    )

    camera_thread.start()
    state_thread.start()
    console_thread.start()

    time.sleep(1.0)
    if stop_event.is_set():
        arduino.close()
        camera_thread.join(timeout=2.0)
        sys.exit(1)

    print("[CAMERA] Controls in camera window: e = event, r = reset, s = screenshot, q = quit")
    print("[TERMINAL] You can also type e/r/s/q then press Enter here.")

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                try:
                    command = control_queue.get_nowait()
                except queue.Empty:
                    command = None

                if command == "q":
                    break
                if command == "e":
                    push_mock_event(signal_queue)
                elif command == "r":
                    arduino.normal()
                    print("[ACTUATOR] Reset to NORMAL.")
                elif command == "s":
                    print("[CAMERA] No frame available for screenshot yet")

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            command = None

            try:
                command = control_queue.get_nowait()
            except queue.Empty:
                pass

            if key == ord("q") or command == "q":
                break
            if key == ord("e") or command == "e":
                push_mock_event(signal_queue)
            elif key == ord("r") or command == "r":
                arduino.normal()
                print("[ACTUATOR] Reset to NORMAL.")
            elif key == ord("s") or command == "s":
                cv2.imwrite("tapo_demo_frame.jpg", frame)
                print("[CAMERA] Screenshot saved as tapo_demo_frame.jpg")
    finally:
        print("[ACTUATOR] Sending NORMAL before shutdown")
        arduino.normal()
        stop_event.set()
        camera_thread.join(timeout=2.0)
        state_thread.join(timeout=2.0)
        arduino.close()
        cv2.destroyAllWindows()
        print("[CAMERA] Arduino demo stopped")


if __name__ == "__main__":
    main()
