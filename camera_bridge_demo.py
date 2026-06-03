"""Tapo TC70 camera bridge demo: RTSP -> OpenCV -> frame queue -> mock event -> state machine."""

import os
import queue
import sys
import threading
import time

import cv2
from dotenv import load_dotenv

FRAME_SIZE = (640, 360)
FRAME_QUEUE_SIZE = 2
WINDOW_NAME = "Tapo TC70 Demo"


def camera_capture_loop(rtsp_url, frame_queue, stop_event):
    """Read RTSP frames, resize, and keep only the latest frames in the queue."""
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


def state_machine_loop(signal_queue, stop_event):
    """Wait for detection events and print them as state-machine input."""
    while not stop_event.is_set():
        try:
            event = signal_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        print(f"[STATE MACHINE] Received event: {event}")


def main():
    load_dotenv()
    rtsp_url = os.getenv("TAPO_RTSP_URL")

    if not rtsp_url:
        print("[CAMERA] TAPO_RTSP_URL is not set.")
        print("[CAMERA] Copy .env.example to .env and fill in your camera RTSP URL.")
        sys.exit(1)

    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
    signal_queue = queue.Queue()
    stop_event = threading.Event()

    camera_thread = threading.Thread(
        target=camera_capture_loop,
        args=(rtsp_url, frame_queue, stop_event),
        name="camera-capture",
        daemon=True,
    )
    state_thread = threading.Thread(
        target=state_machine_loop,
        args=(signal_queue, stop_event),
        name="state-machine",
        daemon=True,
    )

    camera_thread.start()
    state_thread.start()

    # Give the camera thread a moment to open the stream or fail.
    time.sleep(1.0)
    if stop_event.is_set():
        camera_thread.join(timeout=2.0)
        sys.exit(1)

    print("[CAMERA] Controls: e = simulate event, s = save screenshot, q = quit")

    try:
        while not stop_event.is_set():
            try:
                frame = frame_queue.get(timeout=0.1)
            except queue.Empty:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                continue

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("e"):
                event = {
                    "type": "mock_abnormal_event",
                    "camera_id": "camera_1",
                    "timestamp": time.time(),
                    "confidence": 0.90,
                }
                signal_queue.put(event)
                print("[DETECTION] Event pushed")
            elif key == ord("s"):
                cv2.imwrite("tapo_demo_frame.jpg", frame)
                print("[CAMERA] Screenshot saved as tapo_demo_frame.jpg")
    finally:
        stop_event.set()
        camera_thread.join(timeout=2.0)
        state_thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        print("[CAMERA] Demo stopped")


if __name__ == "__main__":
    main()
