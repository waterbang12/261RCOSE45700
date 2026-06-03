import time

try:
    import serial
except ImportError:
    serial = None


class ArduinoActuator:
    """Small USB serial wrapper for the Arduino demo actuator."""

    def __init__(self, port: str, baudrate: int = 9600, enabled: bool = True):
        self.port = port
        self.baudrate = baudrate
        self.enabled = enabled
        self.mock_mode = not enabled
        self.serial_conn = None

        if not enabled:
            print("[ARDUINO] Disabled by configuration. Using mock mode.")
            return

        if serial is None:
            print("[ARDUINO] pyserial is not installed. Using mock mode.")
            print("[ARDUINO] Run: pip install pyserial")
            self.mock_mode = True
            return

        try:
            self.serial_conn = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            print(f"[ARDUINO] Connected on {port} at {baudrate} baud")

            ready_message = self.serial_conn.readline().decode(errors="ignore").strip()
            if ready_message:
                print(f"[ARDUINO READY] {ready_message}")
        except Exception as exc:
            print(f"[ARDUINO] Failed to connect on {port} at {baudrate} baud: {exc}")
            print("[ARDUINO] Continuing in mock mode.")
            self.mock_mode = True
            self.serial_conn = None

    def send(self, command: str) -> dict:
        command = command.strip().upper()
        send_start = time.perf_counter()
        print(f"[ARDUINO SEND] {command}")

        if self.mock_mode or self.serial_conn is None:
            print(f"[ARDUINO MOCK] {command}")
            response_received = time.perf_counter()
            return {
                "command": command,
                "mock": True,
                "send_start": send_start,
                "response_received": response_received,
                "round_trip_ms": (response_received - send_start) * 1000,
                "response": f"MOCK {command}",
            }

        try:
            self.serial_conn.write(f"{command}\n".encode("utf-8"))
            self.serial_conn.flush()

            response = self.serial_conn.readline().decode(errors="ignore").strip()
            response_received = time.perf_counter()
            if response:
                print(f"[ARDUINO RESPONSE] {response}")
            else:
                print("[ARDUINO RESPONSE] (no response)")
            return {
                "command": command,
                "mock": False,
                "send_start": send_start,
                "response_received": response_received,
                "round_trip_ms": (response_received - send_start) * 1000,
                "response": response,
            }
        except Exception as exc:
            response_received = time.perf_counter()
            print(f"[ARDUINO] Send failed: {exc}")
            print("[ARDUINO] Switching to mock mode.")
            self.mock_mode = True
            return {
                "command": command,
                "mock": True,
                "send_start": send_start,
                "response_received": response_received,
                "round_trip_ms": (response_received - send_start) * 1000,
                "response": f"ERROR: {exc}",
            }

    def alert(self) -> dict:
        return self.send("ALERT")

    def normal(self) -> dict:
        return self.send("NORMAL")

    def close(self) -> None:
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
                print("[ARDUINO] Serial connection closed")
            except Exception as exc:
                print(f"[ARDUINO] Error while closing serial connection: {exc}")
            finally:
                self.serial_conn = None
