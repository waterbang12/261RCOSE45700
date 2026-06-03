import csv
import json
import time
from statistics import mean


class LatencyProfiler:
    """Collect per-event timing checkpoints for the control pipeline."""

    CHECKPOINTS = [
        "detection_created",
        "signal_queued",
        "state_machine_received",
        "agent_decision_started",
        "agent_decision_finished",
        "actuator_send_started",
        "actuator_response_received",
    ]

    FIELDS = [
        "event_id",
        "event_type",
        "confidence",
        "store_zone",
        *CHECKPOINTS,
        "agent_decision",
        "agent_reason",
        "agent_source",
        "total_detection_to_actuator_ms",
        "detection_to_state_machine_ms",
        "state_machine_to_agent_ms",
        "agent_decision_ms",
        "actuator_round_trip_ms",
    ]

    def __init__(self) -> None:
        self.records: dict[str, dict] = {}
        self.completed_event_ids: list[str] = []

    def _record(self, event_id: str) -> dict:
        if event_id not in self.records:
            self.records[event_id] = {"event_id": event_id}
        return self.records[event_id]

    def mark(self, event_id: str, checkpoint_name: str) -> float:
        if checkpoint_name not in self.CHECKPOINTS:
            raise ValueError(f"Unknown checkpoint: {checkpoint_name}")
        timestamp = time.perf_counter()
        self._record(event_id)[checkpoint_name] = timestamp
        return timestamp

    def set_meta(self, event_id: str, key: str, value) -> None:
        self._record(event_id)[key] = value

    def finalize(self, event_id: str) -> dict:
        record = self._record(event_id)

        self._set_delta(
            record,
            "total_detection_to_actuator_ms",
            "detection_created",
            "actuator_response_received",
        )
        self._set_delta(
            record,
            "detection_to_state_machine_ms",
            "detection_created",
            "state_machine_received",
        )
        self._set_delta(
            record,
            "state_machine_to_agent_ms",
            "state_machine_received",
            "agent_decision_started",
        )
        self._set_delta(
            record,
            "agent_decision_ms",
            "agent_decision_started",
            "agent_decision_finished",
        )

        if "actuator_round_trip_ms" not in record:
            self._set_delta(
                record,
                "actuator_round_trip_ms",
                "actuator_send_started",
                "actuator_response_received",
            )

        if event_id not in self.completed_event_ids:
            self.completed_event_ids.append(event_id)
        return record

    def save_csv(self, path: str) -> None:
        rows = [self.records[event_id] for event_id in self.completed_event_ids]
        with open(path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def save_json(self, path: str) -> None:
        rows = [self.records[event_id] for event_id in self.completed_event_ids]
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(rows, json_file, indent=2, ensure_ascii=False)

    def print_summary(self) -> None:
        rows = [self.records[event_id] for event_id in self.completed_event_ids]
        if not rows:
            print("[LATENCY SUMMARY] No completed events")
            return

        totals = [row["total_detection_to_actuator_ms"] for row in rows]
        actuator_times = [row["actuator_round_trip_ms"] for row in rows]

        print("\n[LATENCY SUMMARY]")
        print(f"events: {len(rows)}")
        print(f"average total latency: {mean(totals):.1f}ms")
        print(f"min total latency: {min(totals):.1f}ms")
        print(f"max total latency: {max(totals):.1f}ms")
        print(f"average actuator round trip: {mean(actuator_times):.1f}ms")
        print()
        print("event_id      total_ms  queue_ms  agent_ms  actuator_ms")
        print("------------  --------  --------  --------  -----------")
        for row in rows:
            print(
                f"{row['event_id']:<12}  "
                f"{row['total_detection_to_actuator_ms']:>8.1f}  "
                f"{row['detection_to_state_machine_ms']:>8.1f}  "
                f"{row['agent_decision_ms']:>8.1f}  "
                f"{row['actuator_round_trip_ms']:>11.1f}"
            )

    @staticmethod
    def _set_delta(record: dict, output_key: str, start_key: str, end_key: str) -> None:
        start = record.get(start_key)
        end = record.get(end_key)
        if start is None or end is None:
            record[output_key] = 0.0
        else:
            record[output_key] = (end - start) * 1000
