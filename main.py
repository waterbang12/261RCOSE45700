"""
무인 매장 LLM 모듈 진입점.

카메라 소스 매핑:
  - int  → USB 웹캠 인덱스
  - str  → RTSP URL (Tapo TC70 등 IP 카메라)

TAPO_RTSP_URL이 .env에 있으면 1번 구역에 RTSP를 사용한다.
카메라가 없는 구역은 ZONE_CAMERA_SOURCES에서 제외하면 bridge가 실행되지 않는다.
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from db.models import init_db
from llm_module.state_machine import ZoneStateMachine, TriggerSignals
from opencv.bridge import run as cv_run, CameraSource

ZONE_IDS = ["1번 구역", "2번 구역", "3번 구역"]
CLOSE_HOUR = int(os.getenv("CLOSE_HOUR", 22))

frame_queues: dict[str, asyncio.Queue] = {}
signal_queues: dict[str, asyncio.Queue] = {}


def _build_zone_camera_sources() -> dict[str, CameraSource]:
    """Build per-zone camera sources from environment variables."""
    sources: dict[str, CameraSource] = {}

    tapo_rtsp = os.getenv("TAPO_RTSP_URL")
    if tapo_rtsp:
        sources["1번 구역"] = tapo_rtsp
        print("[MAIN] Tapo RTSP configured for 1번 구역")
    else:
        sources["1번 구역"] = int(os.getenv("ZONE1_CAMERA_INDEX", "0"))

    zone2 = os.getenv("ZONE2_CAMERA_INDEX")
    if zone2 is not None and zone2 != "":
        sources["2번 구역"] = int(zone2)

    zone3 = os.getenv("ZONE3_CAMERA_INDEX")
    if zone3 is not None and zone3 != "":
        sources["3번 구역"] = int(zone3)

    return sources


async def zone_loop(zone_id: str) -> None:
    machine = ZoneStateMachine(zone_id, frame_queues[zone_id], all_zone_ids=ZONE_IDS)
    while True:
        try:
            signals: TriggerSignals = signal_queues[zone_id].get_nowait()
        except asyncio.QueueEmpty:
            signals = TriggerSignals()
        await machine.update(signals)
        await asyncio.sleep(0.05)


async def report_loop() -> None:
    import datetime
    from llm_module.graph import facility_graph
    from llm_module.state import make_report_state

    while True:
        now = datetime.datetime.now()
        if now.hour == CLOSE_HOUR and now.minute == 0:
            state = make_report_state(zone_id=ZONE_IDS[0], all_zone_ids=ZONE_IDS)
            result = await facility_graph.ainvoke(state)
            print("[DAILY REPORT]\n", result.get("report_text", ""))
            await asyncio.sleep(60)
        await asyncio.sleep(30)


async def main() -> None:
    await init_db()
    zone_camera_sources = _build_zone_camera_sources()
    print(f"[MAIN] DB initialized. Zones: {ZONE_IDS}")
    print(f"[MAIN] Active camera bridges: {list(zone_camera_sources.keys())}")

    for zone_id in ZONE_IDS:
        frame_queues[zone_id] = asyncio.Queue(maxsize=1)
        signal_queues[zone_id] = asyncio.Queue(maxsize=1)

    tasks = [zone_loop(z) for z in ZONE_IDS] + [report_loop()]

    for zone_id, camera_source in zone_camera_sources.items():
        tasks.append(
            cv_run(zone_id, frame_queues[zone_id], signal_queues[zone_id], camera_source)
        )

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
