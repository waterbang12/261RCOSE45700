"""
무인 스크린골프장 LLM 모듈 진입점.

카메라 인덱스 매핑:
  BAY_CAMERAS = {"1번 타석": 0, "2번 타석": 1, ...}
  카메라가 없는 타석은 매핑에서 제외하면 bridge가 실행되지 않는다.
"""
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from db.models import init_db
from llm_module.state_machine import BayStateMachine, TriggerSignals
from llm_module.report_generator import generate_daily_report
from opencv.bridge import run as cv_run

# ── 설정 ──────────────────────────────────────────────────────────────────────

BAY_IDS    = ["1번 타석", "2번 타석", "3번 타석"]
CLOSE_HOUR = int(os.getenv("CLOSE_HOUR", 22))

# 타석별 카메라 인덱스 (없는 타석은 제외)
BAY_CAMERAS: dict[str, int] = {
    "1번 타석": 0,
    "2번 타석": 1,
    "3번 타석": 2,
}

# ── 타석별 큐 ─────────────────────────────────────────────────────────────────

frame_queues:  dict[str, asyncio.Queue] = {}
signal_queues: dict[str, asyncio.Queue] = {}


# ── 루프 ──────────────────────────────────────────────────────────────────────

async def bay_loop(bay_id: str) -> None:
    machine = BayStateMachine(bay_id, frame_queues[bay_id])
    while True:
        try:
            signals: TriggerSignals = signal_queues[bay_id].get_nowait()
        except asyncio.QueueEmpty:
            signals = TriggerSignals()
        await machine.update(signals)
        await asyncio.sleep(0.05)


async def report_loop() -> None:
    import datetime
    while True:
        now = datetime.datetime.now()
        if now.hour == CLOSE_HOUR and now.minute == 0:
            report = await generate_daily_report(BAY_IDS)
            print("[DAILY REPORT]\n", report)
            await asyncio.sleep(60)
        await asyncio.sleep(30)


async def main() -> None:
    await init_db()
    print(f"[MAIN] DB 초기화 완료. 타석: {BAY_IDS}")

    for bay_id in BAY_IDS:
        frame_queues[bay_id]  = asyncio.Queue(maxsize=1)
        signal_queues[bay_id] = asyncio.Queue(maxsize=1)

    tasks = [bay_loop(b) for b in BAY_IDS] + [report_loop()]

    # OpenCV 브릿지 — 카메라가 연결된 타석만 실행
    for bay_id, cam_idx in BAY_CAMERAS.items():
        tasks.append(
            cv_run(bay_id, frame_queues[bay_id], signal_queues[bay_id], cam_idx)
        )

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
