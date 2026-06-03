"""
VLM 호출 빈도 제어 + 감지 카운트 + cooldown 관리.
OpenCV 팀원이 trigger_signals를 넘겨주면 이 모듈이 VLM 호출 여부를 결정한다.
감지 확정 후 dispatch는 LangGraph facility_graph로 위임한다.
"""
import time
import asyncio
import os
from dataclasses import asdict
from dataclasses import dataclass, field
from collections import deque
from llm_module.vlm_analyzer import analyze, DetectionType, AnalysisResult
from db.models import log_event

INFER_INTERVAL_SEC   = 1.0   # VLM 최소 호출 간격
TRIGGER_COUNT        = 2     # 윈도우 안에서 몇 번 감지돼야 트리거
TRIGGER_WINDOW_SEC   = 10.0  # 몇 초 안에 TRIGGER_COUNT번 감지돼야 하는지
CONFIDENCE_THRESHOLD = 0.70
TRACE_CAMERA_DATA = os.getenv("TRACE_CAMERA_DATA", "1") == "1"


@dataclass
class TriggerSignals:
    """OpenCV 팀원이 채워서 넘겨주는 신호 구조체"""
    sweat_wiping:     bool = False   # 손/수건이 얼굴 근처 반복 이동
    theft:            bool = False   # 손이 상품/보관 구역 근접
    property_damage:  bool = False   # 빠른 충격성 움직임 감지
    fall_emergency:   bool = False   # 수평 자세 + 정지
    body_sway:        bool = False   # 어깨 중심점 과도한 흔들림 (BodySwayDetector)
    person_count:     int  = 0       # 구역 내 인원 수
    activity_count:   int  = 0       # 누적 활동 횟수
    temperature:      float = 0.0   # 현재 온도 (센서)
    humidity:         float = 0.0   # 현재 습도 (센서)


@dataclass
class _DetectionState:
    detection_times: deque = field(default_factory=deque)  # 감지 성공한 시각 목록
    last_infer_time: float = 0
    cooldown_until: float  = 0


class ZoneStateMachine:
    """구역 하나의 상태를 관리하는 머신. 구역마다 인스턴스 생성."""

    def __init__(self, zone_id: str, frame_queue: asyncio.Queue, all_zone_ids: list[str] | None = None):
        self.zone_id       = zone_id
        self.frame_queue   = frame_queue
        self._all_zone_ids = all_zone_ids or [zone_id]
        self._states: dict[DetectionType, _DetectionState] = {
            dt: _DetectionState() for dt in DetectionType
        }
        self._last_still_time: float = time.time()

    async def update(self, signals: TriggerSignals) -> None:
        """
        매 루프에서 호출. OpenCV 신호를 보고 필요한 DetectionType만 VLM에 보냄.
        """
        frames = await self._get_frames()
        if not frames:
            return

        mapping = {
            DetectionType.SWEAT_WIPING:    signals.sweat_wiping,
            DetectionType.THEFT:           signals.theft,
            DetectionType.PROPERTY_DAMAGE: signals.property_damage,
            DetectionType.FALL_EMERGENCY:  signals.fall_emergency or signals.body_sway,
        }

        # 센서 선제 판단: 온도/습도 임계값 초과 시 sweat_wiping 강제 활성화
        if signals.temperature >= 28 and signals.humidity >= 70:
            mapping[DetectionType.SWEAT_WIPING] = True

        tasks = [
            self._process(dt, frames, signals)
            for dt, triggered in mapping.items()
            if triggered
        ]
        if tasks:
            await asyncio.gather(*tasks)

        # 움직임 없음 안전 확인 (30분)
        if signals.activity_count == 0 and signals.person_count > 0:
            if time.time() - self._last_still_time > 1800:
                print(f"[{self.zone_id}] 장시간 정지 감지 — 안전 확인 음성 출력")
                self._last_still_time = time.time()
        else:
            self._last_still_time = time.time()

    async def _process(
        self,
        dt: DetectionType,
        frames: list,
        signals: TriggerSignals,
    ) -> None:
        state = self._states[dt]
        now   = time.time()

        if now - state.last_infer_time < INFER_INTERVAL_SEC:
            return
        state.last_infer_time = now

        if TRACE_CAMERA_DATA:
            print(
                f"[{self.zone_id}][DATA][STATE->VLM] "
                f"detection_type={dt.value} frames={len(frames)} "
                f"signals={asdict(signals)}"
            )

        try:
            result: AnalysisResult = await analyze(frames, dt, CONFIDENCE_THRESHOLD)
        except Exception as e:
            print(f"[{self.zone_id}][VLM ERROR] {dt.value}: {e}")
            return

        await log_event(
            zone_id=self.zone_id,
            event_type=dt.value,
            severity=result.severity.value,
            confidence=result.confidence,
            evidence=result.evidence,
            alerted=False,
        )

        if result.detected:
            state.detection_times.append(now)

        # TRIGGER_WINDOW_SEC 지난 감지 기록 제거
        while state.detection_times and now - state.detection_times[0] > TRIGGER_WINDOW_SEC:
            state.detection_times.popleft()

        count = len(state.detection_times)
        print(f"[{self.zone_id}][{dt.value}] conf={result.confidence:.2f} count={count}/{TRIGGER_COUNT} (최근 {TRIGGER_WINDOW_SEC:.0f}초)")

        if count >= TRIGGER_COUNT:
            if now >= state.cooldown_until:
                await self._dispatch(result, signals)
                state.cooldown_until = now + 120
            else:
                print(f"[{self.zone_id}] cooldown 중 — {dt.value} 스킵")
            state.detection_times.clear()

    async def _dispatch(self, result: AnalysisResult, signals: TriggerSignals) -> None:
        """감지 확정 후 LangGraph 그래프로 위임."""
        from llm_module.graph import facility_graph
        from llm_module.state import make_safety_state

        state = make_safety_state(
            zone_id=self.zone_id,
            all_zone_ids=self._all_zone_ids,
            analysis_result={
                "detection_type": result.detection_type.value,
                "detected": result.detected,
                "confidence": result.confidence,
                "severity": result.severity.value,
                "evidence": result.evidence,
                "action_required": result.action_required,
            },
            signals={
                "temperature": signals.temperature,
                "humidity": signals.humidity,
            },
        )
        if TRACE_CAMERA_DATA:
            print(f"[{self.zone_id}][DATA][STATE->AGENT] {state}")
        await facility_graph.ainvoke(state)

    async def _get_frames(self) -> list:
        """큐에서 최신 프레임 묶음 가져오기. 없으면 빈 리스트."""
        try:
            return self.frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            return []
