import os
import time
import asyncio
import httpx
from dataclasses import dataclass, field
from datetime import datetime
from llm_module.vlm_analyzer import AnalysisResult, DetectionType, Severity

# 환경변수
MANAGER_PHONE   = os.getenv("MANAGER_PHONE", "")       # 관리자 전화번호
SMS_API_URL     = os.getenv("SMS_API_URL", "")          # SMS 발송 API (Solapi 등)
SMS_API_KEY     = os.getenv("SMS_API_KEY", "")
PUSH_WEBHOOK    = os.getenv("PUSH_WEBHOOK", "")         # 앱 푸시 웹훅 URL
SENDER_PHONE    = os.getenv("SENDER_PHONE", "")         # 발신 번호


# ── 알림 등급별 행동 정의 ────────────────────────────────────────────────────────

_ALERT_POLICY: dict[Severity, dict] = {
    Severity.HIGH: {
        "sms": True,
        "push": True,
        "cooldown_sec": 60,       # 1분 내 중복 알림 방지
    },
    Severity.MEDIUM: {
        "sms": False,
        "push": True,
        "cooldown_sec": 180,
    },
    Severity.LOW: {
        "sms": False,
        "push": False,
        "cooldown_sec": 300,
    },
}

# 감지 유형별 한국어 레이블
_LABELS: dict[DetectionType, str] = {
    DetectionType.SWEAT_WIPING:     "땀 감지 (온도 조절)",
    DetectionType.THEFT:            "도난 의심",
    DetectionType.PROPERTY_DAMAGE:  "기물 파손 의심",
    DetectionType.FALL_EMERGENCY:   "쓰러짐 / 응급 상황",
}


# ── 이벤트 로그 (일일 리포트용) ──────────────────────────────────────────────────

@dataclass
class AlertEvent:
    detection_type: DetectionType
    severity: Severity
    confidence: float
    evidence: str
    timestamp: str
    alerted: bool


_event_log: list[AlertEvent] = []


# ── cooldown 추적 ─────────────────────────────────────────────────────────────────

_last_alert_time: dict[DetectionType, float] = {}


def _is_on_cooldown(detection_type: DetectionType, cooldown_sec: int) -> bool:
    last = _last_alert_time.get(detection_type, 0)
    return (time.time() - last) < cooldown_sec


def _mark_alerted(detection_type: DetectionType) -> None:
    _last_alert_time[detection_type] = time.time()


# ── 알림 발송 ─────────────────────────────────────────────────────────────────────

async def _send_sms(message: str) -> None:
    if not SMS_API_URL or not MANAGER_PHONE:
        print(f"[SMS 미설정] {message}")
        return

    payload = {
        "to": MANAGER_PHONE,
        "from": SENDER_PHONE,
        "text": message,
    }
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.post(
            SMS_API_URL,
            json=payload,
            headers={"Authorization": f"Bearer {SMS_API_KEY}"},
        )
        r.raise_for_status()
    print(f"[SMS 발송] {message}")


async def _send_push(title: str, body: str) -> None:
    if not PUSH_WEBHOOK:
        print(f"[PUSH 미설정] {title}: {body}")
        return

    payload = {"title": title, "body": body}
    async with httpx.AsyncClient(timeout=5.0) as c:
        r = await c.post(PUSH_WEBHOOK, json=payload)
        r.raise_for_status()
    print(f"[PUSH 발송] {title}: {body}")


# ── 메인 진입점 ───────────────────────────────────────────────────────────────────

async def handle(result: AnalysisResult, bay_id: str = "1번 타석") -> None:
    """
    vlm_analyzer.analyze()의 결과를 받아 알림 처리.
    감지되지 않은 경우는 로그만 남기고 종료.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    label = _LABELS[result.detection_type]
    policy = _ALERT_POLICY[result.severity]

    event = AlertEvent(
        detection_type=result.detection_type,
        severity=result.severity,
        confidence=result.confidence,
        evidence=result.evidence,
        timestamp=now,
        alerted=False,
    )
    _event_log.append(event)

    if not result.detected:
        return

    if _is_on_cooldown(result.detection_type, policy["cooldown_sec"]):
        print(f"[COOLDOWN] {label} - 중복 알림 방지 중")
        return

    _mark_alerted(result.detection_type)
    event.alerted = True

    message = (
        f"[무인골프장 알림] {bay_id}\n"
        f"유형: {label}\n"
        f"신뢰도: {result.confidence:.0%}\n"
        f"근거: {result.evidence}\n"
        f"시각: {now}"
    )

    tasks = []
    if policy["push"]:
        tasks.append(_send_push(f"{label} 감지", message))
    if policy["sms"]:
        tasks.append(_send_sms(message))

    await asyncio.gather(*tasks)


# ── 일일 리포트용 로그 조회 ───────────────────────────────────────────────────────

def get_event_log() -> list[AlertEvent]:
    return list(_event_log)


def clear_event_log() -> None:
    _event_log.clear()
