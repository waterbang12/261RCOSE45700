import os
import cv2
import base64
import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI

client = OpenAI()

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o-mini")


class DetectionType(Enum):
    SWEAT_WIPING = "sweat_wiping"
    THEFT = "theft"
    PROPERTY_DAMAGE = "property_damage"
    FALL_EMERGENCY = "fall_emergency"


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AnalysisResult:
    detection_type: DetectionType
    detected: bool
    confidence: float
    severity: Severity
    evidence: str
    action_required: str


# ── 감지 유형별 설정 ────────────────────────────────────────────────────────────

_CONFIGS: dict[DetectionType, dict] = {

    DetectionType.SWEAT_WIPING: {
        "prompt": """
You are a visual event detector for an unmanned indoor screen golf facility.

Look at the sequence of images.
Determine whether a person is wiping sweat from their face, forehead, neck, or head.

Only answer true if there is visible evidence of wiping:
- hand, towel, tissue, sleeve, or cloth touching/wiping face/forehead/neck/head
- repeated or plausible wiping motion across frames

Do not answer true only because the person looks warm.
Do not infer emotion.
Return JSON only.
""",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "person_visible": {"type": "boolean"},
                "detected": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string"},
                "action_required": {"type": "string"},
            },
            "required": ["person_visible", "detected", "confidence", "evidence", "action_required"],
        },
        "schema_name": "sweat_wiping_detection",
        "severity": Severity.LOW,
    },

    DetectionType.THEFT: {
        "prompt": """
You are a security monitor for an unmanned indoor screen golf facility.

Look at the sequence of images.
Determine whether a person appears to be stealing or attempting to steal equipment,
personal belongings, or facility property.

Only answer true if there is visible evidence such as:
- person picking up or carrying equipment toward an exit
- person accessing storage areas, cash boxes, or restricted zones
- concealing objects inside clothing or bags
- suspicious repeated approach to unattended items

Do not answer true for normal use of golf equipment.
Return JSON only.
""",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "person_visible": {"type": "boolean"},
                "detected": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string"},
                "action_required": {"type": "string"},
            },
            "required": ["person_visible", "detected", "confidence", "evidence", "action_required"],
        },
        "schema_name": "theft_detection",
        "severity": Severity.HIGH,
    },

    DetectionType.PROPERTY_DAMAGE: {
        "prompt": """
You are a security monitor for an unmanned indoor screen golf facility.

Look at the sequence of images.
Determine whether a person is damaging facility property such as:
- striking screens, walls, or fixtures with a golf club
- throwing objects at equipment
- vandalism or forceful impact on any facility surface

Only answer true if there is clear evidence of impact or deliberate damage.
Do not answer true for normal golf swings toward the screen.
Return JSON only.
""",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "person_visible": {"type": "boolean"},
                "detected": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string"},
                "action_required": {"type": "string"},
            },
            "required": ["person_visible", "detected", "confidence", "evidence", "action_required"],
        },
        "schema_name": "property_damage_detection",
        "severity": Severity.HIGH,
    },

    DetectionType.FALL_EMERGENCY: {
        "prompt": """
You are a safety monitor for an unmanned indoor screen golf facility.

Look at the sequence of images.
Determine whether a person has fallen or is in a potential emergency situation:
- person lying on the floor unexpectedly
- person collapsed or slumped against a wall
- sudden drop from standing position visible across frames

Only answer true if the posture appears involuntary or distressed.
Do not answer true if the person appears to be intentionally crouching or sitting.
Return JSON only.
""",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "person_visible": {"type": "boolean"},
                "detected": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "evidence": {"type": "string"},
                "action_required": {"type": "string"},
            },
            "required": ["person_visible", "detected", "confidence", "evidence", "action_required"],
        },
        "schema_name": "fall_emergency_detection",
        "severity": Severity.HIGH,
    },
}


# ── 유틸 ────────────────────────────────────────────────────────────────────────

def encode_frame(frame) -> str:
    frame = cv2.resize(frame, (640, 360))
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
    if not ok:
        raise RuntimeError("Frame encoding failed")
    return base64.b64encode(buffer).decode("utf-8")


def _build_input(prompt: str, encoded_frames: list[str]) -> list[dict]:
    content = [{"type": "input_text", "text": prompt}]
    for img_b64 in encoded_frames:
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{img_b64}",
        })
    return [{"role": "user", "content": content}]


# ── 핵심 분석 함수 ───────────────────────────────────────────────────────────────

async def analyze(
    frames: list,
    detection_type: DetectionType,
    confidence_threshold: float = 0.70,
) -> AnalysisResult:
    """
    frames: OpenCV에서 받은 프레임 리스트 (numpy array)
    detection_type: 감지 유형
    confidence_threshold: 이 값 미만이면 detected=False로 처리
    """
    cfg = _CONFIGS[detection_type]
    encoded = [encode_frame(f) for f in frames]

    raw = await asyncio.to_thread(
        client.responses.create,
        model=VISION_MODEL,
        input=_build_input(cfg["prompt"], encoded),
        text={
            "format": {
                "type": "json_schema",
                "name": cfg["schema_name"],
                "strict": True,
                "schema": cfg["schema"],
            }
        },
        max_output_tokens=200,
        store=False,
    )

    data = json.loads(raw.output_text)

    # confidence 임계값 미달이면 감지 안 된 것으로 처리
    detected = data["detected"] and data["confidence"] >= confidence_threshold

    return AnalysisResult(
        detection_type=detection_type,
        detected=detected,
        confidence=data["confidence"],
        severity=cfg["severity"],
        evidence=data["evidence"],
        action_required=data["action_required"],
    )
